import re

import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import KFold

from settings import BATCH_SIZE


class Corpus(Dataset):
    def __init__(self):
        self.normalizer = MinMaxScaler(feature_range=(-1, 1))
        self.load_corpus()

    def clean_str(self, string):
        """
        mostly copy pasted from Hollenstein's code...

        had to change some because it messes up the matching of the words
        in a sentence :fearful:
        (removes 'words' such as '...' that actually have ET features!)
        """
        # string = string.replace(".", "")
        string = re.sub(r'([\w ])(\.)+', r'\1', string)
        string = string.replace(",", "")
        # string = string.replace("--", "")
        string = string.replace("`", "")
        string = string.replace("''", "")
        string = string.replace('"', "")
        string = string.replace("' ", " ")
        # string = string.replace(" '", " ")
        string = re.sub(r"'\s?$", '', string)
        # string = string.replace("*", "")
        string = string.replace("\\", "")
        string = string.replace(";", "")
        # string = string.replace("- ", " ")
        string = string.replace("/", "-")
        string = string.replace("!", "")
        string = string.replace("?", "")

        string = re.sub(r"\s{2,}", " ", string)

        return string.strip().lower()

    def normalize_et(self):
        # We keep the NaN values at first so that it doesn't mess up
        # the normalization process.
        # Let's only convert them to 0s after normalizing.
        self.sentences_et = [np.nan_to_num(self.normalizer.transform(s))
                             for s in self.sentences_et]

    def print_stats(self, arr):
        print('\n' + self.name, 'ET minimum values:', np.nanmin(arr, 1))
        print(self.name, 'ET maximum values:', np.nanmax(arr, 1))

        # for StandardScaler:
        # print('Normalizer var:', self.normalizer.var_)
        # print('Normalizer mean:', self.normalizer.mean_)

        # for MinMaxScaler:
        print('Normalizer min_:', self.normalizer.min_)
        print('Normalizer scale_:', self.normalizer.scale_)
        print('Normalizer data_min_:', self.normalizer.data_min_)
        print('Normalizer data_max_:', self.normalizer.data_max_)
        print('Normalizer data_range_:', self.normalizer.data_range_)


class ZuCo_Sentiment(Corpus):
    def __init__(self):
        self.name = 'ZuCo_Sentiment'
        self.directory = '../data/ZuCo_Sentiment/sentences'
        super(ZuCo_Sentiment, self).__init__()
        self.load_et_features()

    def load_corpus(self):
        zuco_df = pd.read_csv(
            '../data/ZuCo/task_materials/sentiment_labels_task1.csv',
            sep=';', skiprows=[1])
        self.sentences = [self.clean_str(s).split()
                          for s in zuco_df['sentence'].values]
        self.num_sentences = len(self.sentences)

    def load_et_features(self):
        self.sentences_et = []
        sentence_et_features = np.load(
            '../data/ZuCo_Sentiment/task1_sentence_features.npy',
            allow_pickle=True)

        _feature_values = [[], [], [], [], []]
        for si, sentence in enumerate(sentence_et_features):  # 400 of these
            sentence_et = []
            for word in sentence:
                features = np.array([word['nFixations'],
                                     word['FFD'],
                                     word['TRT'],
                                     word['GD'],
                                     word['GPT']])

                if not np.isnan(features).all():
                    # make NaN fixations be 0.
                    features[0] = np.nan_to_num(features[0])

                    for i in range(5):
                        _feature_values[i].extend(features[i])
                    # sklearn's scaler reads the 2nd axis as the
                    # features. The input to fit/transform should be (11, 5)
                    self.normalizer.partial_fit(features.T)
                    sentence_et.append(np.nanmean(features, axis=1))

                else:  # when a word does not have any recorded ET feature
                    sentence_et.append(np.array([np.nan] * 5))

            self.sentences_et.append(np.array(sentence_et))

        self.print_stats(np.array(_feature_values))
        self.normalize_et()


# TO-DO: Adjust sentence extraction. I should be getting
# around 5k separate sentences! Right now I'm getting around 3.3k
class GECO(Corpus):
    """
    Task material: the novel 'The Mysterious Affair at Styles' by Agatha
    Christie. Novel has 4 parts.
    14 participants.
    5,031 sentences
    (I got to extract only 3,386, though... I may need to modify my
    method of separating each trial into sentences...)
    """
    def __init__(self):
        self.name = 'GECO'
        self.directory = '../data/GECO Corpus/MonolingualReadingData.xlsx'
        self.et_features = {
            'nFixations': 'WORD_FIXATION_COUNT',
            'FFD': 'WORD_FIRST_FIXATION_DURATION',
            'TRT': 'WORD_TOTAL_READING_TIME',
            'GD': 'WORD_GAZE_DURATION',
            'GPT': 'WORD_GO_PAST_TIME'
        }
        super(GECO, self).__init__()

    def load_corpus(self):
        self.sentences = []
        self.sentences_et = []

        _feature_values = [[], [], [], [], []]
        geco_df = pd.read_excel(self.directory)
        # per part
        for part in geco_df['PART'].unique():
            _df = geco_df[geco_df['PART'] == part]

            # Per trial. There are multiple sentences per trial!
            for trial in _df['TRIAL'].unique():
                __df = _df[_df['TRIAL'] == trial]

                # need to break it down to individual sentences
                word_ids = __df['WORD_ID_WITHIN_TRIAL'].unique()
                words = __df['WORD'][:len(word_ids)].values.astype(str)

                sentence_words = []
                sentence_et = []
                for word_id, word in zip(word_ids, words):
                    sentence_words.append(word.strip())

                    ___df = __df[__df['WORD_ID_WITHIN_TRIAL'] == word_id]
                    features = ___df[self.et_features.values()].values
                    features[features == '.'] = np.NaN
                    features = features.astype(float)

                    if np.nan_to_num(features).any():
                        for i in range(5):
                            _feature_values[i].extend(features.T[i])
                        self.normalizer.partial_fit(features)

                    sentence_et.append(np.nanmean(features, axis=0))
                    if (word.endswith('.') and
                            word.lower()[-4:] not in ['mrs.', ' mr.']):
                        # consider this a complete sentence and append
                        # to the final list.
                        self.sentences.append([self.clean_str(w)
                                               for w in sentence_words])
                        self.sentences_et.append(np.array(sentence_et))
                        sentence_words = []
                        sentence_et = []

        self.print_stats(np.array(_feature_values))

        # after going through the data set, normalize.
        self.normalize_et()


class PROVO(Corpus):
    """
    Luke, S.G. & Christianson, K. (2018).
    The Provo Corpus: A Large Eye-Tracking Corpus with Predictability Ratings.
    Behavior Research Methods, 50, 826-833.

    - Info given:
    Uploaded 2017
    84 subjects
    55 paragraphs
    2,689 tokens

    - Info extracted:
    1,192 vocabulary size (cleaned words)
    134 sentences
    """
    def __init__(self):
        self.name = 'PROVO'
        self.directory = '../data/PROVO Corpus/Provo_Corpus-Eyetracking_Data.csv'
        self.et_features = {
            'nFixations': 'IA_FIXATION_COUNT',
            'FFD': 'IA_FIRST_FIXATION_DURATION',
            'TRT': 'IA_DWELL_TIME',
            'GD': 'IA_FIRST_RUN_DWELL_TIME',
            'GPT': 'IA_REGRESSION_PATH_DURATION'
        }
        super(PROVO, self).__init__()

    def load_corpus(self):  # Flow of this extraction is similar to ZuCo.
        self.sentences = []
        self.sentences_et = []

        # csv has 230413 lines!
        provo_df = pd.read_csv(self.directory)
        self.vocabulary = provo_df['Word_Cleaned'].unique()  # 1192

        _feature_values = [[], [], [], [], []]
        # extraction level 1: by text
        for text_id in provo_df['Text_ID'].unique():
            _df = provo_df[provo_df['Text_ID'] == text_id]
            # extraction level 2: by sentence
            for sent_id in _df['Sentence_Number'].unique():
                if np.isnan(sent_id):
                    continue
                __df = _df[_df['Sentence_Number'] == sent_id]
                sentence_words = list(__df['Word_Cleaned'].unique())
                self.sentences.append(sentence_words)
                # extraction level 3: by word
                sentence_et = []

                for clean_word in sentence_words:
                    ___df = __df[__df['Word_Cleaned'] == clean_word]
                    features = ___df[self.et_features.values()].values
                    # PROVO gives 0 values to TRT/DWELL TIME,
                    # but we want those to be 0.
                    features[:, 2][np.where(features[:, 2] == 0)[0]] = np.nan

                    for i in range(5):
                        _feature_values[i].extend(features.T[i])

                    self.normalizer.partial_fit(features)
                    # i think this next line triggers the 
                    # TO-DO: add code that doesn't do this if the values
                    # are none. To save on RuntimeWarning outputs...?
                    sentence_et.append(np.nanmean(features, axis=0))
                self.sentences_et.append(np.array(sentence_et))

        self.print_stats(np.array(_feature_values))
        self.normalize_et()


class NaturalStories(Corpus):
    pass


class UCL(Corpus):
    # txt
    pass


class IITB_Sentiment(Corpus):
    # csv
    pass


class IITB_Complexity(Dataset):
    # csv
    pass


corpus_classes = {
    'ZuCo': ZuCo_Sentiment,
    'PROVO': PROVO,
    'GECO': GECO
}


class CorpusAggregator(Dataset):
    def __init__(self, corpus_list):
        print('Corpuses to use:', corpus_list, 'Loading...')

        self.corpuses = {}
        self.sentences = []
        self.et_targets = []
        # instantiate the classes
        for corpus in corpus_list:
            corpus_ = corpus_classes[corpus]()
            self.sentences.extend(corpus_.sentences)
            self.et_targets.extend(corpus_.sentences_et)
            self.corpuses[corpus] = corpus_

        self.max_seq_len = max([len(s) for s in self.sentences])
        vocab = set(np.hstack(self.sentences))
        # TO-DO: Check if the number of words in vocab matches (more or less)
        # with the sentiment analysis prog!
        self.vocabulary = dict(zip(vocab, range(1, len(vocab) + 1)))
        self.vocabulary.update({'': 0})

        self.indexed_sentences = np.array([
            [self.vocabulary[w] for w in sentence]
            for sentence in self.sentences])
        self.et_targets = np.array(self.et_targets)

    def split_cross_val(self, num_folds=10):
        cv = KFold(num_folds, shuffle=True, random_state=111)
        splitter = cv.split(np.zeros(len(self.sentences)))
        for train_indices, test_indices in splitter:
            yield (self._get_dataloader(train_indices),
                   self._get_dataloader(test_indices))

    def _get_dataloader(self, indices):
        indices = np.array(indices)
        dataset = _SplitDataset(self.max_seq_len,
                                self.indexed_sentences[indices],
                                self.et_targets[indices])
        return torch.utils.data.DataLoader(
            dataset, batch_size=BATCH_SIZE, shuffle=True)

    def __len__(self):
        return len(self.sentences)


class _SplitDataset(Dataset):
    """Send the train/test indices here and use as input to DataLoader."""
    def __init__(self, max_seq_len, indexed_sentences, targets):
        self.max_seq_len = max_seq_len
        self.indexed_sentences = indexed_sentences
        self.targets = targets

    def __len__(self):
        return len(self.indexed_sentences)

    def __getitem__(self, i):
        missing_dims = self.max_seq_len - len(self.indexed_sentences[i])
        sentence = F.pad(torch.Tensor(self.indexed_sentences[i]),
                         (0, missing_dims))
        et_target = F.pad(torch.Tensor(self.targets[i]),
                          (0, 0, 0, missing_dims))
        return sentence, et_target

    # def __getitem__(self, idx):
    #     return (self.indexed_sentences[idx], self.targets[idx])
