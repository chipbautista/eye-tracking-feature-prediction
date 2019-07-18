import re

import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import KFold

from settings import BATCH_SIZE, LOAD_GECO_FROM_FILE


class Corpus(Dataset):
    def __init__(self, normalize):
        self.normalize = normalize
        if self.normalize:
            self.normalizer = StandardScaler()
        self.load_corpus()

    def clean_str(self, string):
        """
        mostly copy pasted from Hollenstein's code...

        had to change some because it messes up the matching of the words
        in a sentence :fearful:
        (removes 'words' such as '...' that actually have ET features!)
        """
        # string = string.replace(".", "")
        string = re.sub(r'([\w ])(\.)+', r'\1', string.strip().lower())
        string = string.replace(",", "")
        string = string.replace("--", "")
        string = string.replace("''", "")
        # string = string.replace("' ", "")
        # string = string.replace(" '", " ")
        string = re.sub(r"'\s?$", '', string)
        # string = string.replace("- ", " ")
        string = re.sub(r'[?!/;*()\\`:&"]', '', string)

        string = re.sub(r"\s{2,}", " ", string)

        # remove '...' or "'" at start or end
        string = re.sub(r"(^'|'$|^\.{1,3}|\.{1,3}$)", '', string)

        # remove apostrophes that are not followed by an alphabet
        # may be unnecessary though
        # string = re.sub(r"'([^a-z])", r"\1", string)

        return string

    def normalize_et(self):
        # We keep the NaN values at first so that it doesn't mess up
        # the normalization process.
        # Let's only convert them to 0s after normalizing.
        self.sentences_et = [np.nan_to_num(self.normalizer.transform(s))
                             for s in self.sentences_et]

    def print_stats(self, arr):
        print('\n' + self.name, 'ET minimum values:', np.nanmin(arr, 1))
        print(self.name, 'ET maximum values:', np.nanmax(arr, 1))
        print_normalizer_stats(self, self.normalizer)


class ZuCo(Corpus):
    def __init__(self, normalize=True, task='sentiment'):
        self.name = 'ZuCo'

        if task == 'sentiment':
            task_code = '_SR'
        elif task == 'normal':
            task_code = '_NR'
        else:
            task_code = '_TSR'

        self.directory = '../data/ZuCo/et_features{}.npy'.format(task_code)
        self.et_features = ['nFixations', 'FFD', 'TRT', 'GD', 'GPT']
        super(ZuCo, self).__init__(normalize)

    def load_corpus(self):
        self.sentences = []
        self.sentences_et = []
        _feature_values = [[], [], [], [], []]

        sentences = np.load(self.directory, allow_pickle=True)
        for sentence in sentences:
            self.sentences.append([self.clean_str(w)
                                   for w in sentence['words']])
            features = np.array([sentence[f] for f in self.et_features])

            features[0] = np.nan_to_num(features[0])
            for i in range(5):
                _feature_values[i].extend(features.reshape(5, -1)[i])

            features = np.nanmean(features.T, axis=1)
            if self.normalize:
                self.normalizer.partial_fit(features)
            self.sentences_et.append(features)

        if self.normalize:
            self.print_stats(np.array(_feature_values))
            self.normalize_et()


class GECO(Corpus):
    """
    Task material: the novel 'The Mysterious Affair at Styles' by Agatha
    Christie. Novel has 4 parts.
    14 participants.
    5,031 sentences
    (I got to extract only 3,386, though... I may need to modify my
    method of separating each trial into sentences...)
    """
    def __init__(self, normalize=True):
        self.name = 'GECO'
        self.directory = '../data/GECO Corpus/MonolingualReadingData.xlsx'
        self.et_features = {
            'nFixations': 'WORD_FIXATION_COUNT',
            'FFD': 'WORD_FIRST_FIXATION_DURATION',
            'TRT': 'WORD_TOTAL_READING_TIME',
            'GD': 'WORD_GAZE_DURATION',
            'GPT': 'WORD_GO_PAST_TIME'
        }
        super(GECO, self).__init__(normalize)

    def load_corpus(self):
        if LOAD_GECO_FROM_FILE:
            print('GECO is loaded from file.')
            self.sentences, self.sentences_et = np.load(
                '../data/GECO Corpus/pre-extracted.npy', allow_pickle=True)
            return

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

                    features_mean = np.nanmean(features, axis=0)
                    if np.nan_to_num(features).any():
                        for i in range(5):
                            _feature_values[i].extend(features.T[i])
                        if self.normalize:
                            self.normalizer.partial_fit(features_mean)

                    sentence_et.append(features_mean)

                    # consider this a complete sentence and append
                    # to the final list.
                    if (word.endswith('.') and
                            word.upper().lower()[-4:] not in [
                                'mrs.', ' mr.', ' dr.']):
                        self.sentences.append([self.clean_str(w)
                                               for w in sentence_words])
                        self.sentences_et.append(np.array(sentence_et))
                        sentence_words = []
                        sentence_et = []

        if self.normalize:
            self.print_stats(np.array(_feature_values))
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
    def __init__(self, normalize=True):
        self.name = 'PROVO'
        self.directory = '../data/PROVO Corpus/Provo_Corpus-Eyetracking_Data.csv'
        self.et_features = {
            'nFixations': 'IA_FIXATION_COUNT',
            'FFD': 'IA_FIRST_FIXATION_DURATION',
            'TRT': 'IA_DWELL_TIME',
            'GD': 'IA_FIRST_RUN_DWELL_TIME',
            'GPT': 'IA_REGRESSION_PATH_DURATION'
        }
        super(PROVO, self).__init__(normalize)

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

                    features = np.nanmean(features, axis=0)
                    if self.normalize:
                        self.normalizer.partial_fit(features)

                    # i think this next line triggers the
                    # TO-DO: add code that doesn't do this if the values
                    # are none. To save on RuntimeWarning outputs...?
                    sentence_et.append(features)
                self.sentences_et.append(np.array(sentence_et))

        if self.normalize:
            self.print_stats(np.array(_feature_values))
            self.normalize_et()


class UCL(Corpus):
    # txt
    pass


corpus_classes = {
    'ZuCo': ZuCo,
    'PROVO': PROVO,
    'GECO': GECO
}


class CorpusAggregator(Dataset):
    def __init__(self, corpus_list, normalize=False):
        print('Corpuses to use:', corpus_list, 'Loading...')

        normalize_aggregate = (normalize is not False)

        self.corpuses = {}
        self.sentences = []
        self.et_targets = []

        # instantiate the classes
        for corpus in corpus_list:
            corpus_ = corpus_classes[corpus](not normalize_aggregate)
            self.sentences.extend(corpus_.sentences)
            self.et_targets.extend(corpus_.sentences_et)
            self.corpuses[corpus] = corpus_

        # zuco_2 = ZuCo(not normalize_aggregate, 'normal')
        # self.sentences.extend(zuco_2.sentences)
        # self.et_targets.extend(zuco_2.sentences_et)

        # zuco_3 = ZuCo(not normalize_aggregate, 'task')
        # self.sentences.extend(zuco_3.sentences)
        # self.et_targets.extend(zuco_3.sentences_et)

        self.max_seq_len = max([len(s) for s in self.sentences])
        vocab = set(np.hstack(self.sentences))
        self.vocabulary = dict(zip(vocab, range(1, len(vocab) + 1)))
        self.vocabulary.update({'': 0})
        print('Num of words in vocabulary:', len(self.vocabulary))

        if normalize_aggregate:
            self.normalizer = StandardScaler()

            # ugly way to "flatten" the values into shape (N, 5) though :(
            feature_values = []
            for sent_et in self.et_targets:
                feature_values.extend(sent_et)
            self.normalizer.fit(np.array(feature_values))

            # a bit inefficient to do this but oh well...
            self.et_targets = [np.nan_to_num(s) for s in self.et_targets]
            self.et_targets_original = np.copy(self.et_targets)
            self.et_targets = [self.normalizer.transform(s)
                               for s in self.et_targets]
            print_normalizer_stats(self, self.normalizer)

        self.indexed_sentences = np.array([
            [self.vocabulary[w] for w in sentence]
            for sentence in self.sentences])
        self.et_targets = np.array(self.et_targets)

    def split_cross_val(self, num_folds=10):
        cv = KFold(num_folds, shuffle=True, random_state=111)
        splitter = cv.split(np.zeros(len(self.sentences)))
        for train_indices, test_indices in splitter:
            yield (self._get_dataloader(train_indices),
                   self._get_dataloader(test_indices, False))

    def _get_dataloader(self, indices, train=True):
        batch_size = BATCH_SIZE if train else len(indices)
        indices = np.array(indices)
        dataset = _SplitDataset(self.max_seq_len,
                                self.indexed_sentences[indices],
                                self.et_targets[indices],
                                self.et_targets_original[indices])
        return torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True)

    def __len__(self):
        return len(self.sentences)


class _SplitDataset(Dataset):
    """Send the train/test indices here and use as input to DataLoader."""
    def __init__(self, max_seq_len, indexed_sentences,
                 targets, targets_original):
        self.max_seq_len = max_seq_len
        self.indexed_sentences = indexed_sentences
        self.targets = targets
        self.targets_original = targets_original

    def __len__(self):
        return len(self.indexed_sentences)

    def __getitem__(self, i):
        missing_dims = self.max_seq_len - len(self.indexed_sentences[i])
        sentence = F.pad(torch.Tensor(self.indexed_sentences[i]),
                         (0, missing_dims))
        et_target = F.pad(torch.Tensor(self.targets[i]),
                          (0, 0, 0, missing_dims))
        et_target_original = F.pad(torch.Tensor(self.targets_original[i]),
                                   (0, 0, 0, missing_dims))
        return sentence, et_target, et_target_original


def print_normalizer_stats(caller, normalizer):
    print('\n--- {} {} Normalizer Stats ---'.format(
        caller.__class__.__name__, normalizer.__class__.__name__))
    if normalizer.__class__.__name__ == 'MinMaxScaler':
        print('min_:', normalizer.min_)
        print('scale_:', normalizer.scale_)
        print('data_min_:', normalizer.data_min_)
        print('data_max_:', normalizer.data_max_)
        print('data_range_:', normalizer.data_range_)
    else:
        print('var:', normalizer.var_)
        print('mean:', normalizer.mean_)
