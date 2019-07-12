import re

import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_files
from tflearn.data_utils import VocabularyProcessor


class ZuCo_Sentiment(Dataset):
    def __init__(self, args):
        self.name = 'ZuCo_Sentiment'
        self.directory = '../data/ZuCo_Sentiment/sentences'
        self.load_corpus()
        self.load_et_features()

    def load_corpus(self):
        dataset = load_files(
            container_path='../data/ZuCo_Sentiment/sentences',
            categories=['NEGATIVE', 'POSITIVE', 'NEUTRAL'],
            load_content=True, encoding='utf-8')
        self.sentences_, self.sentence_numbers, _, _, _ = dataset.values()
        self.sentence_numbers = [int(re.search(r'\d{1,3}', fname).group())
                                 for fname in dataset['filenames']]

        # adopting Hollenstein's method of building the vocab
        self.sentences = [clean_str(s) for s in self.sentences_]
        self.num_sentences = len(self.sentences)
        self.max_sentence_length = max([len(s.split())
                                        for s in self.sentences])
        self.vocab_processor = VocabularyProcessor(self.max_sentence_length)
        self.indexed_sentences = torch.LongTensor(list(
            self.vocab_processor.fit_transform(self.sentences)))
        self.vocabulary = list(
            self.vocab_processor.vocabulary_._mapping.keys())

    def load_et_features(self):
        self.sentences_et = []
        normalizer = StandardScaler()
        sentence_et_features = np.load(
            '../data/ZuCo_Sentiment/task1_sentence_features.npy',
            allow_pickle=True)

        for si, sentence in enumerate(sentence_et_features):  # 400 of these
            sentence_et = []
            for word in sentence:
                features = np.array([word['nFixations'],
                                     word['FFD'],
                                     word['TRT'],
                                     word['GD'],
                                     word['GPT']])

                if not np.all(np.isnan(features)):
                    # sklearn's StandardScaler reads the 2nd axis as the
                    # features. The input to fit/transform should be (11, 5)
                    normalizer.partial_fit(features.T)
                    sentence_et.append(np.nanmean(features, axis=1))
                else:  # when a word does not have any recorded ET feature
                    sentence_et.append(np.array([np.nan] * 5))

            self.sentences_et.append(np.array(sentence_et))

        # We keep the NaN values at first so that it doesn't mess up
        # the normalization process.
        # Let's only convert them to 0s after normalizing.
        self.sentences_et = [np.nan_to_num(normalizer.transform(s))
                             for s in self.sentences_et]

    def __getitem__(self, i):
        # should probably do the padding at the DataLoader level, not here...
        # missing_dims = self.max_sentence_length - et_features.shape[0]
        # et_features = torch.nn.functional.pad(et_features,
        #                                       (0, 0, 0, missing_dims),
        #                                       mode='constant')

        return (self.indexed_sentences[self.sentence_numbers.index(i)],
                torch.Tensor(self.sentences_et[i]))


class GECO(Dataset):
    pass


class PROVO(Dataset):
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
            'FFD': 'IA_FIRST_FIXATION_DURATION',
            'GD': 'IA_FIRST_RUN_DWELL_TIME',
            'TRT': 'IA_DWELL_TIME',
            'nFixations': 'IA_FIXATION_COUNT',
            'GPT': 'IA_REGRESSION_PATH_DURATION'
        }
        self.load_corpus()

    def load_corpus(self):  # Flow of this extraction is similar to ZuCo.
        self.sentences = []
        self.sentences_et = []
        normalizer = StandardScaler()

        # csv has 230413 lines!
        provo_df = pd.read_csv(self.directory)
        self.vocabulary = provo_df['Word_Cleaned'].unique()  # 1192

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
                    features = np.array([___df[column] for column
                                         in self.et_features.values()])

                    normalizer.partial_fit(features.T)
                    sentence_et.append(np.nanmean(features, axis=1))
                self.sentences_et.append(np.array(sentence_et))

        # after going through the data set, normalize.
        self.sentences_et = [np.nan_to_num(normalizer.transform(s))
                             for s in self.sentences_et]

    def __getitem__(self, i):
        return (self.sentences[i], self.sentences_et[i])


class NaturalStories(Dataset):
    pass


class UCL(Dataset):
    # txt
    pass


class IITB_Sentiment(Dataset):
    # csv
    pass


class IITB_Complexity(Dataset):
    # csv
    pass


def clean_str(string):
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
    string = string.replace("' ", " ")
    # string = string.replace("*", "")
    string = string.replace("\\", "")
    string = string.replace(";", "")
    # string = string.replace("- ", " ")
    string = string.replace("/", "-")
    string = string.replace("!", "")
    string = string.replace("?", "")

    string = re.sub(r"\s{2,}", " ", string)

    return string.strip().lower()
