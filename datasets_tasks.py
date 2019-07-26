import pandas as pd
import numpy as np
from sklearn.datasets import load_files

from datasets_corpus import Corpus, ZuCo
from data import _CrossValidator, _SplitDataset, Vocabulary


# instead of passing this on to a train/test splitter,
# just instantiate 2 of them because they're already given as splits.
class IMDb(Corpus):
    def __init__(self, split):
        self.dataset = load_files(
            '../data_tasks/IMDb_sentiment/{}/'.format(split),
            categories=['pos', 'neg'])

    def __len__(self):
        return len(self.dataset['data'])

    def __getitem__(self, i):
        return (self.dataset['data'][i], self.dataset['target'][i])


class CoNLL2003(Corpus):
    """
    Got the data set from:
    https://github.com/glample/tagger/tree/master/dataset

    There's also a copy here:
    https://github.com/davidsbatista/NER-datasets/tree/master/CONLL2003
    """
    def __init__(self):
        pass

    def __len__(self):
        pass

    def __getitem__(self):
        pass


class ZuCo_Task(_CrossValidator):
    def __init__(self, task='sentiment', batch_size=32, gaze_data=None,
                 et_predictor_model=None, et_predictor_vocab=None,
                 use_predictor_vocab=False, filter_vocab=False):
        self.batch_size = batch_size
        self.filter_vocab = False
        self.use_gaze = gaze_data is not None

        _zuco = ZuCo(normalize=True, task=task)
        self.sentences = _zuco.sentences
        self.sentences_et = np.array(_zuco.sentences_et)
        self.max_seq_len = max([len(s) for s in self.sentences])

        # Initialize the ET features per sentence
        if et_predictor_model and et_predictor_vocab:
            print('\nReceived ET Predictor model and vocab. Vocabulary size:',
                  len(et_predictor_vocab))
            print('Running sentences through ET predictor...')
            indexed_sentences = et_predictor_vocab.index_sentences(self.sentences)
            self.sentences_et = et_predictor_model.sentences_to_et(
                indexed_sentences=indexed_sentences,
                max_seq_len=self.max_seq_len)

        # Initialize Vocabulary object
        print('\nuse_predictor_vocab =', use_predictor_vocab)
        if use_predictor_vocab:  # assuming that et_predictor_vocab is provided
            self.vocabulary = et_predictor_vocab
            self.indexed_sentences = indexed_sentences
        else:
            self.vocabulary = Vocabulary(self.sentences, filter_vocab)
            self.indexed_sentences = self.vocabulary.index_sentences(
                self.sentences)

        self.task_num = (1 if task == 'sentiment' else
                         2 if task == 'normal' else
                         3)
        self.load_labels()
        self.num_classes = len(set(self.labels))

    def load_labels(self):
        labels_df = pd.read_csv(
            '../data/ZuCo/task_materials/sentiment_labels_task{}.csv'.format(
                self.task_num), skiprows=[1], delimiter=';')

        self.labels = labels_df.sentiment_label.values
        # some of the labels are NaNs due to incorrect formatting,
        # we have to get the real ones from the control column.
        nans = labels_df[labels_df.sentiment_label.isnull()].control
        for idx, label in nans.items():
            self.labels[idx] = label
        self.labels += 1  # (-1, 0, 1) to (0, 1, 2)

    def _get_dataset(self, indices):
        et_features = (self.sentences_et[indices]
                       if self.use_gaze else None)
        return _SplitDataset(self.max_seq_len,
                             self.indexed_sentences[indices],
                             self.labels[indices],
                             et_features=et_features)
