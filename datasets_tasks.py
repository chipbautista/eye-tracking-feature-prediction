import pandas as pd
import numpy as np
from sklearn.datasets import load_files

from datasets import (Corpus, ZuCo, _CrossValidator,
                      index_sentences, _SplitDataset)


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
    def __init__(self, task='sentiment', batch_size=32, gaze_data=None):
        self.batch_size = batch_size

        _zuco = ZuCo(normalize=True, task=task)
        self.sentences = _zuco.sentences
        self.sentences_et = np.array(_zuco.sentences_et)

        self.use_gaze = (True if (gaze_data and gaze_data.lower() == 'own')
                         else False)

        self.build_vocabulary()
        self.indexed_sentences = index_sentences(
            self.sentences, self.vocabulary)

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
        et_features = None if not self.use_gaze else self.sentences_et[indices]
        return _SplitDataset(self.max_seq_len,
                             self.indexed_sentences[indices],
                             self.labels[indices],
                             et_features=et_features)
