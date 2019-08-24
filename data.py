import pickle
from collections import Counter

import torch
import torch.nn.functional as F
import numpy as np
from gensim.models import KeyedVectors
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset
# from allennlp.commands.elmo import ElmoEmbedder
from allennlp.modules.elmo import batch_to_ids
from flair.data import Sentence

from datasets_corpus import *
from model import STATIC_EMBEDDING
from settings import BATCH_SIZE, WORD_EMBED_DIM, WORD_EMBED_MODEL_DIR

torch.manual_seed(111)
np.random.seed(111)
CORPUS_CLASSES = {'PROVO': PROVO, 'GECO': GECO, 'UCL': UCL}


class _CrossValidator:
    def split_cross_val(self, num_folds=10, stratified=True):
        k_fold = StratifiedKFold if stratified else KFold
        k_fold = k_fold(num_folds, shuffle=True, random_state=111)
        splitter = k_fold.split(np.zeros(len(self.sentences)),
                                self.labels if stratified else None)
        for train_indices, test_indices in splitter:
            yield (self._get_dataloader(train_indices),
                   self._get_dataloader(test_indices, False))

    def _get_dataloader(self, indices, train=True):
        batch_size = self.batch_size if train else len(indices)
        indices = np.array(indices)
        # _get_dataset should be implemented by child classes
        dataset = self._get_dataset(indices)
        return torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True)

    def __len__(self):
        return len(self.sentences)


class CorpusAggregator(_CrossValidator):
    def __init__(self, corpus_list, corpus_normalizer, normalize_wrt_mean=False, filter_vocab=False,
                 use_word_length=False, finetune_elmo=False, static_embedding=None,
                 train_per_sample=False, minmax_aggregate=False):

        self.corpus_normalizer = corpus_normalizer
        self.batch_size = BATCH_SIZE
        self.minmax_aggregate = minmax_aggregate  # clean this up later...
        self.filter_vocab = filter_vocab
        self.use_word_length = use_word_length
        self.finetune_elmo = finetune_elmo
        self.static_embedding = static_embedding
        self.preextracted_elmo_dir = 'models/elmo_embeddings.pickle'
        self.train_per_sample = train_per_sample

        self.normalizers = {}
        self._index_corpus = []  # to keep track of the data point's corpus
        self.sentences = []
        self.sentence_word_lengths = []
        self.et_targets = []
        self.et_targets_original = []

        self._build_corpus(corpus_list, normalize_wrt_mean)

        if self.minmax_aggregate:
            self.normalizer = MinMaxScaler()
            self._scale_minmax()
        else:
            self.normalizer = None

        if self.static_embedding:
            self.indexed_sentences = np.array(self.sentences)
        else:
            self.vocabulary = Vocabulary(self.sentences, filter_vocab,
                                         self.finetune_elmo)
            self.indexed_sentences = self.vocabulary.index_sentences(
                self.sentences)

        self.et_targets = np.array(self.et_targets)
        self.et_targets_original = np.array(self.et_targets_original)

    def _build_corpus(self, corpus_list, normalize_wrt_mean):
        # do_normalization = not self.minmax_aggregate
        print('Aggregating corpuses...')
        print('Normalize with respect to mean features:', normalize_wrt_mean)
        for corpus in corpus_list:
            kwargs = {
                'normalize_wrt_mean': normalize_wrt_mean,
                'aggregate_features': not self.train_per_sample,
                'finetune_elmo': self.finetune_elmo,
                'static_embedding': self.static_embedding,
                'normalizer': self.corpus_normalizer
            }
            if 'ZuCo' in corpus:
                corpus_ = ZuCo(corpus.split('-')[-1], kwargs)
            else:
                corpus_ = CORPUS_CLASSES[corpus](kwargs)

            self.sentences.extend(corpus_.sentences)
            self.et_targets.extend(corpus_.sentences_et)
            self.et_targets_original.extend(corpus_.sentences_et_original)
            self._index_corpus.extend([corpus] * len(corpus_))
            self.normalizers[corpus] = corpus_.normalizer

        self.max_seq_len = max([len(s) for s in self.sentences])
        print('\nMax sentence length:', self.max_seq_len)

    def _scale_minmax(self):
        print('Further normalizing the corpuses\' features...')

        # ugly way to "flatten" the values into shape (N, 5) though :(
        feature_values = []
        for sent_et in self.et_targets:
            feature_values.extend(sent_et)
        self.normalizer.fit(np.array(feature_values))

        self.et_targets = np.array([self.normalizer.transform(s)
                                    for s in self.et_targets])
        print_normalizer_stats('CorpusAggregator', self.normalizer)

    def _get_dataset(self, indices):
        """
        if self.finetune_elmo:
            # sentences = [self.indexed_sentences[i].reshape(-1, 1024 * 3)
            #              for i in indices]
            sentences = [self.indexed_sentences[i].mean(0)
                         for i in indices]
            # sentences = [self.indexed_sentences[i]
            #             for i in indices]
        else:
            sentences = self.indexed_sentences[indices]
        """
        return _SplitDataset(self.max_seq_len,
                             self.indexed_sentences[indices],
                             self.et_targets[indices],
                             self.et_targets_original[indices],
                             indices=indices,
                             finetune_elmo=self.finetune_elmo,
                             static_embedding=self.static_embedding)

    def inverse_transform(self, data_index, value):
        if self.minmax_aggregate:
            value = self.normalizer.inverse_transform(value)

        # find which corpus this specific data point belongs to
        # and use that corpus' normalizer to transform it back
        corpus = self._index_corpus[data_index]
        return self.normalizers[corpus].inverse_transform(value)


class _SplitDataset(Dataset):
    """Send the train/test indices here and use as input to DataLoader."""
    def __init__(self, max_seq_len, indexed_sentences, targets,
                 targets_original=None, et_features=None,
                 indices=None, finetune_elmo=False, static_embedding=None):

        self.static_embedding = static_embedding
        self.max_seq_len = max_seq_len
        self.indexed_sentences = indexed_sentences
        self.targets = targets
        self.targets_original = targets_original
        self.et_features = et_features
        self.aggregate_indices = indices
        self.finetune_elmo = finetune_elmo

    def __len__(self):
        return len(self.indexed_sentences)

    def __getitem__(self, i):  # im sorry this method is like soup
        missing_dims = self.max_seq_len - len(self.indexed_sentences[i])

        # # PREPARE SENTENCE TENSOR
        # if self.static_embedding:
        #     sentence = self.indexed_sentences[i]  # this is already the embedding
        if self.finetune_elmo or self.static_embedding:
            sentence = F.pad(torch.Tensor(self.indexed_sentences[i]),
                             (0, 0, 0, missing_dims))
        else:
            sentence = F.pad(torch.Tensor(self.indexed_sentences[i]),
                             (0, missing_dims)).type(torch.LongTensor)

        if self.targets_original is not None:
            # used when predicting ET features
            et_target = F.pad(torch.Tensor(self.targets[i]),
                              (0, 0, 0, missing_dims))
            et_target_original = F.pad(torch.Tensor(self.targets_original[i]),
                                       (0, 0, 0, missing_dims))

            return (sentence, et_target, et_target_original,
                    self.aggregate_indices[i])

        elif self.et_features is not None:
            # used for NLP tasks with gaze data
            if isinstance(self.et_features, torch.Tensor):
                # no need to pad
                et_feature = self.et_features[i]
            else:
                et_feature = F.pad(torch.Tensor(self.et_features[i]),
                                   (0, 0, 0, missing_dims))
            return sentence, et_feature, self.targets[i]
        else:
            # used for NLP tasks without gaze data
            return sentence, self.targets[i]


class Vocabulary:
    def __init__(self, sentences, filter_vocab=False, finetune_elmo=False):
        print('\nInitializing new Vocabulary object...')
        self.filter_vocab = filter_vocab
        self.finetune_elmo = finetune_elmo

        if not self.finetune_elmo:
            # initialize from word2vec
            vocab, self.word_embeddings = self._init_word_embedding_from_word2vec(
                sentences)
            self.vocabulary = dict(zip(vocab, range(len(vocab))))
            print('Num of words in vocabulary:', len(self.vocabulary))

    def index_sentences(self, sentences):
        try:
            if self.finetune_elmo:
                return batch_to_ids(sentences)
        except AttributeError:
            # happens when we're loading the Vocab object from file and it
            # does not have finetune_elmo attribute
            pass

        return np.array([
            [self.get_index(w) for w in sentence]
            for sentence in sentences])

    def _init_word_embedding_from_word2vec(self, sentences):
        def _init_embed():
            return np.random.uniform(-0.25, 0.25, WORD_EMBED_DIM)

        print('\nLoading pre-trained word2vec from', WORD_EMBED_MODEL_DIR)
        pretrained_w2v = KeyedVectors.load_word2vec_format(
            WORD_EMBED_MODEL_DIR, binary=True)
        print('\nDone. Will now extract embeddings for needed words.',
              'Include ALL words = {}. (filter_vocab is set to {})'.format(
                  not self.filter_vocab, self.filter_vocab))

        counter = Counter(np.hstack(sentences))
        embeddings = {'<PAD>': _init_embed(), '<UNK>': _init_embed()}
        if self.filter_vocab:
            # just to make sure these will always be part of the vocab
            embeddings.update({
                '<NUM>': _init_embed(),
                '<ENTITY>': _init_embed()
            })

        oov_words = []
        for word, count in counter.items():
            _word, in_vocab = self._fix_word(pretrained_w2v, word, count)
            if in_vocab:
                embeddings[_word] = pretrained_w2v[_word]
            else:
                embeddings[_word] = _init_embed()
                if _word in [word, '<UNK>']:
                    oov_words.append(word)

        print(len(oov_words),
              'words were not found in the pre-trained embedding.')
        return list(embeddings.keys()), torch.Tensor(list(embeddings.values()))

    def _fix_word(self, vocab, word, count=0):
        """
        Accepts:
        1. `vocab` - this is either the vocab from pretrained word2vec or the
            actual vocabulary of the instance (after loading from pretrained)
        2. `word`
        3. `count` - optional, used only when building from pretrained word2vec

        Returns:
        1. the word for actual storage in the vocab
            (ex: store the lowercased version instead)
        2. Boolean value, indicating if the word to be returned is already in
            the vocab. (This functionality will only be used when building
            the vocab from the pretrained word2vec)
        """
        if word.lower() in vocab:
            return word.lower(), True

        if word in vocab:
            return word, True

        if word in ["i'll", "i've", "i'm", "i'd"]:
            return word.replace('i', 'I'), True

        if not self.filter_vocab:
            # then just put any unknown word into vocab
            return word.lower(), False

        # else, try to categorize them further
        try:
            # if float() does not raise ValueError then it's a number
            float(word)
            return '<NUM>', False
        except ValueError:
            # if the word occurs frequently then just add it to vocab
            if count > 100:
                return word, False
            # if the first letter is uppercase then it might be a proper noun
            if word and word[0].isupper():
                return '<ENTITY>', False
            # else, just return UNK tag
            return '<UNK>', False

    def get_index(self, word):
        return self.vocabulary[self._fix_word(self.vocabulary, word)[0]]

    def __getitem__(self, key):
        return self.vocabulary[key]

    def __contains__(self, key):
        return key in self.vocabulary

    def __len__(self):
        return len(self.vocabulary)
