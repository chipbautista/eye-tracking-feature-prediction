import pickle
from collections import Counter

import torch
import torch.nn.functional as F
import numpy as np
from gensim.models import KeyedVectors
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset
from allennlp.commands.elmo import ElmoEmbedder

from datasets_corpus import *
from settings import BATCH_SIZE, WORD_EMBED_DIM, WORD_EMBED_MODEL_DIR


np.random.seed(111)
corpus_classes = {'PROVO': PROVO, 'GECO': GECO, 'UCL': UCL}


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
    def __init__(self, corpus_list, normalize=False, filter_vocab=False,
                 use_word_length=False, use_elmo_embeddings=False):
        self.batch_size = BATCH_SIZE
        self.normalize_aggregate = normalize
        self.filter_vocab = filter_vocab
        self.use_word_length = use_word_length
        self.use_elmo_embeddings = use_elmo_embeddings
        self.preextracted_elmo_dir = 'models/elmo_embeddings.pickle'

        self.normalizers = {}
        self._index_corpus = []  # to keep track of the data point's corpus
        self.sentences = []
        self.sentence_word_lengths = []
        self.et_targets = []
        self.et_targets_original = []

        self._build_corpus(corpus_list)

        # IGNORE, i shouldnt be doing this
        if not self.normalize_aggregate:
            # need to normalize the word_lengths, though
            length_scaler = MinMaxScaler()
            length_scaler.fit(np.hstack(
                self.sentence_word_lengths).reshape(-1, 1))

            self.sentence_word_lengths = np.array([
                length_scaler.transform(
                    np.reshape(lengths, (-1, 1))).reshape(-1)
                for lengths in self.sentence_word_lengths])

        if self.normalize_aggregate:
            self._normalize()

        # TO-DO: This has to be done per-corpus!!!
        if self.use_elmo_embeddings:
            try:
                with open(self.preextracted_elmo_dir, 'rb') as f:
                    # TO-DO: Remove np.array(),
                    # I should already be saving a NumPy object
                    self.indexed_sentences = pickle.load(f)
            except FileNotFoundError:
                print('Pre-extracted ELMo embeddings not found.',
                      'Extracting now...')
                elmo = ElmoEmbedder()
                self.indexed_sentences = elmo.embed_sentences(self.sentences)
                # self.indexed_sentences = np.array(elmo.embed_sentences(self.sentences))

                # get only the last ELMo vector
                # (can experiment with this later)
                self.indexed_sentences = np.array([
                    sent[2, :, :] for sent in self.indexed_sentences])

                with open(self.preextracted_elmo_dir, 'wb') as f:
                    pickle.dump(self.indexed_sentences, f)
                print('Saved extracted ELMo embeddings to:',
                      self.preextracted_elmo_dir)

        else:
            self.vocabulary = Vocabulary(self.sentences, filter_vocab)
            self.indexed_sentences = self.vocabulary.index_sentences(
                self.sentences)

        self.et_targets = np.array(self.et_targets)
        self.et_targets_original = np.array(self.et_targets_original)

    def _build_corpus(self, corpus_list):
        do_normalization = True
        # do_normalization = not self.normalize_aggregate
        print('Aggregating corpuses...')
        for corpus in corpus_list:
            if 'ZuCo' in corpus:
                corpus_ = ZuCo(do_normalization,
                               task=corpus.split('-')[-1])
            else:
                corpus_ = corpus_classes[corpus](do_normalization)

            self.sentences.extend(corpus_.sentences)
            self.sentence_word_lengths.extend(corpus_.sentence_word_lengths)
            self.et_targets.extend(corpus_.sentences_et)
            self.et_targets_original.extend(corpus_.sentences_et_original)
            self._index_corpus.extend([corpus] * len(corpus_))
            # if not aggregately normalizing, store the corpus'
            # respective normalizers instead
            self.normalizers[corpus] = corpus_.normalizer

        self.sentence_word_lengths = np.array(self.sentence_word_lengths)
        self.max_seq_len = max([len(s) for s in self.sentences])
        print('\nMax sentence length:', self.max_seq_len)

    def _normalize(self):
        print('Further normalizing the corpuses\' features...')
        self.normalizer = MinMaxScaler()

        # ugly way to "flatten" the values into shape (N, 5) though :(
        feature_values = []
        for sent_et in self.et_targets:
            feature_values.extend(sent_et)
        self.normalizer.fit(np.array(feature_values))

        # a bit inefficient to do this but oh well...
        # for storing original ET features, already convert NaNs to 0
        # self.et_targets_original = np.array(
        #     [np.nan_to_num(s) for s in np.copy(self.et_targets)])
        # for the normalized targets, normalize before converting NaNs to 0
        self.et_targets = np.array(
            [np.nan_to_num(self.normalizer.transform(s))
             for s in self.et_targets])
        print_normalizer_stats('CorpusAggregator', self.normalizer)

    def _get_dataset(self, indices):
        if self.use_elmo_embeddings:
            # sentences = [self.indexed_sentences[i].reshape(-1, 1024 * 3)
            #              for i in indices]
            sentences = [self.indexed_sentences[i].mean(0)
                         for i in indices]
            # sentences = [self.indexed_sentences[i]
            #             for i in indices]
        else:
            sentences = self.indexed_sentences[indices]
        return _SplitDataset(self.max_seq_len,
                             sentences,
                             self.et_targets[indices],
                             self.et_targets_original[indices],
                             word_lengths=(self.sentence_word_lengths[indices]
                                           if self.use_word_length else None),
                             indices=indices,
                             use_elmo_embeddings=self.use_elmo_embeddings)

    def inverse_transform(self, data_index, value):
        if self.normalize_aggregate:
            # return self.normalizer.inverse_transform(value)
            value = self.normalizer.inverse_transform(value)

        # find which corpus this specific data point belongs to
        # and use that corpus' normalizer to transform it back
        corpus = self._index_corpus[data_index]
        return self.normalizers[corpus].inverse_transform(value)


class _SplitDataset(Dataset):
    """Send the train/test indices here and use as input to DataLoader."""
    def __init__(self, max_seq_len, indexed_sentences, targets,
                 targets_original=None, et_features=None, word_lengths=None,
                 indices=None, use_elmo_embeddings=False):
        self.max_seq_len = max_seq_len
        self.indexed_sentences = indexed_sentences
        self.targets = targets
        self.targets_original = targets_original
        self.et_features = et_features
        self.word_lengths = word_lengths
        self.aggregate_indices = indices
        self.use_elmo_embeddings = use_elmo_embeddings

    def __len__(self):
        return len(self.indexed_sentences)

    def __getitem__(self, i):  # im sorry this method like soup
        missing_dims = self.max_seq_len - len(self.indexed_sentences[i])
        if self.use_elmo_embeddings:
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

            if self.word_lengths is not None:

                word_lengths = F.pad(torch.Tensor(self.word_lengths[i]),
                                     (0, missing_dims))
                return (sentence, et_target, et_target_original, word_lengths,
                        self.aggregate_indices[i])

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
    def __init__(self, sentences, filter_vocab=False):
        print('\nInitializing new Vocabulary object...')
        self.filter_vocab = filter_vocab
        vocab, self.word_embeddings = self._init_word_embedding_from_word2vec(
            sentences)
        self.vocabulary = dict(zip(vocab, range(len(vocab))))
        print('Num of words in vocabulary:', len(self.vocabulary))

    def index_sentences(self, sentences):
        return np.array([
            [self.get_index(w) for w in sentence]
            for sentence in sentences
        ])

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
