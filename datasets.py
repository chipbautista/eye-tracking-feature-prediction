import re
from collections import Counter

import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import KFold, StratifiedKFold
from gensim.models import KeyedVectors

from settings import BATCH_SIZE, WORD_EMBED_DIM, WORD_EMBED_MODEL_DIR


np.random.seed(111)


class Corpus(Dataset):
    def __init__(self, normalize):
        self.sentences = []
        self.sentences_et = []
        self.sentences_et_original = []
        self.normalize = normalize
        if self.normalize:
            self.normalizer = RobustScaler()
        print('Initializing', self.name)
        self.load_corpus()

    def clean_str(self, string):
        """
        Note that this will always be used PER-WORD, not PER-SENTENCE.
        mostly copy pasted from Hollenstein's code...

        had to change some because it messes up the matching of the words
        in a sentence :fearful:
        (removes 'words' such as '...' that actually have ET features!)
        """
        string = re.sub(r'([a-zA-Z ])(\.)+', r'\1', string.strip())
        string = string.replace(",", "")
        string = string.replace("--", "")
        string = string.replace("-", "")
        string = string.replace("''", "")
        string = re.sub(r"'\s?$", '', string)
        string = re.sub(r'[?!/;*()\\`:&"\[\]]', '', string)

        string = re.sub(r"\s{2,}", " ", string)

        # remove '...' or "'" at start or end
        string = re.sub(r"(^'|'$|^\.{1,3}|\.{1,3}$)", '', string)

        # remove apostrophes that are not followed by an alphabet
        # may be unnecessary though
        # string = re.sub(r"'([^a-z])", r"\1", string)
        return string

    def normalize_et(self):
        # keep a copy for later evaluation
        self.sentences_et_original = [np.nan_to_num(s)
                                      for s in np.copy(self.sentences_et)]
        # We keep the NaN values at first so that it doesn't mess up
        # the normalization process.
        # Let's only convert them to 0s after normalizing.
        self.sentences_et = [np.nan_to_num(self.normalizer.transform(s))
                             for s in self.sentences_et]

    def print_stats(self, arr=None):
        if arr is not None:
            print('\n' + self.name, 'ET minimum values:', np.nanmin(arr, 1))
            print(self.name, 'ET maximum values:', np.nanmax(arr, 1))
        print_normalizer_stats(self.name, self.normalizer)

    def __len__(self):
        return len(self.sentences)


class ZuCo(Corpus):
    def __init__(self, normalize=True, task='sentiment'):
        self.name = '-'.join(['ZuCo', task])

        if task in ['sentiment', '1']:
            task_code = '_SR'
        elif task in ['normal', '2']:
            task_code = '_NR'
        else:
            task_code = '_TSR'

        self.directory = '../data/ZuCo/et_features{}.npy'.format(task_code)
        self.et_features = ['nFixations', 'FFD', 'TRT', 'GD', 'GPT']
        super(ZuCo, self).__init__(normalize)

    def load_corpus(self):
        _feature_values = [[], [], [], [], []]

        sentences = np.load(self.directory, allow_pickle=True)
        for sentence in sentences:
            self.sentences.append([self.clean_str(w)
                                   for w in sentence['words']])
            features = np.array([sentence[f] for f in self.et_features])

            features[0] = np.nan_to_num(features[0])
            for i in range(5):
                _feature_values[i].extend(features.reshape(5, -1)[i])

            # warning: this nanmean will produce RuntimeWarnings
            # because some words are not fixated on at all ( i think )
            features = np.nanmean(features.T, axis=1)
            self.sentences_et.append(features)

        if self.normalize:
            self.normalizer.fit(np.array(_feature_values).T)
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
        self.pre_extracted = '../data/GECO Corpus/pre-extracted{}.npy'
        self.et_features = {
            'nFixations': 'WORD_FIXATION_COUNT',
            'FFD': 'WORD_FIRST_FIXATION_DURATION',
            'TRT': 'WORD_TOTAL_READING_TIME',
            'GD': 'WORD_GAZE_DURATION',
            'GPT': 'WORD_GO_PAST_TIME'
        }
        super(GECO, self).__init__(normalize)

    def load_corpus(self):
        pre_extracted_dir = self.pre_extracted.format(
            '-normalized' if self.normalize else '')
        try:
            (self.sentences, self.sentences_et,
                self.sentences_et_original, self.normalizer) = np.load(
                pre_extracted_dir, allow_pickle=True)
            print('\tGECO is loaded from file.\n')
            return
        except FileNotFoundError:
            print('Pre-extracted GECO is not found in', pre_extracted_dir,
                  'Extracting it now...')

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

                    features_mean = np.nanmean(features, axis=0)
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
            self.normalizer.fit(np.array(_feature_values).T)
            self.print_stats(np.array(_feature_values))
            self.normalize_et()

        _normalizer = self.normalizer if self.normalize else None
        np.save(pre_extracted_dir,
                (self.sentences, self.sentences_et,
                 self.sentences_et_original, _normalizer),
                allow_pickle=True)
        print('GECO extracted sentences and sentences_et saved to:',
              pre_extracted_dir,
              'This will be automatically loaded at the next run.')


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
                    # but we want those to be NaN.
                    _nonzeros = np.where(features[:, 2] == 0)[0]
                    features[:, 2][_nonzeros] = np.nan

                    for i in range(5):
                        _feature_values[i].extend(features.T[i])

                    # i think this next line triggers the
                    # TO-DO: add code that doesn't do this if the values
                    # are none. To save on RuntimeWarning outputs...?
                    features = np.nanmean(features, axis=0)
                    sentence_et.append(features)
                self.sentences_et.append(np.array(sentence_et))

        if self.normalize:
            self.normalizer.fit(np.array(_feature_values).T)
            self.print_stats(np.array(_feature_values))
            self.normalize_et()


class UCL(Corpus):
    def __init__(self, normalize):
        self.name = 'UCL'
        self.directory = '../data/UCL Corpus/eyetracking.'
        super(UCL, self).__init__(normalize)

    def load_corpus(self):
        def _derive_nfix_and_trt(sent_nr, wp, num_subjects):
            fixations = fix_df[(fix_df.sent_nr == sent_nr) &
                               (fix_df.word_pos == wp)]
            group_by_subject = fixations.groupby('subj_nr')
            nfixations = [len(nfix)
                          for nfix in group_by_subject.groups.values()]
            trts = group_by_subject.apply(
                lambda x: x['fix_duration'].sum()).values

            # sometimes, a word is not fixated by all subjects,
            # so the data obtained in this function is incomplete.
            # have to do padding...
            pad_amount = num_subjects - len(group_by_subject)
            nfixations = np.append(nfixations, np.zeros(pad_amount))
            trts = np.append(trts, [np.nan for _ in range(pad_amount)])
            return nfixations, trts

        def _build_features(wp):
            __df = _df[_df.word_pos == wp]
            num_subjects = __df.subj_nr.unique().shape[0]
            nfixations, trts = _derive_nfix_and_trt(sent_num, wp, num_subjects)
            features = np.zeros((num_subjects, 5))

            features[:, 0] = nfixations  # nFixations
            features[:, 1] = __df['RTfirstfix'].astype(float)  # FFD
            features[:, 2] = trts  # TRT
            features[:, 3] = __df['RTfirstpass'].astype(float)  # GD
            features[:, 4] = __df['RTgopast'].astype(float)  # GPT
            return features

        _feature_values = [[], [], [], [], []]
        ucl_df = pd.read_csv(self.directory + 'RT.txt', delimiter='\t')
        fix_df = pd.read_csv(self.directory + 'fix.txt', delimiter='\t')

        for sent_num in ucl_df.sent_nr.unique():
            _df = ucl_df[ucl_df['sent_nr'] == sent_num]
            word_pos = _df.word_pos.unique()
            num_words = word_pos.shape[0]
            self.sentences.append([self.clean_str(w)
                                   for w in _df['word'].values[:num_words]])

            sentence_et = []
            for wp in word_pos:
                features = _build_features(wp)
                sentence_et.append(features)
                for i in range(5):
                    _feature_values[i].extend(features[:, i])
            sentence_et = np.array(sentence_et)
            self.sentences_et.append(np.nanmean(sentence_et, axis=1))

        if self.normalize:
            self.normalizer.fit(np.array(_feature_values).T)
            self.print_stats(np.array(_feature_values))
            self.normalize_et()


corpus_classes = {'PROVO': PROVO, 'GECO': GECO, 'UCL': UCL}


class _CrossValidator:
    def split_cross_val(self, num_folds=10, stratified=True):
        k_fold = StratifiedKFold if stratified else KFold
        k_fold = k_fold(num_folds, shuffle=True, random_state=321)
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

    def build_vocabulary(self):
        vocab, self.word_embeddings = init_word_embedding_from_word2vec(
            self.sentences, self.filter_vocab)
        self.vocabulary = dict(zip(vocab, range(len(vocab))))
        print('Num of words in vocabulary:', len(self.vocabulary))

    def __len__(self):
        return len(self.sentences)


class CorpusAggregator(_CrossValidator):
    def __init__(self, corpus_list, normalize=False, filter_vocab=True):
        self.batch_size = BATCH_SIZE
        self.normalize_aggregate = normalize
        self.filter_vocab = filter_vocab

        self.normalizers = {}
        self._index_corpus = []  # to keep track of the data point's corpus
        self.sentences = []
        self.et_targets = []
        self.et_targets_original = []

        self.build_corpus(corpus_list)
        self.build_vocabulary()

        if self.normalize_aggregate:
            self.normalize()

        self.indexed_sentences = index_sentences(
            self.sentences, self.vocabulary, self.filter_vocab)
        self.et_targets = np.array(self.et_targets)
        self.et_targets_original = np.array(self.et_targets_original)

    def build_corpus(self, corpus_list):
        print('Loading corpuses...')
        for corpus in corpus_list:
            if 'ZuCo' in corpus:
                corpus_ = ZuCo(not self.normalize_aggregate,
                               task=corpus.split('-')[-1])
            else:
                corpus_ = corpus_classes[corpus](not self.normalize_aggregate)

            self.sentences.extend(corpus_.sentences)
            self.et_targets.extend(corpus_.sentences_et)
            self.et_targets_original.extend(corpus_.sentences_et_original)
            self._index_corpus.extend([corpus] * len(corpus_))
            if not self.normalize_aggregate:
                # if not aggregately normalizing, store the corpus'
                # respective normalizers instead
                self.normalizers[corpus] = corpus_.normalizer

            long_sentences = []
            for sss in corpus_.sentences:
                if len(sss) > 60:
                    long_sentences.append(sss)
            if len(long_sentences) > 1:
                import pdb; pdb.set_trace()

        self.max_seq_len = max([len(s) for s in self.sentences])
        print('Max sentence length:', self.max_seq_len)

    def normalize(self):
        self.normalizer = StandardScaler()

        # ugly way to "flatten" the values into shape (N, 5) though :(
        feature_values = []
        for sent_et in self.et_targets:
            feature_values.extend(sent_et)
        self.normalizer.fit(np.array(feature_values))

        # a bit inefficient to do this but oh well...
        # for storing original ET features, already convert NaNs to 0
        self.et_targets_original = np.array(
            [np.nan_to_num(s) for s in np.copy(self.et_targets)])
        # for the normalized targets, normalize before converting NaNs to 0
        self.et_targets = np.array(
            [np.nan_to_num(self.normalizer.transform(s))
             for s in self.et_targets])
        print_normalizer_stats(self, self.normalizer)

    def _get_dataset(self, indices):
        return _SplitDataset(self.max_seq_len,
                             self.indexed_sentences[indices],
                             self.et_targets[indices],
                             self.et_targets_original[indices],
                             indices=indices)

    def inverse_transform(self, data_index, value):
        if self.normalize_aggregate:
            return self.normalizer.inverse_transform(value)

        corpus = self._index_corpus[data_index]
        return self.normalizers[corpus].inverse_transform(value)


class _SplitDataset(Dataset):
    """Send the train/test indices here and use as input to DataLoader."""
    def __init__(self, max_seq_len, indexed_sentences, targets,
                 targets_original=None, et_features=None, indices=None):
        self.max_seq_len = max_seq_len
        self.indexed_sentences = indexed_sentences
        self.targets = targets
        self.targets_original = targets_original
        self.et_features = et_features
        self.aggregate_indices = indices

    def __len__(self):
        return len(self.indexed_sentences)

    def __getitem__(self, i):
        missing_dims = self.max_seq_len - len(self.indexed_sentences[i])
        sentence = F.pad(torch.Tensor(self.indexed_sentences[i]),
                         (0, missing_dims))
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
            et_feature = F.pad(torch.Tensor(self.et_features[i]),
                               (0, 0, 0, missing_dims))
            return sentence, et_feature, self.targets[i]
        else:
            # used for NLP tasks without gaze data
            return sentence, self.targets[i]


def print_normalizer_stats(caller, normalizer):
    print('\n--- {} {} Normalizer Stats ---'.format(
        caller, normalizer.__class__.__name__))
    if normalizer.__class__.__name__ == 'MinMaxScaler':
        print('min_:', normalizer.min_)
        print('scale_:', normalizer.scale_)
        print('data_min_:', normalizer.data_min_)
        print('data_max_:', normalizer.data_max_)
        print('data_range_:', normalizer.data_range_)
    elif normalizer.__class__.__name__ == 'RobustScaler':
        print('center_:', normalizer.center_)
        print('scale_:', normalizer.scale_)
    else:
        print('var:', normalizer.var_)
        print('std:', np.sqrt(normalizer.var_))
        print('mean:', normalizer.mean_)


def index_sentences(sentences, vocabulary, filter):
    def _get_index(w):
        w = w if w in vocabulary else _fix_oov_word(vocabulary, w, filter=filter)
        return vocabulary[w]

    return np.array([[_get_index(w) for w in sentence]
                     for sentence in sentences])


def init_word_embedding_from_word2vec(sentences, filter_vocab=True):
    print('\nLoading pre-trained word2vec from', WORD_EMBED_MODEL_DIR)
    pretrained_w2v = KeyedVectors.load_word2vec_format(
        WORD_EMBED_MODEL_DIR, binary=True)
    print('\nDone. Will now extract embeddings for needed words.',
          'Include ALL words = {}. (filter_vocab is set to {})'.format(
              not filter_vocab, filter_vocab))

    counter = Counter(np.hstack(sentences))
    embeddings = {
        '<UNK>': np.random.uniform(-0.5, 0.5, WORD_EMBED_DIM),
        '<NUM>': np.random.uniform(-0.5, 0.5, WORD_EMBED_DIM),
        '<ENTITY>': np.random.uniform(-0.5, 0.5, WORD_EMBED_DIM)
    } if filter_vocab else {}

    oov_words = []
    for word, count in counter.items():
        if word in pretrained_w2v:
            embeddings[word] = pretrained_w2v[word]
        else:
            _word = _fix_oov_word(pretrained_w2v, word, count, filter_vocab)
            if filter_vocab and _word == '<UNK>':
                oov_words.append((word, count))
                continue
            if _word in pretrained_w2v and _word not in embeddings:
                embeddings[_word] = pretrained_w2v[_word]
            else:
                oov_words.append(word)
                embeddings[_word] = np.random.uniform(
                    -0.5, 0.5, WORD_EMBED_DIM)

    print(len(oov_words), 'words were not found in the pre-trained embedding.')
    return list(embeddings.keys()), torch.Tensor(list(embeddings.values()))


def _fix_oov_word(vocab, word, count=0, filter=False):
    if word in ["i'll", "i've", "i'm", "i'd"]:
        return word.replace('i', 'I')
    if word.lower() in vocab:
        return word.lower()

    if not filter:
        return word

    try:
        float(word)
        return '<NUM>'
    except ValueError:
        if count > 100:
            return word
        if word[0].isupper():
            return '<ENTITY>'
        return '<UNK>'
