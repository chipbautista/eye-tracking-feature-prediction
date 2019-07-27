import re

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


class Corpus:
    def __init__(self, normalize):
        # the following will be filled up by `load_corpus()`
        self.sentences = []
        self.sentence_word_lengths = []
        self.sentences_et = []

        # will be created by `normalize_et`
        self.sentences_et_original = []

        self.normalizer = StandardScaler() if normalize else None

        print('\n===== Initializing', self.name, '=====')
        # for the case of GECO (and possibly the other data sets later on),
        # load_corpus() will return None when it finds an `.npy` file
        # which contains previously-extracted corpus data. This saves time!
        feature_values = self.load_corpus()
        if feature_values:
            if self.normalizer:
                feature_values = np.array(feature_values)
                self.print_stats(feature_values)

                # default value for nFixations should be 0
                _nans = np.argwhere(np.isnan(feature_values[0]))
                feature_values[0][_nans] = 0

                # default value for other features should be nan
                for features in feature_values[1:]:
                    _zeros = np.where(features == 0)[0]
                    features[_zeros] = np.nan

                self.normalizer.fit(feature_values.T)
                self.normalize_et()
                print_normalizer_stats(self.name, self.normalizer)

            self._save_to_file()

    def clean_str(self, string):
        """
        Note that this will always be used PER-WORD, not PER-SENTENCE.)
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

    def __len__(self):
        return len(self.sentences)

    def _save_to_file(self):  # currently used by GECO
        pass


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
            sent_words = [self.clean_str(w) for w in sentence['words']]
            self.sentences.append(sent_words)
            self.sentence_word_lengths.append([len(w) for w in sent_words])
            features = np.array([sentence[f] for f in self.et_features])

            features[0] = np.nan_to_num(features[0])
            for i in range(5):
                _feature_values[i].extend(features.reshape(5, -1)[i])

            # warning: this nanmean will produce RuntimeWarnings
            # because some words are not fixated on at all ( i think )
            features = np.nanmean(features.T, axis=1)
            self.sentences_et.append(features)

        return _feature_values


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
        self.num_participants = 14
        self.directory = '../data/GECO Corpus/{}.xlsx'
        self.pre_extracted_dir = '../data/GECO Corpus/pre-extracted{}.npy'.format(
            '-normalized' if normalize else '')

        self.et_features = {
            'nFixations': 'WORD_FIXATION_COUNT',
            'FFD': 'WORD_FIRST_FIXATION_DURATION',
            'TRT': 'WORD_TOTAL_READING_TIME',
            'GD': 'WORD_GAZE_DURATION',
            'GPT': 'WORD_GO_PAST_TIME'
        }

        super(GECO, self).__init__(normalize)
        import pdb; pdb.set_trace()

    def load_corpus(self):
        def _get_word_features(word_id):
            """
            Helper function so that we can extract the ET features per word
            using a list comprehension
            """
            values = geco_df[geco_df['WORD_ID'] == word_id
                             ][self.et_features.values()].values
            if np.any(values):
                return values
            else:
                # happens when a word is NEVER fixated on
                values = np.zeros(
                    (self.num_participants, len(self.et_features)))  # (14, 5)
                values.fill(np.NaN)
                return values

        # To save time, try to load GECO's variables from previous run
        try:
            (self.sentences, self.sentences_et, self.sentence_word_lengths,
                self.sentences_et_original, self.normalizer) = np.load(
                self.pre_extracted_dir, allow_pickle=True)
            print('GECO is loaded from file.\n')
            return None
        except FileNotFoundError:
            print('Pre-extracted GECO is not found in', self.pre_extracted_dir,
                  'Extracting it now...')

        # just for stats...
        _unique_tokens = set([])

        # store all the values in here for use in normalization
        _feature_values = [[], [], [], [], []]

        # load data
        material_df = pd.read_excel(self.directory.format('EnglishMaterial'))
        geco_df = pd.read_excel(self.directory.format('MonolingualReadingData'))

        # do this per sentence!
        sentence_ids = material_df['SENTENCE_ID'].unique().astype('str')
        print('Found', len(sentence_ids), 'unique sentence IDs.')

        for sent_id in sentence_ids:
            if sent_id == 'nan':
                continue

            sent_info = material_df[material_df['SENTENCE_ID'] == sent_id]
            sent_words = sent_info['WORD'].values.astype('str')
            _unique_tokens = _unique_tokens.union(set(sent_words))

            self.sentences.append([self.clean_str(w)
                                   for w in sent_words])
            self.sentence_word_lengths.append(sent_info['WORD_LENGTH'].values)

            # extract and clean eye-tracking data
            features = np.array([
                _get_word_features(word_id)
                for word_id in sent_info['WORD_ID'].unique()
            ])
            features[features == '.'] = np.NaN
            features = features.astype(float)

            for i in range(5):
                _feature_values[i].extend(features[:, :, i].flatten())

            self.sentences_et.append(np.nanmean(features, axis=1))

        print('Found', len(_unique_tokens), 'unique words.')
        return _feature_values

    def _save_to_file(self):
        np.save(self.pre_extracted_dir,
                (self.sentences, self.sentences_et, self.sentence_word_lengths,
                 self.sentences_et_original, self.normalizer),
                allow_pickle=True)
        print('GECO extracted sentences and sentences_et saved to:',
              self.pre_extracted_dir,
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
                self.sentence_word_lengths.append([len(w) for w in sentence_words])
                # extraction level 3: by word
                sentence_et = []

                for clean_word in sentence_words:
                    ___df = __df[__df['Word_Cleaned'] == clean_word]
                    features = ___df[self.et_features.values()].values

                    # PROVO gives 0 values to TRT/DWELL TIME,
                    # but we want those to be NaN.
                    # WILL DO THIS IN Corpus.normalize() instead!
                    # _nonzeros = np.where(features[:, 2] == 0)[0]
                    # features[:, 2][_nonzeros] = np.nan

                    for i in range(5):
                        _feature_values[i].extend(features.T[i])

                    # i think this next line triggers the
                    # TO-DO: add code that doesn't do this if the values
                    # are none. To save on RuntimeWarning outputs...?
                    features = np.nanmean(features, axis=0)
                    sentence_et.append(features)
                self.sentences_et.append(np.array(sentence_et))

        return _feature_values


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
            sent_words = [self.clean_str(w)
                          for w in _df['word'].values[:num_words]]
            self.sentences.append(sent_words)
            self.sentence_word_lengths.append([len(w) for w in sent_words])

            sentence_et = []
            for wp in word_pos:
                features = _build_features(wp)
                sentence_et.append(features)
                for i in range(5):
                    _feature_values[i].extend(features[:, i])
            sentence_et = np.array(sentence_et)
            self.sentences_et.append(np.nanmean(sentence_et, axis=1))

        return _feature_values


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
