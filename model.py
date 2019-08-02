
import torch
from flair.embeddings import TokenEmbeddings
from allennlp.modules.elmo import Elmo

from settings import *

torch.manual_seed(111)
ELMO_OPTIONS_URL = "https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
ELMO_WEIGHT_URL = "https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"


class EyeTrackingPredictor(torch.nn.Module):
    def __init__(self, initial_word_embedding=None, out_features=5,
                 use_word_length=False, use_elmo=False,
                 lstm_hidden_units=LSTM_HIDDEN_UNITS,
                 _prediction_inverse_transformer=None):
        super(EyeTrackingPredictor, self).__init__()
        ### FOR TESTING ###
        self._prediction_inverse_transformer = _prediction_inverse_transformer

        self.out_features = out_features

        # Variable Housekeeping first...
        lstm_layers = 2
        if use_elmo:
            word_embed_dim = 1024
            """ FOR WHEN USING PRE-TRAINED EMBEDDINGS
            self.lstm_hidden_units = 256
            # this is a method that just returns the input...
            self.word_embedding = self._pass
            """
            self.lstm_hidden_units = 256
            self.word_embedding = Elmo(ELMO_OPTIONS_URL, ELMO_WEIGHT_URL,
                                       1, dropout=0.5, requires_grad=True)
        else:
            word_embed_dim = WORD_EMBED_DIM
            self.lstm_hidden_units = lstm_hidden_units
            self.word_embedding = torch.nn.Embedding.from_pretrained(
                initial_word_embedding, freeze=False)

        print('\nInitialize ET Predictor: lstm_hidden_units =',
              self.lstm_hidden_units, 'lstm_layers =', lstm_layers)

        if use_word_length is not False:
                word_embed_dim += 1

        # Define Network
        self.lstm = torch.nn.LSTM(
            input_size=word_embed_dim, hidden_size=self.lstm_hidden_units,
            num_layers=lstm_layers, dropout=DROPOUT_PROB, batch_first=True,
            bidirectional=True)
        self.dropout = torch.nn.Dropout(p=DROPOUT_PROB)
        self.out = torch.nn.Linear(
            in_features=self.lstm_hidden_units * 2,
            out_features=self.out_features)

        # more housekeeping
        self.use_cuda = torch.cuda.is_available()
        if self.use_cuda:
            self.cuda()

    def forward(self, x, word_lengths=None):
        if self.use_cuda:
            x = x.cuda()
            if word_lengths is not None:
                word_lengths = word_lengths.cuda()

        word_embeddings = self.word_embedding(x)

        # if word_lengths is not None:
        #     word_embeddings = torch.cat(
        #         (word_embeddings, word_lengths.reshape(_batch_size, -1, 1)),
        #         dim=2)

        # Actual forward pass:
        lstm_out, (h_n, c_n) = self.lstm(word_embeddings)
        lstm_out = self.dropout(lstm_out)
        et_pred = self.out(lstm_out.reshape(-1, self.lstm_hidden_units * 2))
        # et_pred = self._out(et_pred)
        return et_pred.reshape(x.shape[0], -1, self.out_features)

    def sentences_to_et(self, indexed_sentences, max_seq_len):
        """for downstream use, should be called when training NLP tasks"""
        pad_start_indices = []
        padded_indices = []

        # pad
        for sent in indexed_sentences:
            missing_dims = max_seq_len - len(sent)
            pad_start_indices.append(len(sent))
            padded_indices.append(
                torch.nn.functional.pad(torch.Tensor(sent),
                                        (0, missing_dims)))

        predictions = self.forward(torch.stack(padded_indices).long()).cpu().detach()
        # if the model was trained using MinMaxed data (by CorpusAggregator),
        # bring the predictions back to StandardScale
        # this seems to make predictions better because it makes the signals
        # bigger (from 2 decimals to 1 decimal or ~3)
        if self._prediction_inverse_transformer:
            predictions = torch.Tensor([self._prediction_inverse_transformer.inverse_transform(
                pred) for pred in predictions])

        # revert paddings to 0
        for pred, pad_start in zip(predictions, pad_start_indices):
            pred[pad_start:] = 0

        return predictions

    def _pass(self, x):
        # hacky way to support ELMo embeddings without changing a lot of code.
        return x


class NLPTaskClassifier(torch.nn.Module):
    def __init__(self, initial_word_embedding, lstm_units,
                 max_seq_len, num_classes, use_gaze):
        super(NLPTaskClassifier, self).__init__()

        # Housekeeping
        self.use_gaze = use_gaze
        if self.use_gaze:
            et_feature_dim = len(ET_FEATURES)
        else:
            et_feature_dim = 0
        self.max_sentence_length = max_seq_len

        # Network
        self.word_embedding = torch.nn.Embedding.from_pretrained(
            initial_word_embedding, freeze=False)
        self.lstm = torch.nn.LSTM(input_size=WORD_EMBED_DIM + et_feature_dim,
                                  hidden_size=lstm_units, num_layers=1,
                                  batch_first=True, bidirectional=True)
        self.dropout = torch.nn.Dropout(p=DROPOUT_PROB)
        self.att_l1 = torch.nn.Linear(in_features=lstm_units * 2,
                                      out_features=self.max_sentence_length)
        self.att_l2 = torch.nn.Linear(in_features=self.max_sentence_length,
                                      out_features=2)
        self.hidden_l1 = torch.nn.Linear(in_features=lstm_units * 4,
                                         out_features=50)
        self.out = torch.nn.Linear(in_features=50,
                                   out_features=num_classes)

        self.use_cuda = torch.cuda.is_available()
        if self.use_cuda:
            self.cuda()

    def forward(self, indices, et_features):
        if self.use_cuda:
            indices = indices.cuda()
            if et_features is not None:
                et_features = et_features.cuda()

        batch_size = indices.shape[0]
        word_embeddings = self.word_embedding(indices)

        if self.use_gaze:
            x = torch.cat((word_embeddings, et_features), dim=2)
        else:
            x = word_embeddings

        # Actual forward pass:
        lstm_out, (h_n, c_n) = self.lstm(x)
        lstm_out = self.dropout(lstm_out)

        att_l1_out = self.att_l1(lstm_out)
        att_l1_out_ = torch.tanh(att_l1_out)
        att_l2_out = self.att_l2(att_l1_out_).transpose(1, 2)
        att_weights = torch.nn.functional.softmax(att_l2_out, dim=2)

        lstm_embedding = torch.matmul(att_weights, lstm_out)
        lstm_embedding = lstm_embedding.reshape(batch_size, -1)
        hidden_l1_out = self.hidden_l1(lstm_embedding)
        logits = self.out(hidden_l1_out)
        return logits


class EyeTrackingFeatureEmbedding(TokenEmbeddings):
    """
    A class adhering to Flair Embeddings.

    Used for training tasks already available in Flair, to add
    eye-tracking features to the tokens. See `train_flair.py`
    """
    def __init__(self, model_path):
        super().__init__()
        self.name = 'et_features'
        self.et_predictor, self.vocabulary, agg = load_pretrained_et_predictor(
            model_path)
        self.et_predictor.cuda()
        # - Instantiate EyeTrackingPredictor with weights
        # - Should also store dictionary! ^

    @property
    def embedding_length(self):
        return 5

    def _add_embeddings_internal(self, flair_sentences):
        """
        Flow:
        - Tokenize sentences into its actual words
        - Use predictor outputs to do token.set_embedding
        - Pass the sentence through predictor
        """
        _sentences = [[token.text for token in sentence.tokens]
                      for sentence in flair_sentences]
        indexed_sentences = self.vocabulary.index_sentences(_sentences)

        for sent, flair_sent in zip(indexed_sentences, flair_sentences):
            et_features = self.et_predictor(
                torch.Tensor([sent]).long().cuda()).detach().cpu()[0]

            if self.et_predictor._prediction_inverse_transformer:
                et_features = torch.Tensor([
                    self.et_predictor._prediction_inverse_transformer.inverse_transform(
                        et_features)])

            for et_feat, token in zip(et_features[0], flair_sent.tokens):
                token.set_embedding('ET_feature', et_feat)

        return flair_sentences


def load_pretrained_et_predictor(weights_path):
    data = torch.load(weights_path)
    if 'corpus_aggregator' in data:
        aggregator = data['corpus_aggregator']
        vocab = aggregator.vocabulary

    else:
        aggregator = None
        vocab = data['vocabulary']

    model = EyeTrackingPredictor(
        initial_word_embedding=torch.zeros((len(vocab), WORD_EMBED_DIM)),
        lstm_hidden_units=int(
            data['model_state_dict']['lstm.weight_ih_l0'].shape[0] / 4),
        ### FOR TESTING ###
        _prediction_inverse_transformer=aggregator.normalizer
    )

    model.load_state_dict(data['model_state_dict'])
    model.eval()

    return model, vocab, aggregator
