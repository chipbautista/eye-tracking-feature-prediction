
import torch
from flair.embeddings import TokenEmbeddings

from settings import *

torch.manual_seed(111)


class EyeTrackingPredictor(torch.nn.Module):
    def __init__(self, initial_word_embedding, out_features=5,
                 use_word_length=False):
        super(EyeTrackingPredictor, self).__init__()

        word_embed_dim = WORD_EMBED_DIM
        if use_word_length is not False:
            word_embed_dim += 1

        self.out_features = out_features
        self.word_embedding = torch.nn.Embedding.from_pretrained(
            initial_word_embedding, freeze=False)
        self.lstm = torch.nn.LSTM(
            input_size=word_embed_dim, hidden_size=LSTM_HIDDEN_UNITS,
            num_layers=2, dropout=DROPOUT_PROB, batch_first=True,
            bidirectional=True)
        self.dropout = torch.nn.Dropout(p=DROPOUT_PROB)
        self.out = torch.nn.Linear(
            in_features=LSTM_HIDDEN_UNITS * 2,
            out_features=self.out_features)

        self.use_cuda = torch.cuda.is_available()
        if self.use_cuda:
            self.cuda()

    def forward(self, x, word_lengths=None):
        if self.use_cuda:
            x = x.cuda()
            if word_lengths is not None:
                word_lengths = word_lengths.cuda()

        _batch_size = x.shape[0]
        word_embeddings = self.word_embedding(x)

        if word_lengths is not None:
            word_embeddings = torch.cat(
                (word_embeddings, word_lengths.reshape(_batch_size, -1, 1)),
                dim=2)

        lstm_out, (h_n, c_n) = self.lstm(word_embeddings)
        lstm_out = self.dropout(lstm_out)
        et_pred = self.out(lstm_out.reshape(-1, LSTM_HIDDEN_UNITS * 2))
        return et_pred.reshape(x.shape[0], -1, self.out_features)

    def sentences_to_et(self, indexed_sentences, max_seq_len):
        """for downstream use"""
        pad_start_indices = []
        padded_indices = []

        # pad
        for sent in indexed_sentences:
            missing_dims = max_seq_len - len(sent)
            pad_start_indices.append(len(sent))
            padded_indices.append(
                torch.nn.functional.pad(torch.Tensor(sent),
                                        (0, missing_dims)))

        predictions = self.forward(torch.stack(padded_indices).long())

        # revert paddings to 0
        for pred, pad_start in zip(predictions, pad_start_indices):
            pred[pad_start:] = 0

        return predictions.cpu().detach()


class NLPTaskClassifier(torch.nn.Module):
    def __init__(self, initial_word_embedding, lstm_units,
                 max_seq_len, num_classes, use_gaze):
        super(NLPTaskClassifier, self).__init__()

        self.use_gaze = use_gaze
        if self.use_gaze:
            et_feature_dim = len(ET_FEATURES)
        else:
            et_feature_dim = 0
        self.max_sentence_length = max_seq_len

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
    def __init__(self, model_path):
        super().__init__()
        self.name = 'et_features'
        self.et_predictor, self.vocabulary = load_pretrained_et_predictor(
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
        # print(flair_sentences)
        _sentences = [[token.text for token in sentence.tokens]
                      for sentence in flair_sentences]
        import pdb; pdb.set_trace()
        indexed_sentences = self.vocabulary.index_sentences(_sentences)

        for sent, flair_sent in zip(indexed_sentences, flair_sentences):
            et_features = self.et_predictor(
                torch.Tensor([sent]).long().cuda()).detach().cpu()
            for et_feat, token in zip(et_features[0], flair_sent.tokens):
                token.set_embedding('ET_feature', et_feat)

        return flair_sentences

#     # from WordEmbeddings:
#     def _add_embeddings_internal(
#             self, sentences: List[Sentence]) -> List[Sentence]:

#         for i, sentence in enumerate(sentences):

#             for token, token_idx in zip(sentence.tokens, range(len(sentence.tokens))):

#                 if "field" not in self.__dict__ or self.field is None:
#                     word = token.text
#                 else:
#                     word = token.get_tag(self.field).value

#                 if word in self.precomputed_word_embeddings:
#                     word_embedding = self.precomputed_word_embeddings[word]
#                 elif word.lower() in self.precomputed_word_embeddings:
#                     word_embedding = self.precomputed_word_embeddings[word.lower()]
#                 elif (
#                     re.sub(r"\d", "#", word.lower()) in self.precomputed_word_embeddings
#                 ):
#                     word_embedding = self.precomputed_word_embeddings[
#                         re.sub(r"\d", "#", word.lower())
#                     ]
#                 elif (
#                     re.sub(r"\d", "0", word.lower()) in self.precomputed_word_embeddings
#                 ):
#                     word_embedding = self.precomputed_word_embeddings[
#                         re.sub(r"\d", "0", word.lower())
#                     ]
#                 else:
#                     word_embedding = np.zeros(self.embedding_length, dtype="float")

#                 word_embedding = torch.FloatTensor(word_embedding)

#                 token.set_embedding(self.name, word_embedding)

#         return sentences


def load_pretrained_et_predictor(weights_path):
    data = torch.load(weights_path)
    model = EyeTrackingPredictor(
        torch.zeros((len(data['vocabulary']), WORD_EMBED_DIM)))
    model.load_state_dict(data['model_state_dict'])
    model.eval()
    return model, data['vocabulary']
