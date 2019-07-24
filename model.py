
import torch
# from flair import TokenEmbeddings

from datasets import index_sentences
from settings import *

torch.manual_seed(111)


class EyeTrackingPredictor(torch.nn.Module):
    def __init__(self, initial_word_embedding, out_features=5):
        super(EyeTrackingPredictor, self).__init__()

        self.out_features = out_features

        self.word_embedding = torch.nn.Embedding.from_pretrained(
            initial_word_embedding, freeze=False)
        self.lstm = torch.nn.LSTM(
            input_size=WORD_EMBED_DIM, hidden_size=LSTM_HIDDEN_UNITS,
            num_layers=2, dropout=DROPOUT_PROB, batch_first=True,
            bidirectional=True)
        self.dropout = torch.nn.Dropout(p=DROPOUT_PROB)
        self.out = torch.nn.Linear(
            in_features=LSTM_HIDDEN_UNITS * 2,
            out_features=self.out_features)

        self.use_cuda = torch.cuda.is_available()
        if self.use_cuda:
            self.cuda()

    def forward(self, x):
        if self.use_cuda:
            x = x.cuda()

        word_embeddings = self.word_embedding(x)
        lstm_out, (h_n, c_n) = self.lstm(word_embeddings)
        lstm_out = self.dropout(lstm_out)
        et_pred = self.out(lstm_out.reshape(-1, LSTM_HIDDEN_UNITS * 2))
        return et_pred.reshape(x.shape[0], -1, self.out_features)

    def sentences_to_et(self, indexed_sentences, max_seq_len):
        # for downstream use!
        padded_indices = []
        for sent in indexed_sentences:
            missing_dims = max_seq_len - len(sent)
            padded_indices.append(
                torch.nn.functional.pad(torch.Tensor(sent),
                                        (0, missing_dims)))
        return self.forward(torch.stack(padded_indices).long()).cpu().detach()


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


# class FlairEyeTrackingEmbedding(TokenEmbeddings):
#     def __init__(self, et_predictor, vocabulary):
#         self.name = 'et_features'
#         self.et_predictor = et_predictor
#         self.vocabulary = vocabulary
#         # - Instantiate EyeTrackingPredictor with weights
#         # - Should also store dictionary! ^
#         super().__init__()

#     @property
#     def embedding_length(self) -> int:
#         return self.__embedding_length

#     def _add_embeddings_internal(self, sentences):
#         """
#         Flow:
#         - Tokenize sentences into its actual words
#         - Pass the sentence through predictor
#         - Use predictor outputs to do token.set_embedding
#         """
#         # WARNING! this `sentences` object is not a list, but a Flair object
#         indexed_sentences = index_sentences(sentences, self.vocabulary)
#         predicted_ets = self.et_predictor(indexed_sentences)
#         for sentence, predicted_et in zip(sentences, predicted_ets):
#             for token, token_et in zip(sentence.tokens, predicted_et):
#                 token.set_embedding(self.name, token_et)

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
