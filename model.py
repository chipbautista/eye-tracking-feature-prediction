
import numpy as np
import torch
from gensim.models import KeyedVectors

from settings import *

torch.manual_seed(123)
np.random.seed(123)


class EyeTrackingPredictor(torch.nn.Module):
    def __init__(self, initial_word_embedding, max_seq_len, out_features):
        super(EyeTrackingPredictor, self).__init__()

        self.out_features = out_features

        self.word_embedding = torch.nn.Embedding.from_pretrained(
            initial_word_embedding, freeze=False)
        self.lstm = torch.nn.LSTM(
            input_size=WORD_EMBED_DIM, hidden_size=LSTM_HIDDEN_UNITS,
            num_layers=1, batch_first=True, bidirectional=True
        )
        self.dropout = torch.nn.Dropout(p=DROPOUT_PROB)
        self.out = torch.nn.Linear(
            in_features=LSTM_HIDDEN_UNITS * 2,
            out_features=self.out_features
        )

    def forward(self, x):
        word_embeddings = self.word_embedding(x)
        lstm_out, (h_n, c_n) = self.lstm(word_embeddings)
        lstm_out = self.dropout(lstm_out)
        et_pred = self.out(lstm_out.reshape(-1, LSTM_HIDDEN_UNITS * 2))
        return et_pred.reshape(x.shape[0], -1, self.out_features)


class NLPTaskClassifier(torch.nn.Module):
    def __init__(self, initial_word_embedding,
                 max_seq_len, num_classes, use_gaze):
        super(NLPTaskClassifier, self).__init__()
        LSTM_HIDDEN_UNITS = 150

        self.use_gaze = use_gaze
        if self.use_gaze:
            et_feature_dim = len(ET_FEATURES)
        else:
            et_feature_dim = 0
        self.max_sentence_length = max_seq_len

        # self.out_features = num_classes
        self.word_embedding = torch.nn.Embedding.from_pretrained(
            initial_word_embedding, freeze=False)
        self.lstm = torch.nn.LSTM(input_size=WORD_EMBED_DIM + et_feature_dim,
                                  hidden_size=LSTM_HIDDEN_UNITS, num_layers=1,
                                  batch_first=True, bidirectional=True)
        self.dropout = torch.nn.Dropout(p=DROPOUT_PROB)
        self.att_l1 = torch.nn.Linear(in_features=LSTM_HIDDEN_UNITS * 2,
                                      out_features=self.max_sentence_length)
        self.att_l2 = torch.nn.Linear(in_features=self.max_sentence_length,
                                      out_features=2)
        self.hidden_l1 = torch.nn.Linear(in_features=LSTM_HIDDEN_UNITS * 4,
                                         out_features=50)
        self.out = torch.nn.Linear(in_features=50,
                                   out_features=num_classes)

    def forward(self, indices, et_features):
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


def init_word_embedding_from_word2vec(vocabulary):
    print('\nLoading pre-trained word2vec from', WORD_EMBED_MODEL_DIR)
    pretrained_w2v = KeyedVectors.load_word2vec_format(
        WORD_EMBED_MODEL_DIR, binary=True)
    print('Done. Will now extract embeddings for needed words.')

    embeddings = [np.random.uniform(-1.0, 1.0, WORD_EMBED_DIM)]  # index 0 is just padding
    oov_words = []
    for word in vocabulary:
        try:
            embeddings.append(pretrained_w2v[word])
        except KeyError:
            embeddings.append(np.random.uniform(-1.0, 1.0, WORD_EMBED_DIM))
            oov_words.append(word)

    print(len(oov_words), 'words were not found in the pre-trained model.\n')
    return torch.Tensor(embeddings)
