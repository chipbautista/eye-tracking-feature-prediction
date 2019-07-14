
import numpy as np
import torch
from gensim.models import KeyedVectors

from settings import *


class EyeTrackingPredictor(torch.nn.Module):
    def __init__(self, initial_word_embedding, max_seq_len):
        super(EyeTrackingPredictor, self).__init__()

        self.word_embedding = torch.nn.Embedding.from_pretrained(
            initial_word_embedding, freeze=False)
        self.lstm = torch.nn.LSTM(
            input_size=WORD_EMBED_DIM, hidden_size=LSTM_HIDDEN_UNITS,
            num_layers=1, batch_first=True, bidirectional=True
        )
        self.dropout = torch.nn.Dropout(p=DROPOUT_PROB)
        self.out = torch.nn.Linear(
            in_features=LSTM_HIDDEN_UNITS * 2,
            out_features=len(ET_FEATURES)
        )

    def forward(self, x):
        """
        x: (B, max_seq_len)
        word_embeddings: (B, max_seq_len, 300)
        lstm_out: (B, max_seq_len, 2 * 128)
        out: (B, max_seq_len, 5)
        """
        word_embeddings = self.word_embedding(x)
        lstm_out, (h_n, c_n) = self.lstm(word_embeddings)
        lstm_out = self.dropout(lstm_out)
        # add dropout here?
        et_pred = self.out(lstm_out.reshape(-1, LSTM_HIDDEN_UNITS * 2))
        return et_pred.reshape(x.shape[0], -1, len(ET_FEATURES))


def init_word_embedding_from_word2vec(vocabulary):
    print('\nLoading pre-trained word2vec from', WORD_EMBED_MODEL_DIR)
    pretrained_w2v = KeyedVectors.load_word2vec_format(
        WORD_EMBED_MODEL_DIR, binary=True)
    print('Done. Will now extract embeddings for needed words.')

    embeddings = [np.random.uniform(-0.25, 0.25, WORD_EMBED_DIM)]  # index 0 is just padding
    oov_words = []
    for word in vocabulary:
        try:
            embeddings.append(pretrained_w2v[word])
        except KeyError:
            embeddings.append(np.random.uniform(-0.25, 0.25, WORD_EMBED_DIM))
            oov_words.append(word)

    print('>', len(oov_words), 'words were not found in the pre-trained model.\n')
    return torch.Tensor(embeddings)
