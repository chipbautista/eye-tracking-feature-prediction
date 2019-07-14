import time
from argparse import ArgumentParser

import torch
import numpy as np

from datasets import CorpusAggregator
from model import EyeTrackingPredictor, init_word_embedding_from_word2vec
from settings import *


def iterate(dataloader, train=True):
    epoch_loss = 0.0
    for i, (sentences, et_targets) in enumerate(dataloader):
        et_preds = model(sentences.type(torch.LongTensor))

        # starting from the padding index, make the prediction values 0
        for sent, et_pred in zip(sentences, et_preds):
            try:
                pad_start_idx = np.where(sent.numpy() == 0)[0][0]
            except IndexError:
                pad_start_idx = None
            et_pred[pad_start_idx:] = 0
        loss = mse_loss(et_preds, et_targets)

        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        epoch_loss += loss.item()
    return epoch_loss / (i + 1)


parser = ArgumentParser()
parser.add_argument('--zuco', default=False)
parser.add_argument('--provo', default=False)
parser.add_argument('--geco', default=False)
args = parser.parse_args()

if args.zuco is False and args.provo is False and args.geco is False:
    corpus_list = ['ZuCo', 'PROVO', 'GECO']  # add GECO later
else:
    corpus_list = []
    if args.zuco is not False:
        corpus_list.append('ZuCo')
    if args.provo is not False:
        corpus_list.append('PROVO')
    if args.geco is not False:
        corpus_list.append('GECO')

dataset = CorpusAggregator(corpus_list)
initial_word_embedding = init_word_embedding_from_word2vec(
    dataset.vocabulary.keys())
mse_loss = torch.nn.MSELoss()

print('--- PARAMETERS ---')
print('Learning Rate:', INITIAL_LR)
print('# Epochs:', NUM_EPOCHS)
print('LSTM Hidden Units:', LSTM_HIDDEN_UNITS)
print('Number of sentences:', len(dataset))
print('\n--- Starting training (10-CV) ---')

te_losses = []
for k, (train_loader, test_loader) in enumerate(dataset.split_cross_val()):
    _start_time = time.time()
    model = EyeTrackingPredictor(initial_word_embedding.clone(),
                                 dataset.max_seq_len)
    optimizer = torch.optim.Adam(model.parameters(), lr=INITIAL_LR)
    if USE_CUDA:
        model = model.cuda()

    e_tr_losses = []
    e_te_losses = []
    for e in range(NUM_EPOCHS):
        train_loss = iterate(train_loader)
        test_loss = iterate(test_loader, train=False)
        e_tr_losses.append(train_loss)
        e_te_losses.append(test_loss)
        print(k, e, train_loss, test_loss)

    best_epoch = np.argmin(e_te_losses)
    te_losses.append(e_te_losses[best_epoch])
    print(k, e,'- Train MSE: {:.2f}'.format(e_tr_losses[best_epoch]),
          'Test MSE: {:.2f} '.format(e_te_losses[best_epoch]),
          '({:.2f}s)'.format(time.time() - _start_time))

print('\nCV Mean Test Loss:', np.mean(te_losses))
