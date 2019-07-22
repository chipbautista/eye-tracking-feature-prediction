import time
from argparse import ArgumentParser

import torch
import numpy as np
from sklearn.metrics import (accuracy_score, f1_score,
                             precision_score, recall_score)

from datasets_tasks import ZuCo_Task, IMDb
from model import NLPTaskClassifier, init_word_embedding_from_word2vec
from settings import *


def print_metrics(metrics, split):
    # just a helper function to cleanly print metrics...
    print('\n[{}]: '.format(split), end='')
    for k, v in metrics.items():
        if type(v) == np.float64:
            print(' {}: {:.2f}'.format(k, v), end='')
        else:
            print(' {}: {:.2f} '.format(k, np.mean(v)), end='')


def get_metrics(targets, predictions):
    return {
        'accuracy': accuracy_score(targets, predictions),
        'f1': f1_score(targets, predictions, average='weighted'),
        'precision': precision_score(targets, predictions, average='weighted'),
        'recall': recall_score(targets, predictions, average='weighted')
    }


def iterate(loader, train=True):
    epoch_loss = 0.0
    all_predictions = []
    all_targets = []
    for i, (sentences, targets) in enumerate(loader):
        sentences = sentences.long()
        logits = model(sentences, None)
        loss = XE_loss(logits, targets.long())

        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        epoch_loss += loss.item()
        all_targets.extend(targets)
        all_predictions.extend(torch.softmax(logits, dim=1).argmax(dim=1))

    return epoch_loss / (i + 1), get_metrics(all_targets, all_predictions)


parser = ArgumentParser()
parser.add_argument('--dataset', default=None)
parser.add_argument('--use-gaze', default=True)
args = parser.parse_args()

if 'zuco' in args.dataset:
    dataset = ZuCo_Task(args.dataset.split('-')[-1])
    do_cross_validation = True
elif args.dataset == 'imdb':
    train_loader = torch.utils.data.DataLoader(IMDb('train'))
    test_loader = torch.utils.data.DataLoader(IMDb('test'))
    do_cross_validation = False
else:
    print('please input data set to use.')  # sabog, fix later.

initial_word_embedding = init_word_embedding_from_word2vec(
    dataset.vocabulary.keys())
XE_loss = torch.nn.CrossEntropyLoss()

if do_cross_validation:
    for k, (train_loader, test_loader) in enumerate(
            dataset.split_cross_val()):

        _start_time = time.time()
        train_losses = []
        # train_metrics = []
        test_losses = []
        test_metrics = []

        model = NLPTaskClassifier(initial_word_embedding.clone(),
                                  dataset.max_seq_len, dataset.num_classes,
                                  use_gaze=False)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
        for e in range(NUM_EPOCHS):
            train_loss, train_metrics_ = iterate(train_loader)
            train_losses.append(train_loss)
            test_loss, test_metrics_ = iterate(test_loader, train=False)
            test_losses.append(test_loss)
            test_metrics.append(test_metrics_)

            # print(k, e, train_loss, test_loss)

        best_epoch = np.argmin(test_losses)
        print(k, '[e={}] '.format(best_epoch),
              '- Train XE: {:.2f}'.format(train_losses[best_epoch]),
              'Test XE: {:.2f} '.format(test_losses[best_epoch]),
              '({:.2f}s)'.format(time.time() - _start_time))
        print_metrics(test_metrics[best_epoch], 'test')

else:
    train_loss = iterate(train_loader)
    test_loss = iterate(test_loader)
