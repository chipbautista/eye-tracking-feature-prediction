import time
from argparse import ArgumentParser

import torch
import numpy as np
from sklearn.metrics import (accuracy_score, f1_score,
                             precision_score, recall_score)

from datasets_tasks import ZuCo_Task, IMDb
from model import NLPTaskClassifier, load_pretrained_et_predictor
from settings import *


def _print_metrics(metrics, split):
    # just a helper function to cleanly print metrics...
    text = '[{}]: '.format(split)
    for k, v in metrics.items():
        if type(v) == np.float64:
            text += ' {}: {:.4f}'.format(k, v)
        else:
            text += ' {}: {:.4f} '.format(k, np.mean(v))
    print(text, '\n')


def _get_metrics(targets, predictions):
    ave_method = 'macro'
    return {
        'accuracy': accuracy_score(targets, predictions),
        'f1': f1_score(targets, predictions, average=ave_method),
        'precision': precision_score(targets, predictions, average=ave_method),
        'recall': recall_score(targets, predictions, average=ave_method)
    }


def iterate(loader):
    epoch_loss = 0.0
    all_predictions = []
    all_targets = []
    for i, data in enumerate(loader):
        if len(data) > 2:
            sentences, et_features, targets = data
        else:
            sentences, targets = data
            et_features = None

        sentences = sentences.long()

        if USE_CUDA:  # clean this later
            targets = targets.cuda()

        logits = model(sentences, et_features)
        loss = XE_loss(logits, targets.long())

        if model.training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        epoch_loss += loss.item()
        all_targets.extend(targets.cpu())
        all_predictions.extend(torch.softmax(logits, dim=1).argmax(dim=1).cpu())

    return epoch_loss / (i + 1), _get_metrics(all_targets, all_predictions)


parser = ArgumentParser()
parser.add_argument('--dataset', default=None)
parser.add_argument('--gaze-data', default=False,
                    help='OWN=data set\'s own gaze data. [PATH_TO_SAVED_WEIGHTS]=use model')
parser.add_argument('--lr', default=0.01)
parser.add_argument('--num-epochs', default=85)
parser.add_argument('--batch-size', default=32)
parser.add_argument('--use-predictor-vocab', default='False')
args = parser.parse_args()


if str(args.gaze_data).lower() not in ['own', 'false']:
    et_predictor, vocab, aggregator = load_pretrained_et_predictor(
        str(args.gaze_data))
else:
    et_predictor, vocab = None, None

if 'zuco' in args.dataset:
    dataset = ZuCo_Task(args.dataset.split('-')[-1], int(args.batch_size),
                        args.gaze_data, et_predictor, vocab,
                        use_predictor_vocab=args.use_predictor_vocab != 'False')

    lstm_units = 150
    do_cross_validation = True
elif args.dataset == 'imdb':
    train_loader = torch.utils.data.DataLoader(IMDb('train'))
    test_loader = torch.utils.data.DataLoader(IMDb('test'))
    do_cross_validation = False
else:
    print('please input data set to use.')  # sabog, fix later.


XE_loss = torch.nn.CrossEntropyLoss()

print('--- PARAMETERS ---')
print('Data Set:', args.dataset)
print('Gaze Data:', args.gaze_data)
print('Learning Rate:', args.lr)
print('Batch Size:', args.batch_size)
print('# Epochs:', args.num_epochs)
print('LSTM Hidden Units:', lstm_units)
print('Number of samples:', len(dataset))

if do_cross_validation:
    print('\n--- STARTING CROSS-VALIDATION ---')

    best_test_xe = []
    best_test_metrics = []

    for k, (train_loader, test_loader) in enumerate(
            dataset.split_cross_val()):

        _start_time = time.time()
        train_losses = []
        test_losses = []
        test_metrics = []

        model = NLPTaskClassifier(dataset.vocabulary.word_embeddings.clone(),
                                  lstm_units, dataset.max_seq_len,
                                  dataset.num_classes,
                                  use_gaze=args.gaze_data is not False)

        optimizer = torch.optim.SGD(model.parameters(), lr=float(args.lr),
                                    momentum=0.95, nesterov=True)

        for e in range(int(args.num_epochs)):
            model.train()
            train_loss, train_metrics_ = iterate(train_loader)
            train_losses.append(train_loss)

            model.eval()
            test_loss, test_metrics_ = iterate(test_loader)
            test_losses.append(test_loss)
            test_metrics.append(test_metrics_)

            # print(k, e, train_loss, test_loss)

        best_epoch = np.argmin(test_losses)
        print('Fold', k, '[e={}] '.format(best_epoch),
              '- Train XE: {:.2f}'.format(train_losses[best_epoch]),
              'Test XE: {:.2f} '.format(test_losses[best_epoch]),
              '({:.2f}s)'.format(time.time() - _start_time))
        _print_metrics(test_metrics[best_epoch], 'test')

        best_test_xe.append(test_losses[best_epoch])
        best_test_metrics.append(test_metrics[best_epoch])

    print('[Mean Test XE]', np.mean(best_test_xe))
    _print_metrics(
        dict(zip(list(test_metrics[0].keys()),
                 np.array([list(metric.values())
                           for metric in best_test_metrics]).mean(0))),
        'Mean test metrics')

else:
    train_loss = iterate(train_loader)
    test_loss = iterate(test_loader)
