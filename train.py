import time
from argparse import ArgumentParser

import torch
import numpy as np

from datasets import CorpusAggregator
from model import EyeTrackingPredictor, init_word_embedding_from_word2vec
from settings import *

# TO-DO: Convert words that occur <5 times to <UNK>!
# Can use collections.Counter


def iterate(dataloader):
    epoch_loss = 0.0
    # loss calculated on the real/original values (not scaled)
    epoch_loss_ = torch.Tensor([0, 0, 0, 0, 0])
    for i, (sentences, et_targets,
            et_targets_orig, indices) in enumerate(dataloader):
        sentences = sentences.type(torch.LongTensor)
        if USE_CUDA:
            sentences = sentences.cuda()
            et_targets = et_targets.cuda()

        et_preds = model(sentences)

        et_preds_inverse = torch.Tensor([
            dataset.inverse_transform(idx, value)
            for (idx, value) in zip(indices, et_preds.detach().cpu())])

        # starting from the padding index, make the prediction values 0
        for sent, et_pred, et_pred_inverse in zip(
                sentences, et_preds, et_preds_inverse):
            try:
                pad_start_idx = np.where(sent.cpu().numpy() == 0)[0][0]
            except IndexError:
                pad_start_idx = None
            et_pred[pad_start_idx:] = 0
            et_pred_inverse[pad_start_idx:] = 0

        num_data_points = et_targets_orig[et_targets_orig > 0].shape[0]
        # mse loss divided by the actual number of data points
        # (have to disregard the padding!)
        loss = torch.sqrt(mse_loss(et_preds, et_targets) / num_data_points)

        # calculate the loss PER FEATURE
        loss_ = torch.Tensor([mae_loss(et_preds_inverse[:, :, i],
                                       et_targets_orig[:, :, i]).item()
                              for i in range(5)])
        loss_ /= (num_data_points / 5)

        if model.training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        epoch_loss += loss.item()
        epoch_loss_ += loss_

    return epoch_loss / (i + 1), epoch_loss_ / (i + 1)


parser = ArgumentParser()
parser.add_argument('--zuco-1', default=False)
parser.add_argument('--zuco-2', default=False)
parser.add_argument('--zuco-3', default=False)
parser.add_argument('--provo', default=False)
parser.add_argument('--geco', default=False)
parser.add_argument('--ucl', default=False)
parser.add_argument('--normalize-aggregate', default='False')
parser.add_argument('--save-model', default=False)
args = parser.parse_args()

print(args)
if (args.zuco_1 is False and args.zuco_2 is False and args.zuco_3 is False and
        args.provo is False and args.geco is False and args.ucl is False):
    # add UCL later?
    corpus_list = ['ZuCo-1', 'ZuCo-2', 'ZuCo-3', 'PROVO', 'GECO', 'UCL']
else:
    corpus_list = []
    if args.zuco_1 is not False:
        corpus_list.append('ZuCo-1')
    if args.zuco_2 is not False:
        corpus_list.append('ZuCo-2')
    if args.zuco_3 is not False:
        corpus_list.append('ZuCo-3')
    if args.provo is not False:
        corpus_list.append('PROVO')
    if args.geco is not False:
        corpus_list.append('GECO')
    if args.ucl is not False:
        corpus_list.append('UCL')

dataset = CorpusAggregator(corpus_list, eval(args.normalize_aggregate))
initial_word_embedding = init_word_embedding_from_word2vec(
    dataset.vocabulary.keys())
mse_loss = torch.nn.MSELoss(reduction='sum')
mae_loss = torch.nn.L1Loss(reduction='sum')

print('--- PARAMETERS ---')
print('Learning Rate:', INITIAL_LR)
print('# Epochs:', NUM_EPOCHS)
print('LSTM Hidden Units:', LSTM_HIDDEN_UNITS)
print('Number of sentences:', len(dataset))
print('\n--- Starting training (10-CV) ---')

te_losses = []
te_losses_ = []
for k, (train_loader, test_loader) in enumerate(
        dataset.split_cross_val(stratified=False)):
    _start_time = time.time()

    if k == 0:
        print('Train #batches:', len(train_loader))
        print('Test #batches:', len(test_loader))

    model = EyeTrackingPredictor(initial_word_embedding.clone(),
                                 dataset.max_seq_len, len(ET_FEATURES))
    if USE_CUDA:
        model = model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=INITIAL_LR)

    best_epochs = []
    e_tr_losses = []
    e_tr_losses_ = []
    e_te_losses = []
    e_te_losses_ = []
    for e in range(NUM_EPOCHS):
        model.train()
        train_loss, train_loss_ = iterate(train_loader)

        model.eval()
        test_loss, test_loss_ = iterate(test_loader)

        e_tr_losses.append(train_loss)
        e_tr_losses_.append(train_loss_)
        e_te_losses.append(test_loss)
        e_te_losses_.append(test_loss_)

        # print('k:', k, 'e:', e,
        #       '{:.5f}'.format(train_loss), '{:.5f}'.format(test_loss))
        # print(train_loss_)
        # print(test_loss_)

    best_epoch = np.argmin(e_te_losses)
    best_epochs.append(best_epoch)
    te_losses.append(e_te_losses[best_epoch])
    te_losses_.append(e_te_losses_[best_epoch])

    print(k, '[e={}] '.format(best_epoch),
          '- Train rMSE: {:.5f}'.format(e_tr_losses[best_epoch]),
          'Test rMSE: {:.5f} '.format(e_te_losses[best_epoch]),
          '({:.2f}s)'.format(time.time() - _start_time))
    print('Train MSE_:', e_tr_losses_[best_epoch])
    print('Test MSE_:', e_te_losses_[best_epoch])

print('\nCV Mean Test Loss:', np.mean(te_losses))
print(torch.stack(te_losses_).mean(0))

if args.save_model is not False:
    mean_epoch = np.mean(best_epochs)
    print('Will save final model. Will now train on all data points.')
    print('Mean number of epochs until overfit:', mean_epoch)

    model = EyeTrackingPredictor(initial_word_embedding.clone(),
                                 dataset.max_seq_len, len(ET_FEATURES))
    optimizer = torch.optim.Adam(model.parameters(), lr=INITIAL_LR)
    # hacky and i love it :(
    for train_loader, test_loader in dataset.split_cross_val(
            num_folds=2, stratified=False):
        for e in range(round(mean_epoch)):
            loss_1, loss_1_ = iterate(train_loader)
            loss_2, loss_2_ = iterate(test_loader)
        break

    import pdb; pdb.set_trace()
    model_datasets = ''.join([corpus[0] for corpus in corpus_list]).upper()
    torch.save(model.state_dict(), TRAINED_ET_MODEL_DIR.format(model_datasets))
