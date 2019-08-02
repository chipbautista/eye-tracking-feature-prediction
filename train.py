import time
from argparse import ArgumentParser

import torch
import numpy as np

from data import CorpusAggregator
from model import EyeTrackingPredictor
from settings import *


def _get_model_and_optim():
    if args.finetune_elmo != 'False' or args.static_embedding:
        model = EyeTrackingPredictor(
            finetune_elmo=eval(args.finetune_elmo),
            static_embedding=args.static_embedding)
    else:
        model = EyeTrackingPredictor(
            dataset.vocabulary.word_embeddings.clone())

    return (
        model,
        torch.optim.Adam(model.parameters(), lr=eval(args.lr))
    )


# should probably move this to a Trainer class...
def iterate(dataloader):
    epoch_loss = 0.0
    # loss calculated on the real/original values (not scaled)
    epoch_loss_ = torch.Tensor([0, 0, 0, 0, 0])
    for i, (sentences, et_targets,
            et_targets_orig, indices) in enumerate(dataloader):

        # if not args.finetune_elmo_embeddings:
        # sentences = sentences.type(torch.LongTensor)
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
        # loss = torch.sqrt(mse_loss(et_preds, et_targets) / num_data_points)
        loss = mae_loss(et_preds, et_targets) / num_data_points

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
# Data Sets
parser.add_argument('--zuco-1', default=False)
parser.add_argument('--zuco-2', default=False)
parser.add_argument('--zuco-3', default=False)
parser.add_argument('--provo', default=False)
parser.add_argument('--geco', default=False)
parser.add_argument('--ucl', default=False)
# Data Set Preparation
parser.add_argument('--minmax-aggregate', default='False')
parser.add_argument('--use-word-length', default='False')
parser.add_argument('--filter-vocab', default='True')
parser.add_argument('--normalize-wrt-mean', default='False')
parser.add_argument('--train-per-sample', default='False')
# Predictor Settings
parser.add_argument('--static-embedding', default='')
parser.add_argument('--finetune-elmo', default='False',
                    help='Finetune pre-trained ELMo instead of word2vec')
# Training Settings
parser.add_argument('--save-model', default=False)
parser.add_argument('--num-epochs', default=str(NUM_EPOCHS))
parser.add_argument('--batch-size', default=str(BATCH_SIZE))
parser.add_argument('--lr', default=str(INITIAL_LR))


args = parser.parse_args()
print(args)

if (args.zuco_1 is False and args.zuco_2 is False and args.zuco_3 is False and
        args.provo is False and args.geco is False and args.ucl is False):
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

dataset = CorpusAggregator(corpus_list,
                           minmax_aggregate=eval(args.minmax_aggregate),
                           normalize_wrt_mean=eval(args.normalize_wrt_mean),
                           filter_vocab=eval(args.filter_vocab),
                           finetune_elmo=eval(args.finetune_elmo),
                           train_per_sample=eval(args.train_per_sample),
                           static_embedding=args.static_embedding)
mse_loss = torch.nn.MSELoss(reduction='sum')
mae_loss = torch.nn.L1Loss(reduction='sum')

print('--- PARAMETERS ---')
print('Learning Rate:', eval(args.lr))
print('# Epochs:', eval(args.num_epochs))
print('Batch Size:', eval(args.batch_size))
print('Number of sentences:', len(dataset))
print('Finetune ELMo embeddings:', args.finetune_elmo)
print('Static embeddings:', args.static_embedding)
print('\n--- Starting training (10-CV) ---')

te_losses = []
te_losses_ = []
for k, (train_loader, test_loader) in enumerate(
        dataset.split_cross_val(stratified=False)):
    _start_time = time.time()

    if k == 0:
        print('Train #batches:', len(train_loader))
        print('Test #batches:', len(test_loader))

    model, optimizer = _get_model_and_optim()
    optim_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.1, patience=2, verbose=False)

    best_epochs = []
    e_tr_losses = []
    e_tr_losses_ = []
    e_te_losses = []
    e_te_losses_ = []
    for e in range(eval(args.num_epochs)):
        model.train()
        train_loss, train_loss_ = iterate(train_loader)

        model.eval()
        test_loss, test_loss_ = iterate(test_loader)
        optim_scheduler.step(test_loss)

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
    print('Train MAE:', e_tr_losses_[best_epoch])
    print('Test MAE:', e_te_losses_[best_epoch])

print('\nCV Mean Test Loss:', np.mean(te_losses))
print(torch.stack(te_losses_).mean(0))

if args.save_model is not False:
    mean_epoch = int(round(np.mean(best_epochs)))
    print('Mean number of epochs until overfit:', mean_epoch)
    print('Will save final model. Will now train on all data points in',
          mean_epoch + 3, 'epochs.')

    train_loader = dataset._get_dataloader(
        indices=np.array(range(len(dataset.sentences))))

    for e in range(mean_epoch + 3):
        loss, loss_ = iterate(train_loader)
    print(loss, loss_)

    # just building the filename...
    model_datasets = ''
    for corpus in corpus_list:
        if '-' in corpus:
            _corpus = corpus.split('-')
            model_datasets += _corpus[0][0] + _corpus[1][0]
        else:
            model_datasets += corpus[0]

    filename = TRAINED_ET_MODEL_DIR.format(model_datasets)
    if eval(args.filter_vocab):
        filename += '-UNK'
    if eval(args.finetune_elmo_embeddings):
        filename += '-ELMo'

    torch.save({
        'model_state_dict': model.state_dict(),
        'corpus_aggregator': dataset
    }, filename)
    print('Model trained on', corpus_list, 'saved to:', filename)
