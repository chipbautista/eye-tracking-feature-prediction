import time
from argparse import ArgumentParser

import torch
import numpy as np

from data import CorpusAggregator
from trainer import Trainer
from settings import *


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
parser.add_argument('--filter-vocab', default='True')
parser.add_argument('--normalize-wrt-mean', default='False')
parser.add_argument('--train-per-sample', default='False')
parser.add_argument('--normalizer', default='std')
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
                           static_embedding=args.static_embedding,
                           corpus_normalizer=args.normalizer)
trainer = Trainer(dataset)

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
te_r2 = []
for k, (train_loader, test_loader) in enumerate(
        dataset.split_cross_val(stratified=False)):
    _start_time = time.time()

    if k == 0:
        print('Train #batches:', len(train_loader))
        print('Test #batches:', len(test_loader))

    model, optimizer = _get_model_and_optim()
    optim_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.1, patience=2, verbose=True)

    best_epochs = []
    e_tr_losses = []
    e_tr_losses_ = []
    e_tr_r2 = []
    e_te_losses = []
    e_te_losses_ = []
    e_te_r2 = []
    for e in range(eval(args.num_epochs)):
        model, optimizer = trainer.init_model(args)
        model.train()
        train_loss, train_loss_, train_r2 = trainer.iterate(
            model, optimizer, train_loader)

        model.eval()
        test_loss, test_loss_, test_r2 = trainer.iterate(
            model, optimizer, test_loader)
        optim_scheduler.step(test_loss)

        e_tr_losses.append(train_loss)
        e_tr_losses_.append(train_loss_)
        e_tr_r2.append(train_r2)
        e_te_losses.append(test_loss)
        e_te_losses_.append(test_loss_)
        e_te_r2.append(test_r2)

        # print('k:', k, 'e:', e,
        #       '{:.5f}'.format(train_loss), '{:.5f}'.format(test_loss))
        # print(train_loss_)
        # print(test_loss_)
        # print(train_r2)
        # print(test_r2)

    best_epoch = np.argmin(e_te_losses)
    best_epochs.append(best_epoch)
    te_losses.append(e_te_losses[best_epoch])
    te_losses_.append(e_te_losses_[best_epoch])
    te_r2.append(e_te_r2[best_epoch])

    print(k, '[e={}] '.format(best_epoch),
          '- Train rMSE: {:.5f}'.format(e_tr_losses[best_epoch]),
          'Test rMSE: {:.5f} '.format(e_te_losses[best_epoch]),
          '({:.2f}s)'.format(time.time() - _start_time))
    print('Train rMSE:', e_tr_losses_[best_epoch])
    print('Test rMSE:', e_te_losses_[best_epoch])
    print('Train R2:', e_tr_r2[best_epoch])
    print('Test R2:', e_te_r2[best_epoch])

print('\nCV Mean Test Loss:', np.mean(te_losses))
print('Mean rMSE Loss Per Feature:', torch.stack(te_losses_).mean(0))
print('Mean R2 Score Per Feature:', torch.stack(te_r2).mean(0))


if args.save_model is not False:
    mean_epoch = int(round(np.mean(best_epochs)))
    print('Mean number of epochs until overfit:', mean_epoch)
    print('Will save final model. Will now train on all data points in',
          mean_epoch, 'epochs.')

    train_loader = dataset._get_dataloader(
        indices=np.array(range(len(dataset.sentences))))

    for e in range(mean_epoch):
        loss, loss_, r2 = trainer.iterate(
            *trainer.init_model(args), train_loader)
    print(loss, loss_, r2)

    # just building the filename...
    model_datasets = ''
    for corpus in corpus_list:
        if '-' in corpus:
            _corpus = corpus.split('-')
            model_datasets += _corpus[0][0] + _corpus[1][0]
        else:
            model_datasets += corpus[0]

    filename = TRAINED_ET_MODEL_DIR.format(model_datasets)
    # if eval(args.filter_vocab):
    #     filename += '-UNK'
    # if eval(args.finetune_elmo_embeddings):
    #     filename += '-ELMo'
    if eval(args.train_per_sample):
        filename += '-persample'
    if args.static_embedding:
        filename += '-' + args.static_embedding
    if eval(args.normalize_wrt_mean):
        filename += '-wrtmean'
    if eval(args.minmax_aggregate):
        filename += '-minmaxaggr'
    filename += '-meanepoch'

    torch.save({
        'model_state_dict': model.state_dict(),
        'corpus_aggregator': dataset
    }, filename)
    print('Model trained on', corpus_list, 'saved to:', filename)
