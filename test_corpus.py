from argparse import ArgumentParser

import numpy as np

from trainer import Trainer
from data import CorpusAggregator
from model import load_pretrained_et_predictor

parser = ArgumentParser()
parser.add_argument('--test-corpus', default=None)
parser.add_argument('--model-path', default='')
args = parser.parse_args()
print(args)

if not args.test_corpus:
    corpus_list = ['ZuCo-1', 'ZuCo-2', 'ZuCo-3', 'PROVO', 'UCL', 'GECO']
else:
    corpus_list = [args.test_corpus]


model, vocab, _ = load_pretrained_et_predictor(args.model_path)

for corpus in corpus_list:
    print('=== Testing on: {} ==='.format(corpus))
    dataset = CorpusAggregator(
        [corpus],
        normalize_wrt_mean=True,
        filter_vocab=True,
        corpus_normalizer='std',
        vocabulary=vocab)

    trainer = Trainer(dataset)
    test_loader = dataset._get_dataloader(
        indices=np.array(range(len(dataset.sentences))))

    loss, loss_, r2 = trainer.iterate(model, test_loader)

    print('\nMean Test Loss:', loss)
    print('rMSE Loss per feature: ', loss_)
    print('R2 Scores per feature: ', r2)
