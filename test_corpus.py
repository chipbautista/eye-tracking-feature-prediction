from argparse import ArgumentParser

import numpy as np

from trainer import Trainer
from data import CorpusAggregator
from model import load_pretrained_et_predictor

parser = ArgumentParser()
parser.add_argument('--test-corpus', default='')
parser.add_argument('--model-path', default='')
args = parser.parse_args()
print(args)

dataset = CorpusAggregator(
    [args.test_corpus],
    normalize_wrt_mean=True,
    filter_vocab=True)
trainer = Trainer(dataset)

test_loader = dataset._get_dataloader(
    indices=np.array(range(len(dataset.sentences))))
model, _, _ = load_pretrained_et_predictor(args.model_path)
loss, loss_, r2 = trainer.iterate(model, test_loader)
