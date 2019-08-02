
from argparse import ArgumentParser

from flair.models import SequenceTagger
from flair.trainers import ModelTrainer
from flair.data_fetcher import NLPTaskDataFetcher, NLPTask
from flair.embeddings import (WordEmbeddings, StackedEmbeddings,
                              PooledFlairEmbeddings)

from model import EyeTrackingFeatureEmbedding
from settings import TASK_DATASET_DIR


parser = ArgumentParser()
parser.add_argument('--model-path', default='NA')
parser.add_argument('--use-flair-embeddings', default=False)
parser.add_argument('--task', default='ner')
args = parser.parse_args()
print(args)


if args.task == 'ner':
    flair_corpus = NLPTask.CONLL_03
    tag_type = 'ner'
    embedding_types = [WordEmbeddings('glove')]
else:
    flair_corpus = NLPTask.CONLL_2000
    tag_type = 'np'
    embedding_types = [WordEmbeddings('extvec')]

_base_path = 'resources/taggers/{}-{}'.format(args.task, args.model_path)

if args.use_flair_embeddings is True:
    embedding_types.extend([
        # contextual string embeddings, forward
        PooledFlairEmbeddings('news-forward'),
        # contextual string embeddings, backward
        PooledFlairEmbeddings('news-backward')
    ])

if args.model_path != 'NA':
    embedding_types.append(
        EyeTrackingFeatureEmbedding(args.model_path))


corpus = NLPTaskDataFetcher.load_corpus(flair_corpus,
                                        base_path=TASK_DATASET_DIR)
tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)
embeddings = StackedEmbeddings(embeddings=embedding_types)
tagger = SequenceTagger(hidden_size=256,
                        embeddings=embeddings,
                        tag_dictionary=tag_dictionary,
                        tag_type=tag_type)

trainer = ModelTrainer(tagger, corpus)
trainer.train(_base_path,
              # max_epochs=100,
              learning_rate=0.1,  # default
              # min_learning_rate=0.001,  # default = 0.0001
              save_final_model=False,
              mini_batch_size=64,  # default = 32
              # embeddings_storage_mode='gpu'
              )
