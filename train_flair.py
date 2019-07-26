
from argparse import ArgumentParser

from flair.models import SequenceTagger
from flair.trainers import ModelTrainer
from flair.data_fetcher import NLPTaskDataFetcher, NLPTask
from flair.embeddings import (WordEmbeddings, StackedEmbeddings,
                              PooledFlairEmbeddings)
# from typing import List

from model import EyeTrackingFeatureEmbedding
from settings import TASK_DATASET_DIR


parser = ArgumentParser()
parser.add_argument('--model-path', default=None)
args = parser.parse_args()


# 1. get the corpus
corpus = NLPTaskDataFetcher.load_corpus(
    NLPTask.CONLL_03, base_path=TASK_DATASET_DIR)

# 2. what tag do we want to predict?
tag_type = 'ner'

# 3. make the tag dictionary from the corpus
tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)

# initialize embeddings
embedding_types = [
    # GloVe embeddings
    WordEmbeddings('glove'),

    # --- Can try running this without their embeddings,
    # to keep things simple! ---

    # contextual string embeddings, forward
    # PooledFlairEmbeddings('news-forward', pooling='min'),
    # contextual string embeddings, backward
    # PooledFlairEmbeddings('news-backward', pooling='min'),
]

if args.model_path:
    embedding_types.append(
        EyeTrackingFeatureEmbedding(args.model_path))

"""
FRIDAY 748 AM
Running now: 4th tab, 3 embeddings w/o gaze (stopped)
5th tab: glove only f1 0.8477 after 10 epochs, 1.9559 train loss
6th tab: glove + Z1Z2Z3PGU . 0.8418 after 10 epochs, 1.9050 train loss

5th tab: glove only for 150 epochs
"""

embeddings = StackedEmbeddings(embeddings=embedding_types)
tagger = SequenceTagger(hidden_size=256,
                        embeddings=embeddings,
                        tag_dictionary=tag_dictionary,
                        tag_type=tag_type)

trainer = ModelTrainer(tagger, corpus)

# import pdb; pdb.set_trace()

trainer.train('resources/taggers/example-ner',
              max_epochs=150,
              learning_rate=0.1,  # default
              save_final_model=False,
              mini_batch_size=32  # default
              )
