from torch import cuda
USE_CUDA = cuda.is_available()

ET_FEATURES = ['FFD', 'GD', 'TRT', 'nFixations', 'GPT']

LSTM_HIDDEN_UNITS = 128

NUM_EPOCHS = 25
BATCH_SIZE = 64
INITIAL_LR = 1e-4
DROPOUT_PROB = 0.5

WORD_EMBED_DIM = 300

# Directories
WORD_EMBED_MODEL_DIR = 'models/GoogleNews-vectors-negative300.bin'
TRAINED_ET_MODEL_DIR = 'models/predictor-{}'
TASK_DATASET_DIR = '../data_tasks/'

"""
2019-07-25 16:52:34,066 epoch 5 - iter 0/469 - loss 3.14217377
2019-07-25 16:52:41,405 epoch 5 - iter 46/469 - loss 2.35255781
2019-07-25 16:52:47,755 epoch 5 - iter 92/469 - loss 2.44538708
2019-07-25 16:52:53,857 epoch 5 - iter 138/469 - loss 2.45914056
2019-07-25 16:52:59,854 epoch 5 - iter 184/469 - loss 2.43776652
2019-07-25 16:53:07,177 epoch 5 - iter 230/469 - loss 2.45482428
2019-07-25 16:53:13,413 epoch 5 - iter 276/469 - loss 2.43804137
2019-07-25 16:53:19,493 epoch 5 - iter 322/469 - loss 2.44157752
2019-07-25 16:53:25,583 epoch 5 - iter 368/469 - loss 2.41128844
2019-07-25 16:53:32,800 epoch 5 - iter 414/469 - loss 2.42439260
2019-07-25 16:53:38,804 epoch 5 - iter 460/469 - loss 2.42239722
2019-07-25 16:53:39,931 ----------------------------------------------------------------------------------------------------
2019-07-25 16:53:39,931 EPOCH 5 done: loss 2.4307 - lr 0.1000 - bad epochs 0
2019-07-25 16:53:57,028 DEV : loss 1.4311549663543701 - score 0.8527
2019-07-25 16:54:11,783 TEST : loss 1.405474066734314 - score 0.8227

"""
