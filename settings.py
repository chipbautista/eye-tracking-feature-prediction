from torch import cuda
USE_CUDA = cuda.is_available()

ET_FEATURES = ['FFD', 'GD', 'TRT', 'nFixations', 'GPT']

LSTM_HIDDEN_UNITS = 64

NUM_EPOCHS = 50
BATCH_SIZE = 32
INITIAL_LR = 1e-3
DROPOUT_PROB = 0.5

WORD_EMBED_DIM = 300

# Directories
WORD_EMBED_MODEL_DIR = 'models/GoogleNews-vectors-negative300.bin'
TRAINED_ET_MODEL_DIR = '../08-24/models/predictor-{}'
TASK_DATASET_DIR = '../data_tasks/'
STATIC_EMBEDDING_DIR = '../data/_static_embeddings/{}-{}.pickle'
