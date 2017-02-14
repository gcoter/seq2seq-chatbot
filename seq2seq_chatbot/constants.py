""" This module defines some constants (used as default values).

It also helps to define the filesystem required for the whole program to work.
Here is a small tree representation:

PROJECT_FOLDER
|-- DATA_FOLDER
|   |-- CORNELL_FOLDER
|-- DATASETS_FOLDER
|   |-- DEFAULT_DATASET
|   |   |-- X.npz
|   |   |-- Y.npz
|   |   |-- voc.json
|-- CONFIGS_FOLDER
|   |-- DEFAULT_CONFIG_FOLDER
|   |   |-- config.json
|   |   |-- DEFAULT_PARAMETERS_FOLDER
|   |   |   |-- parameters.ckpt

It is possible to store different datasets and different configurations as long
as this structure is respected.
"""
# Cornell Movie Dialogs Corpus Raw Data
DATA_FOLDER = '../data/'
CORNELL_FOLDER = DATA_FOLDER + 'cornell-movie-dialogs-corpus/'
MOVIE_CONVERSATIONS_PATH = CORNELL_FOLDER + 'movie_conversations.txt'
MOVIE_LINES_PATH = CORNELL_FOLDER + 'movie_lines.txt'

# Dataset
DATASETS_FOLDER = '../datasets/'
DEFAULT_DATASET = DATASETS_FOLDER + 'default/'
# Data Set files
DEFAULT_X_PATH = DEFAULT_DATASET + 'X.npz'
DEFAULT_Y_PATH = DEFAULT_DATASET + 'Y.npz'
# Vocabulary
DEFAULT_VOC_SIZE = 70000
DEFAULT_VOC_PATH = DEFAULT_DATASET + 'voc.json'

# Configurations
CONFIGS_FOLDER = '../configs/'
DEFAULT_CONFIG_FOLDER = CONFIGS_FOLDER + 'default/'
# Configuration File
DEFAULT_CONFIG_FILE_PATH = DEFAULT_CONFIG_FOLDER + 'config.json'
# Parameters
DEFAULT_PARAMETERS_FOLDER = DEFAULT_CONFIG_FOLDER + 'parameters/'
DEFAULT_PARAMETERS_PATH = DEFAULT_PARAMETERS_FOLDER + 'parameters.ckpt'

# Special Tokens
GO = u"<go>"
EOS = u"<eos>"
PAD = u"<pad>"
UKN = u"<ukn>"

# Buckets
# This list is used to make small tests
DEFAULT_BUCKETS = [(10,10)]
# The list below must be ordered
ALL_BUCKETS = [(10*(i+1), 10*(i+1)) for i in range(9)] + [(800,800)]
# The list below is the one actually used in the program
BUCKETS = DEFAULT_BUCKETS

# Hyperparameters
EMBEDDING_SIZE = 100
NUM_FEATURES = 1
BATCH_SIZE = 10
LEARNING_RATE = 1e-3
NUM_HIDDEN = 128  # LSTM cell
NUM_RNNS = 2  # Number of LSTM cells
# For training
TEST_PROPORTION = 0.01
NUM_EPOCHS = 1
KEEP_PROB = 0.5
EVAL_STEP = 1
