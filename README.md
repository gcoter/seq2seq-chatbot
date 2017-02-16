# seq2seq-chatbot
## Introduction
I did this project as part of my studies at the [Kyoto Institute of Technology](https://www.kit.ac.jp/en/) from October 2016 to March 2017.

The goal was to implement a Chatbot using the seq2seq framework.

## Dataset
### Choosing a Dataset
At first, we wanted to use the [OpenSubtitles2011](http://opus.lingfil.uu.se/OpenSubtitles2011.php) data set (as Google did in [this paper](http://arxiv.org/pdf/1506.05869.pdf)). However, in this dataset, it's impossible to check whether two consecutive sentences were uttered by two different characters. As a consequence, there is some undesirable "noise".

To solve this issue, I chose the [Cornell Movie Dialogs Corpus](https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html). This corpus not only provides dialogs from movies subtitles but also metadata related to the characters and the movies. In particular, each utterance is related to one character. However, it contains less data.

### Preprocessing
Thanks to those information, it's possible to construct two different sets:

- X: contains sentences (encoded into list of indices).
- Y: contains answers to X's sentences (encoded into list of indices). For example, Y[0] is the answer to X[0].

This step is called "preprocess" and can be executed with the following command:

`python tool.py preprocess --dataset_folder /some/folder/ --voc_size some_int`

At the end of this step, you can find in the provided "dataset_folder":

- X.npz: a dictionary of numpy arrays. Each entry corresponds to a "bucket". For instance, X["(10,20)"] is a numpy array whose shape is (num_sentences,10) (each sentence is padded so that it contains exactly 10 tokens).
- Y.npz: a dictionary of numpy arrays. Each entry corresponds to a "bucket". For instance, Y["(10,20)"] is a numpy array whose shape is (num_sentences,20) (each sentence is padded so that it contains exactly 20 tokens).
- voc.json: a dictionary which maps tokens to their index. Those tokens are the voc_size most common tokens in the whole data set. This dictionary is used to encode sentences into list of indices.

## The Model
As we want to generate one sequence (an answer) from another sequence (some sentence), the seq2seq framework is a good start. I used TensorFlow to easily implement it. The model uses embeddings: each index (each token) is associated to a vector with a given length (--embedding_size parameter). Those vectors are used for the calculations and are updated throughout the training.

The model can be trained using this command:

`python tool.py train --dataset_folder /some/folder/ --config_folder /another/folder/`

*Note: There is a bunch of other parameters you can provide, type* `python tool.py train -h` *for more information.*

At the end of this step, you can find in the provided "config_folder":

- config.json: stores the hyperparameters necessary to rebuild the model and the path to the vocabulary file (voc.json).
- a parameters folder: stores the parameters of the trained model.

Thanks to those elements, it is possible to restore the model later on, to continue training (--restore parameter) or simply to chat with it.

## Chatting
Once you trained a model, you can try chatting with it thanks to this command:

`python tool.py chat --config_folder /some/folder/`

## Results
This is still a work in progress. I'm trying to find the good hyperparameters to make it work. I hope to get satisfactory results soon.
