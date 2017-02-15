""" Defines a class to model a Dataset.

A Dataset contains a vocabulary and two dictionaries of numpy arrays (X and Y).
X contains sentences and Y contains the corresponding answers (encoded as lists
of indices, defined thanks to the vocabulary).
"""
from __future__ import print_function
import numpy as np

from seq2seq_chatbot.data import io
from seq2seq_chatbot.data.classes.vocabulary import Vocabulary

class Dataset(object):
	@staticmethod
	def load(voc_path,X_path,Y_path):
		vocabulary = Vocabulary(json=voc_path)
		X_by_buckets = io.load_numpy_arrays(X_path)
		Y_by_buckets = io.load_numpy_arrays(Y_path)
		return Dataset(vocabulary, X_by_buckets, Y_by_buckets)

	@staticmethod
	def load_from_folder(folder_path):
		print("Loading Dataset from",folder_path)
		return Dataset.load(voc_path=folder_path + 'voc.json',
			X_path=folder_path + 'X.npz',
			Y_path=folder_path + 'Y.npz')

	@staticmethod
	def get_batchs(X,Y,batch_size,batch_id):
		assert batch_id in range(len(X)//batch_size)
		start_index = batch_id * batch_size
		end_index = (batch_id + 1) * batch_size
		return X[start_index:end_index], Y[start_index:end_index]

	@staticmethod
	def get_voc_path(dataset_folder):
		return dataset_folder + 'voc.json'

	def __init__(self,vocabulary,X_by_buckets,Y_by_buckets):
		self.vocabulary = vocabulary
		assert all(bucket_str in Y_by_buckets.keys()
			for bucket_str in X_by_buckets.keys())
		assert all(bucket_str in X_by_buckets.keys()
			for bucket_str in Y_by_buckets.keys())
		self.X_by_buckets = X_by_buckets
		self.Y_by_buckets = Y_by_buckets
		self.buckets = self.X_by_buckets.keys()

	def get_buckets(self):
		return self.buckets

	def get_data_for_bucket(self,bucket):
		bucket_str = str(bucket)
		assert bucket_str in self.X_by_buckets.keys()
		assert bucket_str in self.Y_by_buckets.keys()
		X = self.X_by_buckets[bucket_str]
		Y = self.Y_by_buckets[bucket_str]
		return X, Y

	def split(self,bucket,test_proportion=0.0):
		train_X = None
		train_Y = None
		test_X = None
		test_Y = None
		X, Y = self.get_data_for_bucket(bucket)
		np.random.shuffle(X)
		np.random.shuffle(Y)
		assert len(X) == len(Y)
		if test_proportion > 0.0:
			test_index = int(test_proportion * len(X))
			train_X = X[test_index:]
			train_Y = Y[test_index:]
			test_X = X[:test_index]
			test_Y = Y[:test_index]
		else:
			train_X = X
			train_Y = Y
		return train_X, train_Y, test_X, test_Y

	def print_data(self):
		print("*** X and Y by buckets ***")
		for key in self.X_by_buckets.keys():
			print("=== Bucket",key,"===")
			print("X[",key,"] -->",self.X_by_buckets[key].shape)
			print("Y[",key,"] -->",self.Y_by_buckets[key].shape)
			print()
			for i in range(min(10,len(self.X_by_buckets[key]))):
				print(self.X_by_buckets[key][i])
				print("-->",self.Y_by_buckets[key][i])
				print()

	def save(self,voc_path,X_path,Y_path):
		self.vocabulary.save_as_json(voc_path)
		print("Vocabulary saved to",voc_path)
		io.save_numpy_arrays(self.X_by_buckets, X_path)
		print("X saved to",X_path)
		io.save_numpy_arrays(self.Y_by_buckets, Y_path)
		print("Y saved to",Y_path)

	def save_in_folder(self,folder_path):
		# create_folder(folder_path) makes sure the folder exists before saving
		io.create_folder(folder_path)
		self.save(voc_path=folder_path + 'voc.json',
			X_path=folder_path + 'X.npz',
			Y_path=folder_path + 'Y.npz')
