""" Defines a class to model a Dataset.

A Dataset contains a vocabulary and two dictionaries of numpy arrays (X and Y).
X contains sentences and Y contains the corresponding answers (encoded as lists
of indices, defined thanks to the vocabulary).
"""
from __future__ import print_function

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
		return Dataset.load(voc_path=folder_path + 'voc.json',
			X_path=folder_path + 'X.npz',
			Y_path=folder_path + 'Y.npz')

	def __init__(self,vocabulary,X_by_buckets,Y_by_buckets):
		self.vocabulary = vocabulary
		self.X_by_buckets = X_by_buckets
		self.Y_by_buckets = Y_by_buckets

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
		# create_folder(folder_path) makes sure the folder exists before saving.
		io.create_folder(folder_path)
		self.save(voc_path=folder_path + 'voc.json',
			X_path=folder_path + 'X.npz',
			Y_path=folder_path + 'Y.npz')
