""" Defines the Chatbot class.
"""
from __future__ import print_function
import tensorflow as tf
import numpy as np
import random

from seq2seq_chatbot.data import io
from seq2seq_chatbot.data.classes.vocabulary import Vocabulary
from seq2seq_chatbot.models.seq2seq import BasicSeq2Seq
from seq2seq_chatbot.data.reading.conversation_reader import ConversationReader

class Chatbot(object):
	@staticmethod
	def sample(probabilities,temperature=1.0):
		probabilities = np.log(probabilities+1e-30) / temperature
		probabilities = np.exp(probabilities) / np.sum(np.exp(probabilities))
		r = random.random()  # range: [0,1)
		total = 0.0
		for i in range(len(probabilities)):
			total += probabilities[i]
			if total > r:
				return i
		return len(probabilities)-1

	"""Defines a Chatbot which can be trained and can chat."""
	def __init__(self,
				 buckets=None,
				 vocabulary_path=None,
				 embedding_size=None,
				 num_hidden=None,
				 num_rnns=None,
				 parameters_path=None,
				 config_path=None):
		if config_path is not None:
			# config_path is assumed to be the path to a JSON file
			config_dict = io.load_object_from_json(config_path)
			self.buckets = config_dict['buckets']
			self.vocabulary_path = config_dict['vocabulary_path']
			self.embedding_size = config_dict['embedding_size']
			self.num_hidden = config_dict['num_hidden']
			self.num_rnns = config_dict['num_rnns']
			self.parameters_path = config_dict['parameters_path']
		else:
			self.buckets = buckets
			self.vocabulary_path = vocabulary_path
			self.embedding_size = embedding_size
			self.num_hidden = num_hidden
			self.num_rnns = num_rnns
			self.parameters_path = parameters_path
		assert self.buckets is not None
		assert self.vocabulary_path is not None
		assert self.embedding_size is not None
		assert self.num_hidden is not None
		assert self.num_rnns is not None
		self.vocabulary = Vocabulary(json=self.vocabulary_path)
		voc_size = self.vocabulary.get_voc_size()
		print("Using the vocabulary from {} ({} tokens)".format(
			self.vocabulary_path,
			voc_size))
		# A Generation Model takes batch of size 1 as input.
		# It is only used to generate answers.
		# TODO: Replace BasicSeq2Seq with Seq2SeqWithBuckets
		# For now, a BasicSeq2Seq based on the first bucket is used.
		some_bucket = self.buckets[0]
		print("Defining generation model...")
		self.generation_model = BasicSeq2Seq(
			voc_size=voc_size,
			embedding_size=self.embedding_size,
			batch_size=1,
			num_hidden=self.num_hidden,
			num_rnns=self.num_rnns,
			input_seq_length=some_bucket[0],
			output_seq_length=some_bucket[1],
			name="seq2seq_model")
		self.training_model = None

	def initialize_parameters(self,session):
		self.generation_model.initialize_parameters(session)
		if self.training_model is not None:
			self.training_model.initialize_parameters(session)

	def restore_parameters(self,session):
		if self.parameters_path is not None:
			# Restore variables from disk
			print("Restoring model...")
			saver = tf.train.Saver()
			saver.restore(session, self.parameters_path)
			print("Model restored from file " + self.parameters_path)
		else:
			print("Parameters file not found at",self.parameters_path)
			print("Initialize parameters from scratch")
			self.initialize_parameters(session)

	def get_x_size(self,x):
		size = None
		i = 0
		while size is None and i < len(self.buckets):
			bucket = self.buckets[i]
			if bucket[0] >= len(x):
				size = bucket[0]
			i += 1
		return size

	def answer_from_indices(self,session,indices,temperature=1.0):
		answer = ""
		new_size = self.get_x_size(indices)
		if new_size is not None:
			pad_idx = self.vocabulary.get_pad_index()
			ConversationReader.pad_indices(indices, new_size, pad_idx)
			probabilities = self.generation_model.get_probabilities(session,indices)
			for i in range(len(probabilities)):
				# Generate the ith token of the answer
				generated_index = Chatbot.sample(probabilities[i], temperature=temperature)
				generated_token = self.vocabulary.index_to_token(generated_index)
				answer += generated_token + ' '
		else:
			answer = "Invalid input: your sentence should be shorter than "
			answer += str(self.buckets[-1][0]) + " tokens."
		return answer

	def answer_from_tokens(self,session,tokens,temperature=1.0):
		indices = self.vocabulary.tokens_to_indices(tokens)
		return self.answer_from_indices(session, indices, temperature=temperature)

	def answer(self,session,sentence,temperature=1.0):
		tokens = ConversationReader.string_to_tokens(sentence)
		return self.answer_from_tokens(session, tokens, temperature=temperature)
