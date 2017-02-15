""" Defines the Chatbot class.
"""
from __future__ import print_function
import tensorflow as tf
import numpy as np
import os.path
import random
import time

from seq2seq_chatbot.data import io
from seq2seq_chatbot.data.classes.dataset import Dataset
from seq2seq_chatbot.data.classes.vocabulary import Vocabulary
from seq2seq_chatbot.data.reading.conversation_reader import ConversationReader
from seq2seq_chatbot.models.seq2seq import BasicSeq2Seq
from time_estimation import seconds2minutes, TrainingTimeEstimator

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

	@staticmethod
	def get_config_path(config_folder):
		return config_folder + 'config.json'

	@staticmethod
	def get_parameters_folder(config_folder):
		return config_folder + 'parameters/'

	@staticmethod
	def get_parameters_path(parameters_folder):
		return parameters_folder + 'parameters.ckpt'

	"""Defines a Chatbot which can be trained and can chat."""
	def __init__(self,
				 config_folder,
				 buckets=None,
				 vocabulary_path=None,
				 embedding_size=None,
				 num_hidden=None,
				 num_rnns=None):
		config_path = Chatbot.get_config_path(config_folder)
		if os.path.isfile(config_path):
			# config_path is assumed to be the path to a JSON file
			config_dict = io.load_object_from_json(config_path)
			self.buckets = config_dict['buckets']
			self.vocabulary_path = config_dict['vocabulary_path']
			self.embedding_size = config_dict['embedding_size']
			self.num_hidden = config_dict['num_hidden']
			self.num_rnns = config_dict['num_rnns']
		else:
			self.buckets = buckets
			self.vocabulary_path = vocabulary_path
			self.embedding_size = embedding_size
			self.num_hidden = num_hidden
			self.num_rnns = num_rnns
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
		self.seq2seq_class = BasicSeq2Seq
		# For now, a BasicSeq2Seq based on the first bucket is used.
		some_bucket = self.buckets[0]
		print("Defining generation model...")
		with tf.variable_scope("models"):
			self.generation_model = self.seq2seq_class(
				voc_size=voc_size,
				embedding_size=self.embedding_size,
				batch_size=1,
				num_hidden=self.num_hidden,
				num_rnns=self.num_rnns,
				input_seq_length=some_bucket[0],
				output_seq_length=some_bucket[1],
				name="seq2seq_model")
		self.training_model = None

	def get_config_dict(self):
		return {
			'buckets': self.buckets,
			'vocabulary_path': self.vocabulary_path,
			'embedding_size': self.embedding_size,
			'num_hidden': self.num_hidden,
			'num_rnns': self.num_rnns
		}

	def initialize_parameters(self,session):
		self.generation_model.initialize_parameters(session)
		if self.training_model is not None:
			self.training_model.initialize_parameters(session)

	def restore_parameters(self,session,parameters_path):
		# Restore variables from disk
		print("Restoring model...")
		saver = tf.train.Saver()
		saver.restore(session, parameters_path)
		print("Model restored from file " + parameters_path)

	def restore_parameters_from_config_folder(self,session,config_folder):
		parameters_folder = Chatbot.get_parameters_folder(config_folder)
		parameters_path = Chatbot.get_parameters_path(parameters_folder)
		self.restore_parameters(session, parameters_path)

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
			if len(indices) < new_size:
				ConversationReader.pad_indices(indices, new_size, pad_idx)
			probabilities = self.generation_model.get_probabilities(session,indices)
			generated_tokens = []
			for i in range(len(probabilities)):
				# Generate the ith token of the answer
				generated_index = Chatbot.sample(probabilities[i], temperature=temperature)
				generated_token = self.vocabulary.index_to_token(generated_index)
				generated_tokens.append(generated_token)
			answer = self.vocabulary.tokens_to_string(generated_tokens)
		else:
			answer = "Invalid input: your sentence should be shorter than "
			answer += str(self.buckets[-1][0]) + " tokens."
		return answer

	def answer_from_tokens(self,session,tokens,temperature=1.0):
		indices = self.vocabulary.tokens_to_indices(tokens)
		return self.answer_from_indices(session, indices, temperature=temperature)

	def answer(self,session,sentence,temperature=1.0):
		sentence = sentence.lower()
		tokens = ConversationReader.string_to_tokens(sentence)
		return self.answer_from_tokens(session, tokens, temperature=temperature)

	def test_on_valid_data(self,session,valid_X,valid_Y,batch_size):
		if self.training_model is not None:
			num_valid_examples = len(valid_X)
			num_steps = num_valid_examples//batch_size
			avg_loss = 0.0
			for step in range(num_steps):
				batch_X, batch_Y = Dataset.get_batchs(
					valid_X,
					valid_Y,
					batch_size,
					batch_id=step)
				loss_value = self.training_model.get_test_loss(
					session,
					batch_X,
					batch_Y)
				avg_loss += loss_value
			return avg_loss/num_steps
		else:
			print("Can't test on valid data if training_model is None")

	def save_config_as_json(self,json_path):
		config_dict = self.get_config_dict()
		print("Saving configuration to",json_path)
		io.save_object_as_json(config_dict, json_path)

	def save_config_in_folder(self,config_folder):
		# create_folder(folder_path) makes sure the folder exists before saving
		io.create_folder(config_folder)
		config_path = Chatbot.get_config_path(config_folder)
		self.save_config_as_json(json_path=config_path)

	def save_parameters(self,session,parameters_path):
		saver = tf.train.Saver()
		print("Saving parameters to",parameters_path)
		saver.save(session,parameters_path)

	def save_parameters_in_folder(self,session,parameters_folder):
		# create_folder(folder_path) makes sure the folder exists before saving
		io.create_folder(parameters_folder)
		parameters_path = Chatbot.get_parameters_path(parameters_folder)
		self.save_parameters(session, parameters_path)

	def save(self,session,config_path,parameters_path):
		self.save_config_as_json(json_path=config_path)
		self.save_parameters(session,parameters_path)

	def save_in_folder(self,session,config_folder):
		parameters_folder = Chatbot.get_parameters_folder(config_folder)
		self.save_config_in_folder(config_folder)
		self.save_parameters_in_folder(session, parameters_folder)

	def train(self,
			  dataset,
			  num_epochs,
			  batch_size,
			  eval_step,
			  learning_rate,
			  keep_prob,
			  config_folder,
			  test_proportion=0.0,
			  restore=False):
		# For now, use only one bucket
		# TODO: Use several buckets
		input_seq_length = self.generation_model.input_seq_length
		output_seq_length = self.generation_model.output_seq_length
		bucket = (input_seq_length,output_seq_length)
		train_X, train_Y, test_X, test_Y = dataset.split(
			bucket,
			test_proportion=test_proportion)

		with tf.Session() as session:
			num_sequences = len(train_X)
			num_steps_per_epoch = num_sequences//batch_size
			time_estimator = TrainingTimeEstimator(num_steps_per_epoch)

			# This model is used for training
			# It shares its parameters with the generation model
			input_seq_length = self.generation_model.input_seq_length
			output_seq_length = self.generation_model.output_seq_length
			voc_size = self.vocabulary.get_voc_size()

			with tf.variable_scope("models") as scope:
				print("Defining training model...")
				scope.reuse_variables()
				self.training_model = self.seq2seq_class(
					voc_size=voc_size,
					embedding_size=self.embedding_size,
					batch_size=batch_size,
					num_hidden=self.num_hidden,
					num_rnns=self.num_rnns,
					input_seq_length=input_seq_length,
					output_seq_length=output_seq_length,
					name="seq2seq_model")

			# Restore parameters
			# Otherwise, simply initialize parameters
			if restore:
				self.restore_parameters_from_config_folder(session, config_folder)
			else:
				self.initialize_parameters(session)

			print("START TRAINING (",num_epochs,"epochs,",num_steps_per_epoch,"steps per epoch with batch size =",batch_size,")")
			begin_time = time_0 = time.time()
			for epoch_id in range(num_epochs):
				print("*** EPOCH",epoch_id,"***")
				for step in range(num_steps_per_epoch):
					# TODO: Use several buckets
					batch_X, batch_Y = Dataset.get_batchs(
						train_X,
						train_Y,
						batch_size,
						batch_id=step)
					self.training_model.train_step(
						session=session,
						batch_X=batch_X,
						batch_Y=batch_Y,
						learning_rate=learning_rate,
						keep_prob=keep_prob)
					absolute_step = epoch_id * num_steps_per_epoch + step
					loss_value = self.training_model.get_train_loss(
						session,
						batch_X,
						batch_Y,
						keep_prob=1.0)
					if np.isnan(loss_value):
						print("LOSS = NAN at step =",step,"with batch_size =",batch_size)
						print("STOP TRAINING")
						return
					if step % eval_step == 0:
						print("Batch Loss =",loss_value,"at step",absolute_step)
						# Evaluate on test data if it is defined
						if test_X is not None and test_Y is not None:
							test_loss = self.test_on_valid_data(session,test_X,test_Y,batch_size)
							print("Test Loss =",test_loss,"at step",absolute_step)
							# Print an example
							test_indices = random.choice(test_X)
							sentence = self.vocabulary.indices_to_string(test_indices)
							answer = self.answer_from_indices(
								session,
								test_indices,
								temperature=1.0)
							print("*** Example ***")
							print(sentence)
							print("--> %s" % answer)
							print()
						# Time spent is measured
						if absolute_step > 0:
							t = time.time()
							time_spent = t - time_0
							time_0 = t
							# Estimate remaining time
							num_epochs_left = num_epochs - epoch_id - 1
							num_steps_left_in_current_epoch = num_steps_per_epoch - step - 1
							num_steps_processed_in_time_spent = eval_step
							remaining_time = time_estimator.get_remaining_time(
								num_epochs_left,
								num_steps_left_in_current_epoch,
								num_steps_processed_in_time_spent,
								time_spent)
							remaining_time_hours, remaining_time_minutes, remaining_time_seconds = seconds2minutes(remaining_time)
							print("Time left: {} h {} min {} s".format(
								remaining_time_hours,
								remaining_time_minutes,
								remaining_time_seconds))
				# Save config and parameters
				self.save_in_folder(session, config_folder)
			total_time = time.time() - begin_time
			total_time_hours, total_time_minutes, total_time_seconds = seconds2minutes(total_time)
			print("*** Total time to compute",num_epochs,"epochs:",total_time_hours,"hours,",total_time_minutes,"minutes and",total_time_seconds,"seconds ***")
