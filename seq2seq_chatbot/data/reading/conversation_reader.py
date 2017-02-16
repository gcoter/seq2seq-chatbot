""" Defines a class to read conversations from the Cornell Movie Dialogs
Corpus.
"""
from __future__ import print_function
import pandas as pd
import numpy as np
import nltk
import time
from collections import Counter
import copy

from seq2seq_chatbot import constants
from seq2seq_chatbot.time_estimation import seconds2minutes, ReadingTimeEstimator
from seq2seq_chatbot.data.classes.dataset import Dataset
from seq2seq_chatbot.data.classes.vocabulary import Vocabulary

class ConversationReader(object):
	# Tokenizer (helpful to extract tokens from sentences).
	sentence_detector = nltk.data.load('tokenizers/punkt/english.pickle')

	@staticmethod
	def read_text_file(file_path,names):
		return pd.read_csv(
			file_path,
			delimiter=' \+\+\+\$\+\+\+ ',
			names=names,
			header=None,
			engine='python')

	@staticmethod
	def string_to_tokens(string_or_list):
		tokens = []
		if isinstance(string_or_list, (str, unicode)):
			sub_sentences = ConversationReader.sentence_detector.tokenize(
				unicode(string_or_list.strip(), errors='ignore'))
			for sub_sentence in sub_sentences:
				words = nltk.word_tokenize(sub_sentence)
				# Each word is converted to lowercase
				for i in range(len(words)):
					words[i] = words[i].lower()
				tokens.extend(words)
				tokens.append(constants.EOS)
		elif isinstance(string_or_list, (list, tuple)):
			tokens = copy.deepcopy(string_or_list)
		else:
			print("ERROR: turn should be String or List")
			tokens = None
		return tokens

	@staticmethod
	def get_right_bucket(buckets,x,y):
		right_bucket = None
		i = 0
		while(right_bucket is None and i < len(buckets)):
			bucket = buckets[i]
			if bucket[0] >= len(x) and bucket[1] >= len(y):
				right_bucket = bucket
			else:
				i += 1
		if right_bucket is None:
			print("Bucket not found to fit (",len(x),",",len(y),")")
		return right_bucket

	@staticmethod
	def pad_indices(indices,new_size,pad_idx):
		indices.extend((new_size-len(indices))*[pad_idx])

	@staticmethod
	def pad_x_and_y(buckets,x,y,pad_idx):
		bucket = ConversationReader.get_right_bucket(buckets,x,y)
		if bucket is not None:
			size_x = bucket[0]
			size_y = bucket[1]
			ConversationReader.pad_indices(x, size_x, pad_idx)
			ConversationReader.pad_indices(y, size_y, pad_idx)
		return bucket

	def __init__(self,movie_lines_path,movie_conversations_path):
		self.movie_lines_path = movie_lines_path
		self.movie_conversations_path = movie_conversations_path
		# Load data in Pandas DataFrame
		print("Reading movie lines from",self.movie_lines_path)
		self.movie_lines_df = ConversationReader.read_text_file(
			self.movie_lines_path,
			names=['LineId','CharacterId','MovieId','CharacterName','Sentence'])
		print("Reading movie conversations from",self.movie_conversations_path)
		self.movie_conversations_df = ConversationReader.read_text_file(
			self.movie_conversations_path,
			names=['CharacterID1','CharacterID2','MovieId','Utterances'])
		# Cleaning
		print("Cleaning data...")
		self.movie_lines_df.loc[self.movie_lines_df['Sentence'].isnull(),'Sentence'] = ""

	def create_vocabulary(self,voc_size):
		sentences = self.movie_lines_df['Sentence']
		print("Creating vocabulary...")
		# Note: Each word is converted to lowercase
		words = [word.lower()
			for sentence in sentences
			for word in ConversationReader.string_to_tokens(sentence)]
		counter = Counter(words)
		# (voc_size - 2) because we add PAD and UKN
		most_common_tokens = counter.most_common(voc_size - 2)
		most_common_tokens = [item[0] for item in most_common_tokens]
		most_common_tokens.insert(0,constants.PAD)
		most_common_tokens.insert(0,constants.UKN)
		print(len(most_common_tokens),"different tokens")
		return Vocabulary(most_common_tokens=most_common_tokens)

	def add_to_X(self,X,turn):
		# Get tokens
		tokens = ConversationReader.string_to_tokens(turn)
		# Encode tokens into indices
		indices = self.vocabulary.tokens_to_indices(tokens)
		self.add_indices_to_X(X,indices)

	def add_indices_to_X(self,X,indices):
		indices = copy.deepcopy(indices)
		X.append(indices)

	def add_to_Y(self,Y,turn):
		# Get tokens
		tokens = ConversationReader.string_to_tokens(turn)
		# Encode tokens into indices
		indices = self.vocabulary.tokens_to_indices(tokens)
		Y.append(indices)

	def read_one_conversation(self,movie_conversation,verbose=False):
		utterances = movie_conversation['Utterances']
		utterances = utterances.replace("[","")
		utterances = utterances.replace("]","")
		utterances = utterances.replace("'","")
		utterances = utterances.replace(" ","")
		utterances = utterances.split(",")
		new_turn_str = None
		previousCharacterId = None
		firstTurn = None
		secondTurn = None
		X = []
		Y = []
		for lineId in utterances:
			movie_line = self.movie_lines_df[self.movie_lines_df['LineId'] == lineId]
			characterId = movie_line['CharacterId'].tolist()[0]
			turn_str = movie_line['Sentence'].tolist()[0]
			if verbose:
				print()
				print("LineId:",lineId)
				print("CharacterID:",characterId)
				print("Turn:",turn_str)
			if previousCharacterId is not None:
				if characterId == previousCharacterId:
					# Same character speaking
					new_turn_str += turn_str
				else:
					# Answer from another character
					if firstTurn is None:
						# First sentence of the conversation
						self.add_to_X(X,new_turn_str)
						firstTurn = X[-1]
					else:
						if secondTurn is None:
							# Second sentence of the conversation
							self.add_to_Y(Y,new_turn_str)
							secondTurn = Y[-1]
						else:
							# In the middle of the conversation
							previous_indices = Y[-1]
							self.add_indices_to_X(X,previous_indices)
							self.add_to_Y(Y,new_turn_str)
			new_turn_str = turn_str
			previousCharacterId = characterId
		# Last sentence
		if secondTurn is None:
			# If last sentence is the second sentence of the conversation
			self.add_to_Y(Y,new_turn_str)
			secondTurn = Y[-1]
		else:
			# Otherwise...
			previous_indices = Y[-1]
			self.add_indices_to_X(X,previous_indices)
			self.add_to_Y(Y,new_turn_str)
		return X,Y

	def read(self,verbose=False,max=-1,display_step=100):
		X = []
		Y = []
		# Start reading
		num_conversations = len(self.movie_conversations_df)
		print(num_conversations,"conversations found")
		if max == -1:
			print("All conversations will be read")
		else:
			print(max,"conversations will be read")
		num_conversations_to_read = num_conversations
		if max != -1:
			num_conversations_to_read = max
		conversationCount = 0
		start = time.time()
		time_estimator = ReadingTimeEstimator(num_conversations_to_read)
		for index, movie_conversation in self.movie_conversations_df.iterrows():
			if verbose:
				print()
				print("*** Conversation",conversationCount," ***")
			X_conversation, Y_conversation = self.read_one_conversation(
				movie_conversation,
				verbose=verbose)
			X.extend(X_conversation)
			Y.extend(Y_conversation)
			conversationCount += 1
			if conversationCount > 0 and conversationCount % display_step == 0:
				end = time.time()
				# Estimate time left
				num_conversations_left = num_conversations_to_read - conversationCount
				num_conversations_read_in_time_spent = display_step
				time_spent = end - start
				remaining_time = time_estimator.get_remaining_time(
					num_conversations_left,
					num_conversations_read_in_time_spent,
					time_spent)
				remaining_time_hours, remaining_time_minutes, remaining_time_seconds = seconds2minutes(remaining_time)
				print("{}/{} ({} h {} min {} s left)".format(
					conversationCount,
					num_conversations_to_read,
					remaining_time_hours,
					remaining_time_minutes,
					remaining_time_seconds))
				start = end
			if max != -1 and conversationCount > max:
				break
		print(conversationCount-1,"conversations read")
		return X, Y

	def padding(self,buckets,X,Y,display_step=100):
		""" Pad list of indices and classifies them by bucket. """
		X_by_buckets = {}
		Y_by_buckets = {}
		pad_idx = self.vocabulary.get_pad_index()
		print("\nPadding...")
		assert len(X) == len(Y)
		num_examples = len(X)
		for i in range(num_examples):
			x = X[i]
			y = Y[i]
			bucket = ConversationReader.pad_x_and_y(buckets, x, y, pad_idx)
			if bucket is not None:
				if str(bucket) not in X_by_buckets.keys():
					X_by_buckets[str(bucket)] = []
					Y_by_buckets[str(bucket)] = []
				X_by_buckets[str(bucket)].append(x)
				Y_by_buckets[str(bucket)].append(y)
			if i % display_step == 0:
				print(i,"/",num_examples)
		# Convert to numpy arrays
		for bucket in X_by_buckets.keys():
			X_by_buckets[bucket] = np.array(X_by_buckets[bucket])
			Y_by_buckets[bucket] = np.array(Y_by_buckets[bucket])
		return X_by_buckets, Y_by_buckets

	def construct_dataset(self,
						  voc_size,
						  buckets,
						  dataset_folder=None,
						  verbose=False,
						  max=-1,
						  display_step=100):
		""" Returns a Dataset object. """
		dataset = None
		t_0 = time.time()
		# Create Vocabulary
		self.vocabulary = self.create_vocabulary(voc_size)
		t_1 = time.time()
		print(t_1 - t_0,"s")
		# Read conversations
		X, Y = self.read(verbose=verbose,max=max,display_step=display_step)
		t_2 = time.time()
		print(t_2 - t_1,"s")
		# Pad all entries
		X_by_buckets, Y_by_buckets = self.padding(
			buckets,
			X,
			Y,
			display_step=display_step)
		# Create the dataset object
		dataset = Dataset(self.vocabulary, X_by_buckets, Y_by_buckets)
		if dataset_folder is not None:
			dataset.save_in_folder(dataset_folder)
		return dataset
