""" Define a class to store and manipulate a vocabulary.

The vocabulary can be defined with two dictionaries (depending on each other):
* token_to_index_dict: define the mapping from token to index.
* index_to_token_dict: define the mapping from index to token.
"""
from seq2seq_chatbot import constants
from seq2seq_chatbot.data import io

class Vocabulary(object):
	def __init__(self,most_common_tokens=None,json=None):
		# Mapping from token to index
		self.token_to_index_dict = {}
		if json is not None:
			self.token_to_index_dict = io.load_object_from_json(json)
		elif most_common_tokens is not None:
			self.token_to_index_dict = dict((token, index)
				for index, token in enumerate(most_common_tokens))
		# Mapping from index to token
		self.index_to_token_dict = dict((index, token)
			for token, index in self.token_to_index_dict.iteritems())

	def get_voc_size(self):
		return len(self.token_to_index_dict.keys())

	def token_to_index(self,token):
		if token in self.token_to_index_dict.keys():
			return self.token_to_index_dict[token]
		else:
			return self.get_ukn_index()

	def tokens_to_indices(self,tokens):
		# Tokens is a list
		return [self.token_to_index(token) for token in tokens]

	def index_to_token(self,index):
		if index in self.index_to_token_dict.keys():
			return self.index_to_token_dict[index]
		else:
			return constants.UKN

	def indices_to_tokens(self,indices):
		# ints is a list
		return [self.index_to_token(index) for index in indices]

	def tokens_to_string(self,tokens):
		string = ""
		eos_counter = 0
		pad_counter = 0
		ukn_counter = 0
		for i in range(len(tokens)):
			token = tokens[i]
			if token == constants.EOS:
				eos_counter += 1
			elif token == constants.PAD:
				pad_counter += 1
			elif token == constants.UKN:
				ukn_counter += 1
			else:
				if eos_counter > 0:
					string += "<" + str(eos_counter) + " eos> "
					eos_counter = 0
				if pad_counter > 0:
					string += "<" + str(pad_counter) + " pad> "
					pad_counter = 0
				if ukn_counter > 0:
					string += "<" + str(ukn_counter) + " ukn> "
					ukn_counter = 0
			string += token + " "
		return string

	def indices_to_string(self,indices):
		tokens = self.indices_to_tokens(indices)
		return self.tokens_to_string(tokens)

	def get_ukn_index(self):
		return self.token_to_index_dict[constants.UKN]

	def get_pad_index(self):
		return self.token_to_index_dict[constants.PAD]

	def save_as_json(self,json_path):
		io.save_object_as_json(self.token_to_index_dict, json_path)
