""" This module defines the seq2seq TensorFlow graphs used by the Chatbot.
"""
import tensorflow as tf
import numpy as np

class Seq2Seq(object):
	"""Abstract class used to define Seq2Seq classes."""
	def __init__(self,
				 voc_size,
				 embedding_size,
				 batch_size,
				 num_hidden,
				 num_rnns):
		self.voc_size = voc_size
		self.embedding_size = embedding_size
		self.batch_size = batch_size
		self.num_hidden = num_hidden
		self.num_rnns = num_rnns

	def initialize_parameters(self,session):
		""" Initialize all parameters """
		raise NotImplementedError("Please Implement this method")

	def train_step(self,session,batch_X,batch_Y,keep_prob):
		""" Update the model to fit batch_X and batch_Y """
		raise NotImplementedError("Please Implement this method")

	def get_loss(self,session,X,Y,keep_prob):
		""" Return the loss for X and Y """
		raise NotImplementedError("Please Implement this method")

	def get_probabilities(self,session,X):
		""" Return for each output token the vector of probabilities.

		The result must have this shape: (output_seq_length,voc_size).
		probabilities[i,j] is the probability of the jth token (from the
		vocabulary) being the ith token of the answer.
		"""
		raise NotImplementedError("Please Implement this method")

class BasicSeq2Seq(Seq2Seq):
	"""Define a TensorFlow Graph based on the seq2seq framework."""
	def __init__(self,
				 voc_size,
				 embedding_size,
				 batch_size,
				 num_hidden,
				 num_rnns,
				 input_seq_length,
				 output_seq_length,
				 name="seq2seq_model"):
		super(BasicSeq2Seq, self).__init__(
			voc_size=voc_size,
			embedding_size=embedding_size,
			batch_size=batch_size,
			num_hidden=num_hidden,
			num_rnns=num_rnns)
		self.input_seq_length = input_seq_length
		self.output_seq_length = output_seq_length
		self.name = name
		self.suffix = str(self.batch_size) + '_' + str(self.input_seq_length) + '_' + str(self.output_seq_length)

		with tf.variable_scope(name):
			with tf.variable_scope('learning_rate'):
				self.learning_rate = tf.placeholder(tf.float32)

			with tf.variable_scope('keep_prob'):
				self.keep_prob = tf.placeholder(tf.float32)

			with tf.variable_scope('encoder_inputs_' + self.suffix):
				self.X_ = tf.placeholder(
					tf.int32,
					shape=(self.batch_size,self.input_seq_length))
				# Split to get a list of 'input_seq_length' tensors of shape (batch_size)
				self.encoder_inputs = tf.split(1, self.input_seq_length, self.X_)
				self.encoder_inputs = [tf.reshape(encoder_input, [self.batch_size])
					for encoder_input in self.encoder_inputs]

			with tf.variable_scope('targets_' + self.suffix):
				self.Y_ = tf.placeholder(
					tf.int32,
					shape=(self.batch_size,self.output_seq_length))
				# Split to get a list of 'output_seq_length' tensors of shape (batch_size)
				self.targets = tf.split(1, self.output_seq_length, self.Y_)
				self.targets = [tf.reshape(target, [self.batch_size])
					for target in self.targets]

			with tf.variable_scope('decoder_inputs_' + self.suffix):
				# Decoder input: prepend some "GO" token and drop the final
				# token of the target input
				self.decoder_inputs = [tf.zeros_like(
					self.encoder_inputs[0],
					dtype=tf.int32,
					name="GO")]
				self.decoder_inputs += self.targets[:-1]

			# *** SEQ2SEQ ***
			with tf.variable_scope('seq2seq') as scope:
				lstm_cell = tf.nn.rnn_cell.LSTMCell(self.num_hidden,state_is_tuple=True)
				lstm_cell_with_dropout = tf.nn.rnn_cell.DropoutWrapper(
					cell=lstm_cell,
					output_keep_prob=self.keep_prob)
				self.rnn_cell = tf.nn.rnn_cell.MultiRNNCell(
					cells=[lstm_cell_with_dropout] * self.num_rnns,
					state_is_tuple=True)
				# This op is used for training (feed_previous=False)
				self.decoder_outputs, self.states = tf.nn.seq2seq.embedding_rnn_seq2seq(
					encoder_inputs=self.encoder_inputs,
					decoder_inputs=self.decoder_inputs,
					cell=self.rnn_cell,
					num_encoder_symbols=self.voc_size,
					num_decoder_symbols=self.voc_size,
					embedding_size=self.embedding_size,
					output_projection=None,
					feed_previous=False,
					dtype=None,
					scope=None)
				self.seq2seq_out = self.decoder_outputs[-1]
				scope.reuse_variables()  # To avoid defining other embeddings
				# This op is used for testing (feed_previous=True)
				self.outputs_test, self.states_test = tf.nn.seq2seq.embedding_rnn_seq2seq(
					encoder_inputs=self.encoder_inputs,
					decoder_inputs=self.decoder_inputs,
					cell=self.rnn_cell,
					num_encoder_symbols=self.voc_size,
					num_decoder_symbols=self.voc_size,
					embedding_size=self.embedding_size,
					output_projection=None,
					feed_previous=True,
					dtype=None,
					scope=None)

				self.probabilities = [tf.nn.softmax(output_test)
					for output_test in self.outputs_test]

			# *** LOSS ***
			with tf.variable_scope('Loss'):
				self.loss_weights = [tf.ones_like(t, dtype=tf.float32)
					for t in self.targets]
				self.loss = tf.nn.seq2seq.sequence_loss(
					self.decoder_outputs,
					self.targets,
					self.loss_weights)

			# *** TRAIN STEP ***
			with tf.variable_scope('Train_step'):
				self.optimize = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

			# *** INITIALIZATION ***
			self.init = tf.global_variables_initializer()

	def initialize_parameters(self,session):
		session.run(self.init)

	def train_step(self,session,batch_X,batch_Y,learning_rate,keep_prob):
		return session.run(self.optimize, feed_dict={
			self.X_: batch_X,
			self.Y_: batch_Y,
			self.learning_rate: learning_rate,
			self.keep_prob: keep_prob})

	def get_loss(self,session,X,Y,keep_prob):
		return session.run(self.loss, feed_dict={
			self.X_: X,
			self.Y_: Y,
			self.keep_prob: keep_prob})

	def get_train_loss(self,session,X,Y,keep_prob):
		return self.get_loss(session, X, Y, keep_prob=keep_prob)

	def get_test_loss(self,session,X,Y):
		return self.get_loss(session, X, Y, keep_prob=1.0)

	def train_and_get_loss(self,session,batch_X,batch_Y,keep_prob):
		return session.run([self.train_step,self.loss], feed_dict={
			self.X_: batch_X,
			self.Y_: batch_Y,
			self.keep_prob: keep_prob})

	def get_probabilities(self,session,X):
		# Input must have shape (batch_size,input_seq_length)
		X = np.reshape(X,newshape=(self.batch_size,self.input_seq_length))
		probabilities = session.run(self.probabilities, feed_dict={
			self.X_: X,
			self.Y_: np.zeros(shape=(self.batch_size,self.output_seq_length)),
			self.keep_prob: 1.0})
		for i in range(len(probabilities)):
			# Remove unwanted first dimension
			# Shape (1,voc_size) -> (voc_size,)
			np_array = probabilities[i]
			probabilities[i] = np_array.reshape(np_array.shape[1:])
		probabilities = np.array(probabilities)
		return probabilities
