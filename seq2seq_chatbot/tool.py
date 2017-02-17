""" Simple user interface using the command line.

The tool can execute several tasks such as:
* preprocess: extract data from the Cornell Movie Dialogs Corpus and save it in
a 'dataset' folder (a subfolder of /datasets containing the transformed data
and the vocabulary).
"""
from __future__ import print_function
import argparse
import os
import sys
# Add current folder to PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), os.pardir)))
from seq2seq_chatbot import constants

# === MAIN PARSER ===
parser = argparse.ArgumentParser(description="Tool for seq2seq-chatbot.")

# === SUBPARSERS ===
subparsers = parser.add_subparsers(dest="command",help=None)

# ***********************
# *** Preprocess Task ***
# ***********************
def preprocess(args):
	from seq2seq_chatbot.data.reading.conversation_reader import ConversationReader

	conversation_reader = ConversationReader(
		args.movie_lines_path,
		args.movie_conversations_path)
	dataset = conversation_reader.construct_dataset(voc_size=args.voc_size,
		buckets=constants.BUCKETS,
		dataset_folder=args.dataset_folder,
		verbose=False,
		max=args.max,
		display_step=100)
	dataset.print_data()

preprocess_parser = subparsers.add_parser("preprocess",
	help="Extract data from the Cornell Movie Dialogs Corpus.")
preprocess_parser.add_argument(
	"--movie_lines_path",
	default=constants.MOVIE_LINES_PATH,
	type=str,
	required=False,
	help="")
preprocess_parser.add_argument(
	"--movie_conversations_path",
	default=constants.MOVIE_CONVERSATIONS_PATH,
	type=str,
	required=False,
	help="")
preprocess_parser.add_argument(
	"--dataset_folder",
	default=constants.DEFAULT_DATASET,
	type=str,
	required=False,
	help="")
preprocess_parser.add_argument(
	"--voc_size",
	default=constants.DEFAULT_VOC_SIZE,
	type=int,
	required=False,
	help="")
preprocess_parser.add_argument(
	"--max",
	default=-1,
	type=int,
	required=False,
	help="")

# ******************
# *** Train Task ***
# ******************
def train(args):
	from seq2seq_chatbot.chatbot import Chatbot
	from seq2seq_chatbot.data.classes.dataset import Dataset

	dataset = Dataset.load_from_folder(args.dataset_folder)
	vocabulary_path = Dataset.get_voc_path(args.dataset_folder)
	chatbot = Chatbot(
		buckets=constants.BUCKETS,
		vocabulary_path=vocabulary_path,
		embedding_size=args.embedding_size,
		num_hidden=args.num_hidden,
		num_rnns=args.num_rnns,
		config_folder=args.config_folder,
		restore_config=args.restore)
	chatbot.train(
		dataset=dataset,
		num_epochs=args.num_epochs,
		batch_size=args.batch_size,
		eval_step=args.eval_step,
		keep_prob=args.keep_prob,
		learning_rate=args.learning_rate,
		config_folder=args.config_folder,
		test_proportion=args.test_proportion,
		restore_parameters=args.restore)

train_parser = subparsers.add_parser("train",
	help="Train a Chatbot.")
train_parser.add_argument(
	"--dataset_folder",
	default=constants.DEFAULT_DATASET,
	type=str,
	required=False,
	help="")
train_parser.add_argument(
	"--embedding_size",
	default=constants.EMBEDDING_SIZE,
	type=int,
	required=False,
	help="")
train_parser.add_argument(
	"--num_hidden",
	default=constants.NUM_HIDDEN,
	type=int,
	required=False,
	help="")
train_parser.add_argument(
	"--num_rnns",
	default=constants.NUM_RNNS,
	type=int,
	required=False,
	help="")
train_parser.add_argument(
	"--config_folder",
	default=constants.DEFAULT_CONFIG_FOLDER,
	type=str,
	required=False,
	help="")
train_parser.add_argument(
	"--num_epochs",
	default=constants.NUM_EPOCHS,
	type=int,
	required=False,
	help="")
train_parser.add_argument(
	"--batch_size",
	default=constants.BATCH_SIZE,
	type=int,
	required=False,
	help="")
train_parser.add_argument(
	"--eval_step",
	default=constants.EVAL_STEP,
	type=int,
	required=False,
	help="")
train_parser.add_argument(
	"--keep_prob",
	default=constants.KEEP_PROB,
	type=float,
	required=False,
	help="")
train_parser.add_argument(
	"--learning_rate",
	default=constants.LEARNING_RATE,
	type=float,
	required=False,
	help="")
train_parser.add_argument(
	"--test_proportion",
	default=constants.TEST_PROPORTION,
	type=float,
	required=False,
	help="")
train_parser.add_argument(
	"--restore",
	action='store_true',
	default=False,
	required=False,
	help="")

# *****************
# *** Chat Task ***
# *****************
def chat(args):
	import tensorflow as tf
	import os.path

	from seq2seq_chatbot.chatbot import Chatbot

	if os.path.isfile(Chatbot.get_config_path(args.config_folder)):
		chatbot = Chatbot(config_folder=args.config_folder,restore_config=True)
	else:
		print("Configuration file not found in",args.config_folder)
		print("Define a Chatbot with default values")
		chatbot = Chatbot(
			config_folder=constants.DEFAULT_CONFIG_FOLDER,
			buckets=constants.BUCKETS,
			vocabulary_path=constants.DEFAULT_VOC_PATH,
			embedding_size=constants.EMBEDDING_SIZE,
			num_hidden=constants.NUM_HIDDEN,
			num_rnns=constants.NUM_RNNS)
	with tf.Session() as session:
		# Restore parameters if necessary
		chatbot.restore_parameters_from_config_folder(session, args.config_folder)
		# Start conversation
		print("\n*** CONVERSATION ***")
		print("Enter <STOP> to stop the conversation\n")
		while(True):
			sentence = raw_input("ME: ")
			if sentence == "<STOP>":
				break
			else:
				answer = chatbot.answer(
					session,
					sentence,
					temperature=args.temperature)
				print("BOT:",answer)

chat_parser = subparsers.add_parser("chat",
	help="Chat with the user.")
chat_parser.add_argument(
	"--config_folder",
	default=constants.DEFAULT_CONFIG_FOLDER,
	type=str,
	required=False,
	help="")
chat_parser.add_argument(
	"--temperature",
	default=1.0,
	type=float,
	required=False,
	help="")

# === PARSE ARGUMENTS ===
args = parser.parse_args()

# === DISPLAY ARGUMENTS ===
args_dict = vars(args)
print("============= ARGUMENTS =============")
for arg in args_dict.keys():
	print(arg,"-->",str(args_dict[arg]))
print()

# === EXECUTE COMMAND ===
command = args.command
if command == "preprocess":
	preprocess(args)
elif command == "train":
	train(args)
elif command == "chat":
	chat(args)
else:
	print("Command",command,"not recognized")
