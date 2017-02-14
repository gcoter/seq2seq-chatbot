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
	default=-1,type=int,
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
