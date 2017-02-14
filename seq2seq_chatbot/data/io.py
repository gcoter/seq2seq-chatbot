""" Define saving and loading functions.
"""
import numpy as np
import json
import os
import errno

def create_folder(folder_path):
    try:
        os.makedirs(folder_path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise
    return folder_path

def save_numpy_arrays(numpy_arrays,npz_path):
    np.savez_compressed(npz_path,**numpy_arrays)

def load_numpy_arrays(npz_path):
    return np.load(npz_path)

def save_object_as_json(obj,json_path):
    with open(json_path, 'w') as json_file:
        json.dump(obj, json_file)

def load_object_from_json(json_path):
    with open(json_path, 'r') as json_file:
        return json.load(json_file)
    return None
