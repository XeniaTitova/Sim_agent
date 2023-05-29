import tensorflow as tf
from numpy.lib import format
import numpy as np
import os
from os.path import isfile, join
from tqdm import tqdm

from prerender_utils.prerender_utils import create_dataset, save_map_id, save_original, save_by_id
from prerender_utils.features_description import generate_features_description

path =os.getcwd()
dataset, len_dataset = create_dataset(path + "/input_tf")

# Créez une barre de progression avec le nombre total d'éléments
progress_bar = tqdm(total=len_dataset, ncols=80, bar_format='\033[92m{l_bar}{bar}| {n_fmt}/{total_fmt}')
i = 0
for data in dataset.as_numpy_iterator():
    data = tf.io.parse_single_example(data, generate_features_description())
    i+=1
    #save_map_id(data)
    save_by_id(data)
    progress_bar.update(1)
progress_bar.close()
