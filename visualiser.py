import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from tqdm import tqdm

import visualiser_utils.visualiser_utils as vu
import numpy as np
from numpy import load
from os import listdir
import os

# trouver tout les fichier du prerender:
path = os.getcwd() + "/out/"  # scid_1dde730d8ab3ae4a__aid_1__atype_1.npz"
list = listdir(path)

progress_bar = tqdm(total=len(list), ncols=80, bar_format='\033[92m{l_bar}{bar}| {n_fmt}/{total_fmt}', position=0)

for l in list:
    # telecharger les donné de la map
    data = load(path + l)
    fig, ax = vu.make_fig(data["map/xy"], data["senario_ID"])

    vu.plot_map(ax, data)  # dessiner les routes:
    vu.plot_veicul(ax, data)    #dessiner les veicule

    plt.savefig(f"out_visual/{data['senario_ID']}.jpg")
    plt.close()
    progress_bar.update(1)

progress_bar.close()
