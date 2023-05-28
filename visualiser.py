import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from tqdm import tqdm

import visualiser_utils.visualiser_utils as vu
import numpy as np
from numpy import load
from os import listdir
import os

#trouver tout les fichier du prerender:
path = os.getcwd() + "/out/"  # scid_1dde730d8ab3ae4a__aid_1__atype_1.npz"
list = listdir(path)

progress_bar = tqdm(total=len(list), ncols=80, bar_format='\033[92m{l_bar}{bar}| {n_fmt}/{total_fmt}', position=0)

for l in list:
    # telecharger les donné de la map
    data = load(path + l)
    lst = data.files
    xy = data["map/xy"]
    dir = data["map/dir"]
    type = data["map/type"]

    # calcule les bords du graphe
    xy_filt = xy[xy[:, 0] != -1]
    x = xy[:, 0]
    y = xy[:, 1]
    x_min = np.min(xy_filt[:, 0])
    x_max = np.max(xy_filt[:, 0])
    y_min = np.min(xy_filt[:, 1])
    y_max = np.max(xy_filt[:, 1])

    # créé le graphique
    fig, ax = plt.subplots()
    plt.xlim(x_min - 10, x_max + 10)
    plt.ylim(y_min - 10, y_max + 10)
    plt.axis('off')
    plt.title(data["senario_ID"])
    ax.set_aspect('equal')

    # dessiner les routes:
    try:
        progress_bar_subprocess1.close()
    except:
        pass
    progress_bar_subprocess1 = tqdm(total=len(xy), ncols=80, bar_format='\033[36m{l_bar}{bar}| {n_fmt}/{total_fmt}', leave=False ,position=1)
    for xy1, dir1, type1 in zip(xy, dir, type):
        if xy1[0] != -1:
            if type1 == 1:
                vu.drawSegment(ax, xy1, dir1, color='green', linewidth=0.1, linestyle='dashed')
            elif type1 == 2:
                vu.drawSegment(ax, xy1, dir1, color='lime', linewidth=0.1, linestyle='dashed')
            elif type1 == 3:
                vu.drawSegment(ax, xy1, dir1, color='turquoise', linewidth=0.07, linestyle='dashed')
            elif type1 == 6 or type1 == 9 or type1 == 10:
                vu.drawSegment(ax, xy1, dir1, color='royalblue', linewidth=0.15, linestyle='dashed')
            elif type1 == 7 or type1 == 8 or type1 == 11 or type1 == 12:
                vu.drawSegment(ax, xy1, dir1, color='royalblue', linewidth=0.2, linestyle='dashed')
            elif type1 == 15:
                vu.drawSegment(ax, xy1, dir1, color='black', linewidth=0.3, linestyle='dashed')
            elif type1 == 16:
                vu.drawSegment(ax, xy1, dir1, color='blue', linewidth=0.3, linestyle='dashed')
            elif type1 == 18:
                vu.drawSegment(ax, xy1, dir1, color='yellow', linewidth=0.3, linestyle='dashed')
            elif type1 == 17 or type1 == 19:
                vu.drawSegment(ax, xy1, dir1, color='red', linewidth=0.3, linestyle='dashed')
            # print(type1)
            # print(xy1)
            # print(dir1)
            # print("\n")
        progress_bar_subprocess1.update(1)


    plt.savefig(f"out_visual/{data['senario_ID']}.png")
    plt.close()
    progress_bar.update(1)
try:
    progress_bar_subprocess1.close()
except:
    pass
progress_bar.close()