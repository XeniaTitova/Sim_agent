from draw import drawRectangle, drawSegment
import numpy as np
import matplotlib.pyplot as plt


def make_fig(xy, titel=''):
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
    plt.title(titel)
    ax.set_aspect('equal')

    return fig, ax


def findID(data):
    id_list = []
    for d in data.keys():
        # Vérifier si la clé commence par 'state/' suivi d'un nombre
        if d.startswith('state/'):
            # Ajouter la clé à la liste des catégories correspondantes
            segments = d.split('/')
            id_list.append(segments[1])
    return set(id_list)


def plot_veicul(ax, data):
    ids = findID(data)
    for id in ids:
        xy = data[f"state/{id}/my/shift"]
        w = data[f"state/{id}/my/width"]
        l = data[f"state/{id}/my/length"]
        ang = data[f"state/{id}/my/current/ang"] * 180 / 3.14
        type1 = data[f"state/{id}/my/type"]
        plt.plot(xy, color='pink')
        if type1 == 1:
            drawRectangle(ax, xy, ang, l, w, color='red')
        if type1 == 2:
            drawRectangle(ax, xy, ang, l, w, color='lightcoral')
        if type1 == 2:
            drawRectangle(ax, xy, ang, l, w, color='violet')


def plot_map(ax, data):
    # recuperation des donné
    xy_map = data["map/xy"]
    dir_map = data["map/dir"]
    type_map = data["map/type"]

    # dessin de chaque bout de route en fonction de leur type
    for xy1, dir1, type1 in zip(xy_map, dir_map, type_map):
        if xy1[0] != -1:
            if type1 == 1:
                drawSegment(ax, xy1, dir1, color='green', linewidth=0.1, linestyle='dashed')
            elif type1 == 2:
                drawSegment(ax, xy1, dir1, color='lime', linewidth=0.1, linestyle='dashed')
            elif type1 == 3:
                drawSegment(ax, xy1, dir1, color='turquoise', linewidth=0.07, linestyle='dashed')
            elif type1 == 6 or type1 == 9 or type1 == 10:
                drawSegment(ax, xy1, dir1, color='royalblue', linewidth=0.15, linestyle='dashed')
            elif type1 == 7 or type1 == 8 or type1 == 11 or type1 == 12:
                drawSegment(ax, xy1, dir1, color='royalblue', linewidth=0.2, linestyle='dashed')
            elif type1 == 15:
                drawSegment(ax, xy1, dir1, color='black', linewidth=0.3, linestyle='dashed')
            elif type1 == 16:
                drawSegment(ax, xy1, dir1, color='blue', linewidth=0.3, linestyle='dashed')
            elif type1 == 18:
                drawSegment(ax, xy1, dir1, color='yellow', linewidth=0.3, linestyle='dashed')
            elif type1 == 17 or type1 == 19:
                drawSegment(ax, xy1, dir1, color='red', linewidth=0.3, linestyle='dashed')
