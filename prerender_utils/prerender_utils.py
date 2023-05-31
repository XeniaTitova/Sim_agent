import tensorflow as tf
import os
import numpy as np
import yaml
from tqdm import tqdm
from yaml import Loader
import numpy as np


def create_dataset(datapath):
    files = os.listdir(datapath)
    dataset = tf.data.TFRecordDataset(
        [os.path.join(datapath, f) for f in files], num_parallel_reads=1
    )
    len = 0
    for _ in dataset.as_numpy_iterator():
        len += 1
    return dataset, len


def save_map_id(data,
                nb_veicule=5):  # enregistre dans le format avec une map et un fichier par veicule avec les veicule l'entourant
    # créé les donnée de la carte
    id_senario = data["scenario/id"].numpy()[0].decode('utf-8')
    data_true = {}
    data_true["senario_ID"] = id_senario

    data_true["map/dir"] = [[x, y] for x, y, _ in data["roadgraph_samples/dir"].numpy()]
    data_true["map/xy"] = [[x, y] for x, y, _ in data["roadgraph_samples/xyz"].numpy()]
    data_true["map/type"] = data["roadgraph_samples/type"].numpy()
    data_true["map/valid"] = data["roadgraph_samples/valid"].numpy()

    # créé un fichier avec les plus proche voisin pour chaque véicule qui est sur la carte au temps présant
    for valid, id, x, y in zip(data["state/current/valid"].numpy(), data["state/id"].numpy()[:, 0],
                               data["state/current/x"].numpy()[:, 0], data["state/current/y"].numpy()):
        if valid == 1:
            distance, nb_inv_id = distance_calcule(data["state/current/valid"].numpy(),
                                                   data["state/current/x"].numpy(), data["state/current/y"].numpy(),
                                                   x[0], y[0])

            i = np.argsort(distance)[nb_inv_id]

            data_true[f"state/{id}/my/id"] = id
            data_true[f"state/{id}/my/shift"] = [x[0], y[0]]
            data_true[f"state/{id}/my/width"] = data["state/current/width"].numpy()[i]
            data_true[f"state/{id}/my/length"] = data["state/current/length"].numpy()[i]
            data_true[f"state/{id}/my/type"] = data["state/type"].numpy()[i]

            data_true[f"state/{id}/my/current/speed"] = data["state/current/speed"].numpy()[i]
            data_true[f"state/{id}/my/current/ang"] = data["state/current/bbox_yaw"].numpy()[i]

            data_true[f"state/{id}/my/past/xy"] = creat_xy(data["state/past/x"].numpy()[i],
                                                           data["state/past/y"].numpy()[i],
                                                           -x[0], -y[0])
            data_true[f"state/{id}/my/past/speed"] = data["state/past/speed"].numpy()[i]
            data_true[f"state/{id}/my/past/ang"] = data["state/past/bbox_yaw"].numpy()[i]
            data_true[f"state/{id}/my/past/valid"] = data["state/past/valid"].numpy()[i]

            data_true[f"state/{id}/my/future/xy"] = creat_xy(data["state/future/x"].numpy()[i],
                                                             data["state/future/y"].numpy()[i],
                                                             -x[0], -y[0])
            data_true[f"state/{id}/my/future/speed"] = data["state/future/speed"].numpy()[i]
            data_true[f"state/{id}/my/future/ang"] = data["state/future/bbox_yaw"].numpy()[i]
            data_true[f"state/{id}/my/future/valid"] = data["state/future/valid"].numpy()[i]

            indice_triees = np.argsort(distance)[nb_inv_id + 1: nb_inv_id + 1 + nb_veicule]

            # data_true[f"state/{id}/other/current/dist"] = np.array(distance)[indice_triees]
            data_true[f"state/{id}/other/id"] = data["state/id"].numpy()[indice_triees]
            data_true[f"state/{id}/other/width"] = data["state/current/width"].numpy()[indice_triees]
            data_true[f"state/{id}/other/length"] = data["state/current/length"].numpy()[indice_triees]

            data_true[f"state/{id}/other/current/xy"] = creat_xy(data["state/current/x"].numpy()[indice_triees],
                                                                 data["state/current/y"].numpy()[indice_triees],
                                                                 -x[0], -y[0])
            data_true[f"state/{id}/other/current/speed"] = data["state/current/speed"].numpy()[indice_triees]
            data_true[f"state/{id}/other/current/ang"] = data["state/current/bbox_yaw"].numpy()[indice_triees]

            data_true[f"state/{id}/other/past/xy"] = creat_xy(data["state/past/x"].numpy()[indice_triees],
                                                              data["state/past/y"].numpy()[indice_triees],
                                                              -x[0], -y[0])
            data_true[f"state/{id}/other/past/speed"] = data["state/past/speed"].numpy()[indice_triees]
            data_true[f"state/{id}/other/past/ang"] = data["state/past/bbox_yaw"].numpy()[indice_triees]
            data_true[f"state/{id}/other/past/valid"] = data["state/past/valid"].numpy()[indice_triees]

    np.savez_compressed(f"out/{id_senario}.npz", **data_true)  # print(data_true)


def save_by_id(data,
               distance_max=30):  # enregistre dans le format avec une map et un fichier par veicule avec les veicule l'entourant
    # créé les donnée de la carte
    id_senario = data["scenario/id"].numpy()[0].decode('utf-8')

    # créé un fichier avec les plus proche voisin pour chaque véicule qui est sur la carte au temps présant
    for valid, id, x, y in zip(data["state/current/valid"].numpy(), data["state/id"].numpy(),
                               data["state/current/x"].numpy(), data["state/current/y"].numpy()):
        if valid == 1:
            data_true = {}
            data_true["senario_ID"] = id_senario
            dist_other, nb_inv_id = distance_calcule(data["state/current/valid"].numpy(),
                                                     data["state/current/x"].numpy()[:, 0],
                                                     data["state/current/y"].numpy()[:, 0],
                                                     x[0], y[0])

            # les variables qui conserne le veicule à predir
            i = np.argsort(dist_other)[nb_inv_id]

            data_true[f"my/id"] = id
            data_true[f"my/shift"] = [x[0], y[0]]
            data_true[f"my/width"] = data["state/current/width"].numpy()[i]
            data_true[f"my/length"] = data["state/current/length"].numpy()[i]
            data_true[f"my/type"] = data["state/type"].numpy()[i]

            data_true[f"my/current/speed"] = data["state/current/speed"].numpy()[i]
            data_true[f"my/current/ang"] = data["state/current/bbox_yaw"].numpy()[i]

            data_true[f"my/past/xy"] = creat_xy(data["state/past/x"].numpy()[i],
                                                data["state/past/y"].numpy()[i],
                                                -x[0], -y[0])
            data_true[f"my/past/speed"] = data["state/past/speed"].numpy()[i]
            data_true[f"my/past/ang"] = data["state/past/bbox_yaw"].numpy()[i]
            data_true[f"my/past/valid"] = data["state/past/valid"].numpy()[i]

            data_true[f"my/future/xy"] = creat_xy(data["state/future/x"].numpy()[i],
                                                  data["state/future/y"].numpy()[i],
                                                  -x[0], -y[0])
            data_true[f"my/future/speed"] = data["state/future/speed"].numpy()[i]
            data_true[f"my/future/ang"] = data["state/future/bbox_yaw"].numpy()[i]
            data_true[f"my/future/valid"] = data["state/future/valid"].numpy()[i]

            # les variables qui conserne les veicules aualontoure
            dist_other = np.array(dist_other)
            indice_triees = (dist_other < distance_max ** 2) & (dist_other != -1)

            # data_true[f"state/{id}/other/current/dist"] = np.array(distance)[indice_triees]
            data_true[f"other/type"] = data["state/type"].numpy()[indice_triees]
            data_true[f"other/id"] = data["state/id"].numpy()[indice_triees]
            data_true[f"other/width"] = data["state/current/width"].numpy()[indice_triees]
            data_true[f"other/length"] = data["state/current/length"].numpy()[indice_triees]

            data_true[f"other/current/xy"] = creat_xy(data["state/current/x"].numpy()[indice_triees],
                                                      data["state/current/y"].numpy()[indice_triees],
                                                      -x[0], -y[0])
            data_true[f"other/current/speed"] = data["state/current/speed"].numpy()[indice_triees]
            data_true[f"other/current/ang"] = data["state/current/bbox_yaw"].numpy()[indice_triees]

            data_true[f"other/past/xy"] = creat_xy(data["state/past/x"].numpy()[indice_triees],
                                                   data["state/past/y"].numpy()[indice_triees],
                                                   -x[0], -y[0])
            data_true[f"other/past/speed"] = data["state/past/speed"].numpy()[indice_triees]
            data_true[f"other/past/ang"] = data["state/past/bbox_yaw"].numpy()[indice_triees]
            data_true[f"other/past/valid"] = data["state/past/valid"].numpy()[indice_triees]

            # les variables qui conserne la map des alentour

            xyz_map = data["roadgraph_samples/xyz"].numpy()
            dist_map, _ = distance_calcule(data["roadgraph_samples/valid"], xyz_map[:, 0], xyz_map[:, 1], x[0], y[0])

            dist_map = np.array(dist_map)
            indice_triees = (dist_map < distance_max ** 2) & (dist_map != -1)

            data_true["map/dir"] = [[x, y] for x, y, _ in data["roadgraph_samples/dir"].numpy()[indice_triees]]
            data_true["map/xy"] = creat_xy(xyz_map[:, 0][indice_triees],
                                           xyz_map[:, 1][indice_triees],
                                           -x[0], -y[0])[0]  # [[x, y] for x, y, _ in data["roadgraph_samples/xyz"].numpy()[indice_triees]]
            data_true["map/type"] = data["roadgraph_samples/type"].numpy()[indice_triees]
            data_true["map/valid"] = data["roadgraph_samples/valid"].numpy()[indice_triees]

            np.savez_compressed(f"out/{id_senario}_{id}.npz", **data_true)
            #print(data_true)


def distance_calcule(valid, x, y, x0, y0):
    distance = []
    nb_inv_id = 0
    for valid1, x1, y1 in zip(valid, x, y):
        if valid1 == 1:
            distance.append((x1 - x0) ** 2 + (y1 - y0) ** 2)
        else:
            nb_inv_id += 1
            distance.append(-1)
    return distance, nb_inv_id


def save_original(data):
    id = data["scenario/id"].numpy()[0].decode('utf-8')  # met en forme le id
    data["scenario/id"] = id
    np.savez_compressed(f"out/original/{id}.npz", **data)  # sauve tout les donné comme il sont


def creat_xy(x_data, y_data, x_shift, y_shift):
    xy = np.dstack((x_data + x_shift, y_data + y_shift))
    return xy
