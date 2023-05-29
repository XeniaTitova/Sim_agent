from numpy.lib import format
from numpy import load
from os import listdir
import os

path =os.getcwd()+"/out/"#scid_1dde730d8ab3ae4a__aid_1__atype_1.npz"
list = listdir(path)
data = load(path+list[0])
lst = data.files
id_list = []
for d in data.keys():
    # Vérifier si la clé commence par 'state/' suivi d'un nombre
    if d.startswith('state/'):
        # Ajouter la clé à la liste des catégories correspondantes
        segments = d.split('/')
        id_list.append(segments[1])