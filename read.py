from numpy.lib import format
from numpy import load
from os import listdir
import os

path =os.getcwd()+"/out/"#scid_1dde730d8ab3ae4a__aid_1__atype_1.npz"
list = listdir(path)
for l in list :
    print('\n \n '+l+'\n')
    data = load(path+l)
    # lst = data.files
    # for item in lst:
    #     print(item)
    #     print(data[item])
    # break #a enlever si on veux afficher tout
    #
    print(data["my/past/xy"])