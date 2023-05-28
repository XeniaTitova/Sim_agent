from numpy.lib import format
from numpy import load
from os import listdir
import os

path =os.getcwd()+"/output_prerender/"#scid_1dde730d8ab3ae4a__aid_1__atype_1.npz"
list = listdir(path)
for l in  list :
    print('\n \n '+l+'\n')
    data = load(path+l)
    lst = data.files
    for item in lst:
        print(item)
        print(data[item])
        # if item == 'scenario_id':
        #     print(f'scenario_id :{data[item]}')
        # elif item == 'agent_id':
        #     print(f'agent_id :{data[item]}')
        #print(item)
        #print(data[item])
