import pickle 
import numpy as np
import os

# filePath = 'SEN12MS\\label_split_dir\\train_list.txt'
# #Simply read a pickle file
# file = open(filePath,'rb')
# inf = pickle.load (file, encoding = 'iso-8859-1') # reads the contents of the file pkl
# print(inf)

filelist_path = 'SEN12MS\\label_split_dir\\train_list.txt'
f = open(filelist_path, "r")
filelist = f.read()
filelist_string = str(filelist)
filelist_array = filelist_string.split('\n')
#print(filelist_array)

filename = 'SEN12MS\\label_split_dir\\train_list.pkl'
with open(filename, 'wb') as filehandler:
    pickle.dump(filelist_array, filehandler)


    

