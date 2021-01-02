import csv
import cv2
import numpy as np
import pandas as pd
from skimage.io import imread_collection

def load_data(path, id_dict):
    """
    Loads images in the right format
    """
    image_name = path + '/'
    path = path + '/*.jpeg'
    image = imread_collection(path)
    i = 1
    image_set = []
    if image_name == 'train/':
        classes = np.empty((750))
    else:
        classes = np.empty((250))
    for n in image:
        classes[i-1:i] = int(id_dict[image_name + str(i).zfill(4)])
        n = cv2.cvtColor(n,cv2.COLOR_RGB2GRAY)
        image_set.append(n)
        i+= 1
    return (image_set, classes)

def get_dict_eth(eth_file):
    """
    Creates dict for ethnicity.
    """
    eth = {}
    with open(eth_file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=';')
        for row in csv_reader:
            eth[int(row[0])] = int(row[1])
    return eth

def get_dict(translation_file, get_eth):
    """
    Creates dict for target variable. Need to specify transaltion file and if target is ethnicity.
    """
    id_dict = {}
    id_list = []
    with open(translation_file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        eth = get_dict_eth('eth.csv')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                line_count += 1
            else:
                if row[2] == 'M':
                    id_dict[row[0][:-4]] = int(0)
                    id_list.append(0)
                elif row[2] == 'F':
                    id_dict[row[0][:-4]] = int(1)
                    id_list.append(1)
                else:
                    if get_eth:
                        id_dict[row[0][:-4]] = int(eth[int(row[2])]) - 1
                        id_list.append(int(eth[int(row[2])]) - 1)
                    else:
                        id_dict[row[0][:-4]] = int(row[2])-1
                        id_list.append(int(row[2]))
    n_classes = len(set(id_list))
    return (id_dict, n_classes)

def CMatrix(CM, get_eth):
    """
    Confusion matrix
    """
    if get_eth:
        labels=['1','2','3','4','5','6','7']
    else:
        labels= ['Male', 'Female']
    df = pd.DataFrame(data=CM, index=labels, columns=labels)
    df.index.name='TRUE'
    df.columns.name='PREDICTION'
    df.loc['Total'] = df.sum()
    df['Total'] = df.sum(axis=1)
    return df