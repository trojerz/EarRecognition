import sys
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
from gender_model import train
from ethnicity_model import train_eth
from aux_functions import load_data, get_dict_eth, get_dict, CMatrix
from evaluate_model import evaluate_model

val_size = 0.15

if str(sys.argv[1]) == 'gender':
    get_eth = False
    fl = 'awe-translation_gender.csv'
if str(sys.argv[1]) == 'ethnicity':
    get_eth = True
    fl = 'awe-translation.csv'

id_dict, n_classes = get_dict(fl, get_eth)

# load train data
data_set_train, classes_train = load_data('train', id_dict=id_dict)
classes_train_old = classes_train
classes_train = classes_train.astype(np.int)
classes_train = np_utils.to_categorical(classes_train, n_classes)
data_set_train, valid_data, classes_train, classes_valid = train_test_split(data_set_train, classes_train, test_size = val_size)

# load test data
data_set_test, classes_test = load_data('test', id_dict=id_dict)
classes_test_old = classes_test
classes_test = classes_test.astype(np.int)
classes_test = np_utils.to_categorical(classes_test, n_classes)
data = [(np.asarray(data_set_train), classes_train),(np.asarray(valid_data), classes_valid), (np.asarray(data_set_test), classes_test)]

# model
if get_eth:
    model = train_eth(data, batch_size = 15, epochs = 7, pool_size =4, kernel_size =2, n_classes= n_classes)
else:
    model = train(data, batch_size = 10 ,epochs =6, pool_size = 4, kernel_size =2, n_classes= n_classes)

# evaluate model
evaluate_model(input_model=model, data_set_test=data_set_test, classes_test_old=classes_test_old, get_eth=get_eth)

model.save('latest_model.h5', overwrite = True)