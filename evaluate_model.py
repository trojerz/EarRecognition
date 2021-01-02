import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from aux_functions import CMatrix

def evaluate_model(input_model, data_set_test, classes_test_old, get_eth):
    """
    It evaluates the model, depends on the target variable
    """
    if get_eth:
        CM = [[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0]]
        for i in range(250):
            image_ord = [data_set_test[i]]
            act = classes_test_old[i]
            image = image_ord
            image = np.asarray(image)
            image = image[:,:,:,np.newaxis] 
            y_out = input_model.predict(image)
            y_out = y_out.argmax(axis=1)
            act = int(act)
            y_out = int(y_out)
            CM[act][y_out] += 1
        print(CMatrix(CM, get_eth))
    else:
        CM = [[0,0],[0,0]]
        for i in range(250):
            image_ord = [data_set_test[i]]
            act = classes_test_old[i]
            image = image_ord
            image = np.asarray(image)
            image = image[:,:,:,np.newaxis] 
            y_out = input_model.predict(image)
            y_out = y_out.argmax(axis=1)
            act = int(act)
            y_out = int(y_out)
            CM[act][y_out] += 1
        print(CMatrix(CM, get_eth))