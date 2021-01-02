import sys
import numpy as np
from keras.models import load_model
from aux_functions import load_data, get_dict
from evaluate_model import evaluate_model
import matplotlib.image as img
import matplotlib.pyplot as plt 

from PIL import Image

model_path = 'ethnicity_models/latest_model_nadam.h5'
model = load_model(model_path)

if str(sys.argv[1]) == 'gender':
    get_eth = False
    fl = 'awe-translation_gender.csv'
if str(sys.argv[1]) == 'ethnicity':
    get_eth = True
    fl = 'awe-translation.csv'

id_dict, n_classes = get_dict(fl, get_eth)

data_set_test, classes_test = load_data('test', id_dict=id_dict)

i = int(sys.argv[2]) 
image_ord = [data_set_test[i- 1]]
actual = int(classes_test[i- 1])
image = image_ord
image = np.asarray(image)
image = image[:,:,:,np.newaxis] 
y_out = model.predict(image)
position = int(y_out.argmax(axis=1))
prob = round(np.max(y_out) * 100, 1)

print("Predicted: class %s with probability %s, actual: class %s." % (str(position), str(prob), str(actual)))

evaluate_model(input_model=model, data_set_test=data_set_test, classes_test_old=classes_test, get_eth=get_eth)

img_name = 'test_predict/' + str(i).zfill(4) + '.png'
img = Image.open(img_name)
img.show()