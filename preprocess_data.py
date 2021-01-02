import pathlib
from get_coordinates import get_coordinates
from PIL import Image
import pandas as pd
import csv

id_dict = {}
id_list = []
# identify persons on image - set id for each of them.
with open('awe-translation.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            line_count += 1
        else:
            id_dict[row[0][:-4]] = row[2]
            id_list.append(row[2])

#prepare test pictures and annotations
pic_list = [p for p in pathlib.Path('AWEForSegmentation/testannot_rect').iterdir() if p.is_file()]
dataset = dict()
dataset['img_name'] = list()
dataset['x_min'] = list()
dataset['y_min'] = list()
dataset['x_max'] = list()
dataset['y_max'] = list()
dataset['class_name'] = list()
counter = 1

for name in pic_list:
    prefix = 'test/'
    img_name = str(name)[-8:]
    coordinates, object_n = get_coordinates(str(str(name)[-8:-4]), 'testannot_rect')
    img_dir = 'AWEForSegmentation/test/' + str(name)[-8:-4] + '.png'
    img_name_id = prefix + str(name)[-8:-4]
    for j in range(object_n):
        dataset['x_min'].append(0)
        dataset['y_min'].append(0)
        dataset['x_max'].append(200)
        dataset['y_max'].append(200)
        #dataset['img_name'].append(f'test/ears_test_{counter}.jpeg')
        dataset['img_name'].append(f'ears_test_{counter}.jpeg')
        dataset['class_name'].append(id_dict[img_name_id])
        box = (coordinates[j][0], coordinates[j][1], coordinates[j][2], coordinates[j][3] )
    img = Image.open(img_dir).convert('RGB')
    img = img.crop(box)
    img = img.resize((200,200))
    img.save(f'test/ears_test_{counter}.jpeg', 'JPEG')
    counter += 1
df_test = pd.DataFrame(dataset)
df_test.to_csv('df_test.csv', index = False, header = None) 

#prepare train pictures and annotations
pic_list_train = [p for p in pathlib.Path('AWEForSegmentation/trainannot_rect').iterdir() if p.is_file()]
dataset_train = dict()
dataset_train['img_name'] = list()
dataset_train['x_min'] = list()
dataset_train['y_min'] = list()
dataset_train['x_max'] = list()
dataset_train['y_max'] = list()
dataset_train['class_name'] = list()
counter_train = 1

for name in pic_list_train:
    prefix = 'train/'
    img_name = str(name)[-8:]
    coordinates, object_n = get_coordinates(str(str(name)[-8:-4]), 'trainannot_rect')
    img_dir = 'AWEForSegmentation/train/' + str(name)[-8:-4] + '.png'
    img_name_id = prefix + str(name)[-8:-4]
    for j in range(object_n):
        dataset_train['x_min'].append(0)
        dataset_train['y_min'].append(0)
        dataset_train['x_max'].append(200)
        dataset_train['y_max'].append(200)
        dataset_train['img_name'].append(f'ears_train_{counter_train}.jpeg')
        #dataset_train['img_name'].append(f'train/ears_train_{counter_train}.jpeg')
        dataset_train['class_name'].append(id_dict[img_name_id])
        box = (coordinates[j][0], coordinates[j][1], coordinates[j][2], coordinates[j][3] )
    img = Image.open(img_dir).convert('RGB')
    img = img.crop(box)
    img = img.resize((200,200))
    img.save(f'train/ears_train_{counter_train}.jpeg', 'JPEG')
    counter_train += 1
df_train = pd.DataFrame(dataset_train)

ANNOTATIONS_FILE = 'annotations.csv'
CLASSES_FILE = 'classes.csv'

df_train.to_csv(ANNOTATIONS_FILE, index = False, header = None)
df_test.to_csv('annotations_test.csv', index = False, header = None)

classes = set(id_list)

with open(CLASSES_FILE, 'w') as f:
    for i, line in enumerate(sorted(classes)):
        f.write('{}, {}\n'.format(line, line))