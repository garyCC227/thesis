import os 
import json
import pandas as pd

dirname = "/srv/scratch/z5163479/aptos/labels/"

data15 = dirname + "trainLabels15.csv"
data19 = dirname + "trainLabels19.csv"

dirdata15 = "/srv/scratch/z5163479/aptos/labels/resized_train_15/"
dirdata19 = "/srv/scratch/z5163479/aptos/labels/resized_train_19/"

dataset_list = ['base', 'val', 'novel']

def create(dirname, filename, dirdata, image_title, label_title, output):
    datasets = [{'label_name':[1,2,3,4,5],
                'image_names':[],
                'image_labels':[]} for name in dataset_list]
    train_file = pd.read_csv(filename)
    files = os.listdir(dirdata)
    for index, row in train_file.iterrows():
        file = dirdata + row[image_title] + ".jpg"
        if index%5 in [0, 3, 4]:
            datasets[0]['image_names'].append(file)
            datasets[0]['image_labels'].append(row[label_title])
        if index%5 == 1:
            datasets[1]['image_names'].append(file)
            datasets[1]['image_labels'].append(row[label_title])
        if index%5 == 2:
            datasets[2]['image_names'].append(file)
            datasets[2]['image_labels'].append(row[label_title])
    for name in dataset_list:
        with open(dirname + name + output + '.json', 'w') as f:
            json.dump(datasets[dataset_list.index(name)], f, indent=4)

create(dirname, data15, dirdata15, 'image', 'level', '15')
create(dirname, data19, dirdata19, 'id_code', 'diagnosis', '19')
        