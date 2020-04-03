import os 
import json
import pandas as pd
from pathlib import Path

dataset_list = ['public', 'private']
def test():
  datasets = [{'label_name':[1,2,3,4,5],
                'image_names':[],
                'image_labels':[]} for name in dataset_list]
  dirdata15 = r"/resized_test_15/"
  cwd = os.path.abspath('')
  dirname = Path(cwd + dirdata15)
  for index, row in train_file.iterrows():
    if row['Usage'] == 'Public':
      file = str(Path(dirname, '{}.jpg'.format(row['image'])))
      datasets[0]['image_names'].append(file)
      datasets[0]['image_labels'].append(row['level'])
#       print(file, row['level'], row['Usage'])
    elif row['Usage'] == 'Private':
      file = str(Path(dirname, '{}.jpg'.format(row['image'])))
      datasets[1]['image_names'].append(file)
      datasets[1]['image_labels'].append(row['level'])
  for name in dataset_list:
      with open(name + '15' + '.json', 'w') as f:
          json.dump(datasets[dataset_list.index(name)], f, indent=4)


test()