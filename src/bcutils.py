import pickle
import numpy as np

from ann_benchmarks.datasets import get_dataset
from ann_benchmarks.distance import dataset_transform
def load_input_data(dataset="glove-25-angular"):
	D, dimension = get_dataset(dataset)
	tags = np.array(D['tags'])
	distance = D.attrs['distance']
	neighbors = np.array(D['neighbors'])
	items, users = dataset_transform(D)
	print("Load data complete!")
	return items,users,tags,neighbors

def save_object(obj,filename):
	f = open(filename,'wb+')
	pickle.dump(obj,f)
	f.close()

def load_object(filename):
	f = open(filename,'r')
	obj = pickle.load(f)
	f.close()
	return obj