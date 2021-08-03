import h5py
import  numpy as np
f = h5py.File('data/glove-25-angular.hdf5','r')
new_f = h5py.File('data/rew_data','w')
print(f.attrs)
for file in f.file:
    # print(file,type(f[file]))
    data = np.array(f[file])
    print(file,data.shape)


def testdata(name):
    f = h5py.File(name)

if __name__ =='main':
    testdata('glove-25-angular.hdf5')
