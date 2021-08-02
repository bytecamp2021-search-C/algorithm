import numpy as np

a = np.array([[]])
a = np.concatenate((a,[[5,8]]),axis=1)
b =np.concatenate(([[1,3]],[[5,6]]),axis=0)
print(a)
print(b)
