import torch
import numpy as np

##TENSORS

#from array
data = [[1,2], [3,4], [5,6]]
arr_tensor = torch.tensor(data)

#numpy
np_array = np.array(data)
np_tensor = torch.from_numpy(np_array)

#creates tensor of ones with same shape as data
ones_arr = torch.ones_like(data)
#creates tensor of rand nums with same shape as data
rand_arr = torch.rand_like(data)

print(f"ones array: {ones_arr}")
print(f"rand array: {rand_arr}")