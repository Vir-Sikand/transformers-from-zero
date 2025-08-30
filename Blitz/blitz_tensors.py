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
ones_arr = torch.ones_like(np_tensor)
#creates tensor of rand nums with same shape as data
rand_arr = torch.rand_like(arr_tensor, dtype=torch.float)#can't be a long

print(f"ones array: {ones_arr}")
print(f"rand array: {rand_arr}")

#shape and size stuff

shape = (2,3)

zeros = torch.zeros(shape)
print(f"Zeros from shape: {zeros}")

#Tensor attributes:

tensor = torch.rand(3, 4)
print(f"shape of tensor: {tensor.shape}")
print(f"type of tensor: {torch.dtype}")
print(f"device stored on: {torch.device}")

#tensor ops:

if torch.cuda.is_available() :
    tensor = tensor.to('cuda')
    print(f"device is stored on {tensor.device}")

tensor = torch.zeros(3,3)
tensor[:, 2] = 1 #should make everything in third column all ones
print(tensor)
print()
#concatenating tensors
tensor2 = torch.cat([tensor, tensor], dim=1) #along columns
print(f"concatenated tensor: {tensor2}")
print()

mul1 = torch.rand(2,2, dtype=torch.float)
mul_tensor = mul1.mul(mul1) #element wise product
print(f"multiplied tensor: {mul_tensor}")
print()

