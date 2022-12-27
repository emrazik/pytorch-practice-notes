# *** Notes on PyTorch tutorial here: pytorch.org/tutorials/beginner/blitz/tensor_tutorial.html#tensor-attributes ***

import torch
import numpy as np

# initialize tensor
data = [[1,2], [3,4]]
x_data = torch.tensor(data)

# initialize tensor with numpy
np_array = np.array(data)
x_np = torch.from_numpy(np_array)

# shape is a tuple of tensor dimensions
shape = (2,3,)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print(f"Random Tensor: \n {rand_tensor} \n")
print(f"Ones Tensor: \n {ones_tensor} \n")
print(f"Zeros Tensor: \n {zeros_tensor}")

# Tensor attributes describe their shape, datatype, and the device on which they are stored.
tensor = torch.rand(3, 4)

print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")


''' TENSOR OPERATIONS '''
#move the tensor to the GPU if available
if torch.cuda.is_available():
  tensor = tensor.to('cuda')
  print(f"Device tensor is stored on: {tensor.device}")

# numpy style indexing and slicing
tensor = torch.ones(4, 4)
tensor[:,1] = 0
print(tensor)

# join tensors
t1 = torch.cat([tensor, tensor, tensor], dim=1)
print(t1)

# multiply tensors
# This computes the element-wise product
print(f"tensor.mul(tensor) \n {tensor.mul(tensor)} \n")
# Alternative syntax:
print(f"tensor * tensor \n {tensor * tensor}")

# operations with _ are in place
print(tensor, "\n")
tensor.add_(5)
print(tensor)

# tensor to numpy array
t = torch.ones(5)
print(f"t: {t}")
n = t.numpy()
print(f"n: {n}")

# numpy array to tensor
n = np.ones(5)
t = torch.from_numpy(n)






