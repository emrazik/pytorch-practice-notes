# *** Notes on Pytorch tutorial here: pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html ***

'''Background
Neural networks (NNs) are a collection of nested functions that are executed on some input data. These functions are defined by parameters (consisting of weights and biases), which in PyTorch are stored in tensors.

Training a NN happens in two steps:

Forward Propagation: In forward prop, the NN makes its best guess about the correct output. It runs the input data through each of its functions to make this guess.

Backward Propagation: In backprop, the NN adjusts its parameters proportionate to the error in its guess.
It does this by traversing backwards from the output, collecting the derivatives of the error with respect to the parameters of the functions (gradients), and optimizing the parameters using gradient descent. '''

import torch
from torchvision.models import resnet18, ResNet18_Weights

model = resnet18(weights=ResNet18_Weights.DEFAULT)
data = torch.rand(1, 3, 64, 64)
labels = torch.rand(1, 1000)

# run input data through model (forward pass)
prediction = model(data)

# use model's prediction and label to calculate loss
loss = (prediction - labels).sum()
# back propagation
loss.backward()
# then Autograd calcs and stores gradients for each model parameter. This data in parameter's .grad attribute

# load the optimizer
# register all parameters of the model in the optimizer
optim = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)

# initiates gradient descent. Optimizer adjusts each parameter by its gradient stored in .grad
optim.step()

# Other notes
'''torch.autograd tracks operations on all tensors which have their requires_grad flag set to True. 
For tensors that don’t require gradients, setting this attribute to False excludes it from the gradient computation DAG.

In a NN, parameters that don’t compute gradients are usually called frozen parameters. 
It is useful to “freeze” part of your model if you know in advance that you won’t need the gradients of those parameters (this offers some performance benefits by reducing autograd computations).
'''