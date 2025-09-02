import torch
from torchvision.models import resnet18, ResNet18_Weights

model = resnet18(weights=ResNet18_Weights.DEFAULT)
data = torch.rand(1,3,64,64)
labels = torch.rand(1,1000)

#forward pass
prediction = model(data) 

#loss calculation
loss = (prediction - labels).sum()

#backwards pass (backprop)
loss.backward()

#load optimizer
optim = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)

#initiate gradient descent by adjusting model paramaters by gradient
optim.step()

#differentiatin:

a = torch.tensor([2., 3.], requires_grad=True)
b = torch.tensor([6., 4.], requires_grad=True)

#create tensor from a and b
Q = 3*a**3 - b**2
#assume a and b are paranaters and Q is the error
# so derivative of a is 9*a**2 and same for b
#when we xcall .backward with Q we use the gradients and store that gradient as tensors grad attribute (require_grad)

external_grad = torch.tensor([1., 1.])
Q.backward(external_grad) #we need to pass this because we want to apply tx weigting since we are not giving backward a scalar (the loss)
#basiccally I want to backprop the sum of both elements of Q which are a and b

print(9*a**2 == a.grad)
print(-2*b == b.grad)



