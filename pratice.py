import torch
x = torch.tensor([[1, 2, 3], [4, 5, 6]])
print(x)
print(x[:,:])
print(x[:,None, :])
print(x[:, :, None])