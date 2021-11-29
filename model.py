import torch
import torch.nn as nn
import torch.nn.functional as F

class Network(nn.Module):
	def __init__(self, shape):
		super(Network, self).__init__()
		self.linear_relu_stack = nn.Sequential(
			nn.Linear(3, 512),
			nn.ReLU(),
			nn.Linear(512, 512),
			nn.ReLU(),
			nn.Linear(512, shape.prod(dtype=torch.int).item()),
		)

	def forward(self, x):
		output = self.linear_relu_stack(x)
		return output
