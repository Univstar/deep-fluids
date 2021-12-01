import torch

def curl(x):
	u = x[..., 1:, :] - x[..., :-1, :]  # ds/dy
	v = x[..., :, :-1] - x[..., :, 1:]  # -ds/dx,
	u = torch.cat([u, torch.unsqueeze(u[..., -1, :], dim=-2)], dim=-2)
	v = torch.cat([v, torch.unsqueeze(v[..., :, -1], dim=-1)], dim=-1)
	c = torch.cat([u, v], dim=-1)
	return c
