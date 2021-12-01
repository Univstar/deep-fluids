import torch

def nchw_to_nhwc(x):
	return x.permute(0, 2, 3, 1)

def nhwc_to_nchw(x):
	return x.permute(0, 3, 1, 2)

def curl(x):
	u = x[:, 0, 1:, :  ] - x[:, 0, :-1,  :]  # ds/dy
	v = x[:, 0,  :, :-1] - x[:, 0, :  , 1:]  # -ds/dx,
	u = torch.cat([u, torch.unsqueeze(u[..., -1, :], dim=1)], dim=1)
	v = torch.cat([v, torch.unsqueeze(v[..., :, -1], dim=2)], dim=2)
	c = torch.stack([u, v], dim=-1)
	return c

def jacobian(x):
    dudx = x[:,:,1:,0] - x[:,:,:-1,0]
    dudy = x[:,1:,:,0] - x[:,:-1,:,0]
    dvdx = x[:,:,1:,1] - x[:,:,:-1,1]
    dvdy = x[:,1:,:,1] - x[:,:-1,:,1]
    
    dudx = torch.cat([dudx, torch.unsqueeze(dudx[:,:,-1], dim=2)], dim=2)
    dvdx = torch.cat([dvdx, torch.unsqueeze(dvdx[:,:,-1], dim=2)], dim=2)
    dudy = torch.cat([dudy, torch.unsqueeze(dudy[:,-1,:], dim=1)], dim=1)
    dvdy = torch.cat([dvdy, torch.unsqueeze(dvdy[:,-1,:], dim=1)], dim=1)

    j = torch.stack([dudx,dudy,dvdx,dvdy], dim=-1)
    w = torch.unsqueeze(dvdx - dudy, dim=-1) # vorticity (for visualization)

    return j, w
