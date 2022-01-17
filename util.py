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

    return j

def vorticity(x):
    dudy = x[:,1:,:,0] - x[:,:-1,:,0]
    dvdx = x[:,:,1:,1] - x[:,:,:-1,1]
    
    dvdx = torch.cat([dvdx, torch.unsqueeze(dvdx[:,:,-1], dim=2)], dim=2)
    dudy = torch.cat([dudy, torch.unsqueeze(dudy[:,-1,:], dim=1)], dim=1)
    
    w = torch.unsqueeze(dvdx - dudy, dim=-1)

    return w

def to_img(img):
    if img.shape[-1] == 1:
        tmp = torch.ones(img.shape, device=img.device)
        img_r = torch.cat((tmp, img*2+1, img*2+1), dim=-1)
        img_b = torch.cat((1-img*2, 1-img*2, tmp), dim=-1)
        img = torch.cat((img, img, img), dim=-1)
        img = torch.where(img < 0, img_r, img_b)
    elif img.shape[-1] == 2:
        shape = list(img.shape[:-1]) + [3 - img.shape[-1]]
        img = torch.cat((img, torch.zeros(shape, device=img.device)), dim=-1)
    return torch.clamp(img * .5 + .5, 0, 1)

def to_img_uint(img):
    if img.shape[-1] == 1:
        img[torch.abs(img)>1e-3] *= 10
        # img[torch.abs(img)<1e-3] = 0
        tmp = torch.ones(img.shape, device=img.device)
        img_r = torch.cat((tmp, img*2+1, img*2+1), dim=-1)
        img_b = torch.cat((1-img*2, 1-img*2, tmp), dim=-1)
        img = torch.cat((img, img, img), dim=-1)
        img = torch.where(img < 0, img_r, img_b)
    elif img.shape[-1] == 2:
        img[torch.abs(img)>1e-3] *= 10
        # img[torch.abs(img)<1e-3] = 0
        shape = list(img.shape[:-1]) + [3 - img.shape[-1]]
        img = torch.cat((img, torch.zeros(shape, device=img.device)), dim=-1)
    return torch.tensor((torch.clamp(img*0.5 + .5, 0, 1)*255).detach(), dtype=torch.uint8)
