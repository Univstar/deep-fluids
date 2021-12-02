import torch
import time
from tqdm import tqdm

from util import *

def train(model, device, loader, optimizer, scheduler, epochs, logger):
    model.train()

    for epoch in range(epochs):
        loop = tqdm(enumerate(loader), total=len(loader))
        loop.set_description(f'Epoch [{epoch+1}/{epochs}]')
        loss_fn = torch.nn.L1Loss()
            
        for i, (data, target) in loop:
            data, target = data.to(device), target.to(device)
            jaco_gt, vort_gt = jacobian(target)
            optimizer.zero_grad()

            output = model(data)
            if output.shape[1] == 1: output = curl(output)
            else: output = nchw_to_nhwc(output)
            jaco, vort = jacobian(output)

            loss = loss_fn(output, target) + loss_fn(jaco, jaco_gt)
            loss.backward()

            optimizer.step()

            iter = epoch * len(loader) + i + 1

            logger.log_scalar(loss.item(), 'loss', iter)
            if iter % logger.freq == 0:            
                n_img = output.shape[0]
                for j in range(min(4, n_img)):
                    total_img = torch.cat([output[j], target[j]], dim=1)
                    logger.log_image(to_img(total_img), f'vel_vs_gt_{j}', iter, 'HWC')
                    vort_img = torch.cat([vort[j], vort_gt[j]], dim=1)
                    logger.log_image(to_img(vort_img), f'vort_vs_gt{j}', iter, 'HWC')                
                logger.flush()
                    
            loop.set_postfix(loss=f'{loss.item():.6e}')
        
        scheduler.step()
