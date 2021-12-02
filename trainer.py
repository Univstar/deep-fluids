import torch
import time
from tqdm import tqdm

from util import *

def run(model, data):
    output = model(data)
    if output.shape[1] == 1: output = curl(output)
    else: output = nchw_to_nhwc(output)
    return output

def compare_and_log(model, data_0, target_0, vort_0_gt, logger, iter):
    output_0 = run(model, data_0)
    vort_0 = vorticity(output_0)

    for j in range(data_0.shape[0]):
        total_img = torch.cat([output_0[j], target_0[j]], dim=1)
        logger.log_image(to_img(total_img), f'vel_vs_gt_{j}', iter, 'HWC')
        vort_img = torch.cat([vort_0[j], vort_0_gt[j]], dim=1)
        logger.log_image(to_img(vort_img), f'vort_vs_gt{j}', iter, 'HWC')                
    logger.flush()

def train(model, device, loader, optimizer, scheduler, epochs, batch_0, logger):
    model.train()

    data_0, target_0 = batch_0
    data_0, target_0 = data_0.to(device), target_0.to(device)
    vort_0_gt = vorticity(target_0)

    for epoch in range(epochs):
        loop = tqdm(enumerate(loader), total=len(loader))
        loop.set_description(f'Epoch [{epoch+1}/{epochs}]')
        loss_fn = torch.nn.L1Loss()
            
        for i, (data, target) in loop:
            data, target = data.to(device), target.to(device)
            jaco_gt = jacobian(target)
            optimizer.zero_grad()

            output = run(model, data)
            jaco = jacobian(output)

            loss = loss_fn(output, target) + loss_fn(jaco, jaco_gt)
            loss.backward()

            optimizer.step()

            iter = epoch * len(loader) + i + 1

            logger.log_scalar(loss.item(), 'loss', iter)
            if iter % logger.freq == 0: compare_and_log(model, data_0, target_0, vort_0_gt, logger, iter)
                    
            loop.set_postfix(loss=f'{loss.item():.6e}')
        
        scheduler.step()
