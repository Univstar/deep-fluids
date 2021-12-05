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
    with torch.no_grad():
        output_0 = run(model, data_0)
        vort_0 = vorticity(output_0)

        out_vel_img = torch.cat(output_0.split(1), dim=2).squeeze(dim=0).flip([0])
        tgt_vel_img = torch.cat(target_0.split(1), dim=2).squeeze(dim=0).flip([0])
        vel_img = torch.cat((out_vel_img, tgt_vel_img), dim=0)

        out_vort_img = torch.cat(vort_0.split(1), dim=2).squeeze(dim=0).flip([0])
        out_vort_img /= torch.max(torch.abs(out_vort_img))
        tgt_vort_img = torch.cat(vort_0_gt.split(1), dim=2).squeeze(dim=0).flip([0])
        tgt_vort_img /= torch.max(torch.abs(tgt_vort_img))
        vort_img = torch.cat((out_vort_img, tgt_vort_img), dim=0)

        logger.log_image(to_img(vel_img), 'vel_vs_gt', iter, 'HWC')
        logger.log_image(to_img(vort_img), 'vort_vs_gt', iter, 'HWC')

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

def test(name, model, data_set):
    
    import cv2.cv2 as cv
    model.eval()
    data_name = 'data/liquid/v/0_0_%d.npz'
    idx = 0
    gt = []
    pred = []
    m = torch.nn.Upsample(scale_factor=10, mode='bilinear', align_corners=True)
    for i in range(200):
        p, v = data_set.get_item_by_name(data_name%i)
        output = m(model(p))
        v = nchw_to_nhwc( m(nhwc_to_nchw(v.unsqueeze(0)))).squeeze(0).flip(0)
        if output.shape[1] == 1: output = curl(output)
        else: output = nchw_to_nhwc(output).squeeze(0)
        # gt_vort = vorticity(torch.unsqueeze(v, 0)).squeeze(0)
        # pred_vort = vorticity(torch.unsqueeze(output, 0)).squeeze(0)
        gt.append(to_img_uint(v).detach().numpy())
        pred.append(to_img_uint(output.flip(0)).detach().numpy())  
        # print("???")
    size = list(reversed(gt[0].shape[:-1]))
    
    gt_out = cv.VideoWriter(name+'_gt.avi', cv.VideoWriter_fourcc(*'MJPG'), 20, size)
    pred_out = cv.VideoWriter(name+'_pred.avi', cv.VideoWriter_fourcc(*'MJPG'), 20, size)
    
    for i in range(len(gt)):
        rgb_img = cv.cvtColor(gt[i], cv.COLOR_RGB2BGR)
        gt_out.write(rgb_img)
        rgb_img = cv.cvtColor(pred[i], cv.COLOR_RGB2BGR)
        pred_out.write(rgb_img)
    gt_out.release()
    pred_out.release()
    