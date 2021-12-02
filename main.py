import argparse
import torch
from torch.utils.data import Dataset
import time
from tqdm import tqdm
import os

from data import FluidDataset
from model import DFModel
from util import *
from logger import Logger

def train(args, model, device, loader, optimizer, epoch, log_param):
    model.train()
    loop = tqdm(enumerate(loader), total=len(loader))
    loop.set_description(f'Epoch [{epoch}/{args.epochs}]')
    loss_fn = torch.nn.L1Loss()
        
    for i, (data, target) in loop:
        data, target = data.to(device), target.to(device)
        jaco_gt, vort_gt = jacobian(target)
        optimizer.zero_grad()

        output = model(data)
        if args.no_curl: output = nchw_to_nhwc(output)
        else: output = curl(output)
        jaco, vort = jacobian(output)

        loss = loss_fn(output, target) + loss_fn(jaco, jaco_gt)
        loss.backward()

        optimizer.step()
        iter = (epoch - 1) * len(loader) + i
        if iter % log_param['log_freq'] == 0:
            log_param['cur_step'] += 1
            log_param['logger'].log_scalar(loss, 'loss', log_param['cur_step'])
        
            n_img = output.shape[0]
            for i in range(min(4, n_img)):
                total_img = torch.cat([output[i], target[i]], dim=1)
                log_param['logger'].log_image(to_img(total_img), f'vel_vs_gt_{i}', log_param['cur_step'], 'HWC')
                vort_img = torch.cat([vort[i], vort_gt[i]], dim=1)
                log_param['logger'].log_image(to_img(vort_img), f'vort_vs_gt{i}', log_param['cur_step'], 'HWC')
            
            log_param['logger'].flush()
                
        loop.set_postfix(loss=f'{loss.item():.6e}')
        
    return loss

def main():
    # Initialize training settings
    parser = argparse.ArgumentParser(description='Deep fluids: a PyTorch version')
    parser.add_argument('--test',            action='store_true', default=False,   help='selects the test mode')
    parser.add_argument('--no-cuda',         action='store_true', default=False,   help='disables the cuda device')
    parser.add_argument('--no-curl',         action='store_true', default=False,   help='disables curl operator')
    parser.add_argument('--name',            type=str,            default='smoke', help='the name of the dataset')
    parser.add_argument('--batch-size',      type=int,            default=8,       help='input batch size for training (default: 8)')
    parser.add_argument('--test-batch-size', type=int,            default=100,     help='input batch size for testing (default: 100)')
    parser.add_argument('--seed',            type=int,            default=1,       help='random seed (default: 1)')
    parser.add_argument('--num-workers',     type=int,            default=1,       help='the number of data loading workers (default : 1)')
    parser.add_argument('--epochs',          type=int,            default=100,     help='number of epochs to train (default: 100)')
    parser.add_argument('--lr-max',          type=float,          default=1e-4,    help='maximal learning rate (default: 1e-4)')
    parser.add_argument('--lr-min',          type=float,          default=2.5e-6,  help='minimal learning rate (default: 2.5e-6)')
    parser.add_argument('--beta-1',          type=float,          default=0.5,     help='smoothing coefficient beta_1 (default: 0.5)')
    parser.add_argument('--beta-2',          type=float,          default=0.999,   help='smoothing coefficient beta_2 (default: 0.999)')
    parser.add_argument('--num-conv',        type=int,            default=4,       help='the number of convolutional layers (default: 4)')
    parser.add_argument('--num-chnl',        type=int,            default=128,     help='the number of channels (default: 128)')
    parser.add_argument('--log-freq',        type=int,            default=200,     help='num of subiters between two logs (default: 200)')

    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    # Initialize logger.
    time_now = time.strftime("%Y%m%d_%H-%M-%S", time.localtime())
    log_dir = os.path.join('log/', args.name + '_' + time_now)
    os.makedirs(log_dir, exist_ok=True)
    logger = Logger(log_dir)

    # Initialize torch with kwargs.
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f'\033[36m[*] Device: {device}\033[0m')
    kwargs = {
        'batch_size': (args.test_batch_size if args.test else args.batch_size),
        'num_workers': args.num_workers
    }
    if not args.test: kwargs.update({'shuffle': True})
    if use_cuda: kwargs.update({'pin_memory': True})
    dataset = FluidDataset("data/" + args.name)
    loader = torch.utils.data.DataLoader(dataset, **kwargs)
    
    # Set network and run.
    model = DFModel(
        dataset.cnt_p,
        args.num_chnl,
        args.num_conv,
        (dataset.res_y, dataset.res_x, (2 if args.no_curl else 1))
    )
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), args.lr_max, [args.beta_1, args.beta_2])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, args.lr_min)

    log_param = {
        'logger': logger,
        'log_freq': args.log_freq,
        'cur_step': 0
    }
    
    for epoch in range(1, args.epochs + 1):
        if not args.test:
            loss = train(args, model, device, loader, optimizer, epoch, log_param)
        else:
            pass
        scheduler.step()

    if not args.test:
        torch.save(model.state_dict(), os.path.join(log_dir, 'weight.pt'))

if __name__ == '__main__':
    main()
