import argparse
import torch
from torch.utils.data import Dataset
import time
import os

from data import FluidDataset
from model import DFModel
from logger import Logger
from trainer import *

def main():
    # Initialize settings.
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

    # Set dataset and loader.
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f'\033[36m[*] Device: {device}\033[0m')
    kwargs = {
        'batch_size': (args.test_batch_size if args.test else args.batch_size),
        'num_workers': args.num_workers,
        'shuffle': True
    }
    if use_cuda: kwargs.update({'pin_memory': True})
    dataset = FluidDataset("data/" + args.name)
    loader = torch.utils.data.DataLoader(dataset, **kwargs)
    
    # Set network.
    model = DFModel(
        dataset.cnt_p,
        args.num_chnl,
        args.num_conv,
        (dataset.res_y, dataset.res_x, (2 if args.no_curl else 1))
    )
    model = model.to(device)

    if not args.test:
        # Set logger.
        time_now = time.strftime("%Y%m%d_%H-%M-%S", time.localtime())
        log_dir = os.path.join('log/', args.name + '_' + time_now)
        os.makedirs(log_dir, exist_ok=True)
        logger = Logger(log_dir, args.log_freq)

        # Set trainer.
        optimizer = torch.optim.Adam(model.parameters(), args.lr_max, [args.beta_1, args.beta_2])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, args.lr_min)

        # Sample a batch for comparison randomly.
        batch_sampler = torch.utils.data.RandomSampler(dataset, replacement=True, num_samples=args.batch_size)
        batch_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, sampler=batch_sampler)
        batch = next(iter(batch_loader))

        train(model, device, loader, optimizer, scheduler, args.epochs, batch, logger)

        torch.save(model.state_dict(), os.path.join(log_dir, 'weight.pt'))

if __name__ == '__main__':
    main()
