import os
from glob import glob
import numpy as np
import torch
from torch.utils.data import Dataset

class FluidDataset(Dataset):
    def __init__(self, dir):
        self.dir = dir
        print(f"\033[32m[*] Loading the dataset from {self.dir}...")

        # Read data generation arguments.
        self.args = {}
        print(os.path.join(self.dir, 'args.txt'))
        with open(os.path.join(self.dir, 'args.txt'), 'r') as file:
            while True:
                line = file.readline()
                if not line: break
                key, value = line[:-1].split(': ')
                self.args[key] = value
        
        # Initialize meta data.
        self.paths = sorted(glob(f"{self.dir}/v/*"))
        self.cnt_p = int(self.args['num_param'])
        self.res_x = int(self.args['resolution_x'])
        self.res_y = int(self.args['resolution_y'])
        print(f"[*] - {len(self.paths)} items. {self.cnt_p} parameters. {self.res_x}x{self.res_y}.")
        
        # Set value ranges.
        r = np.loadtxt(os.path.join(self.dir, 'v_range.txt'))
        self.v_max = max(abs(r[0]), abs(r[1]))
        print(f"[*] - v_max: {self.v_max}")

        self.p_range = []
        self.p_num = []
        for i in range(self.cnt_p):
            pi_name = self.args[f'p{i}']
            pi_min = float(self.args[f'min_{pi_name}'])
            pi_max = float(self.args[f'max_{pi_name}'])
            pi_num = int(self.args[f'num_{pi_name}'])
            print(f"[*] - {pi_name}, {pi_num} values, ranged in [{pi_min}, {pi_max}].")
            self.p_range.append([pi_min, pi_max])
            self.p_num.append(pi_num)
        
        print("\033[0m", end="")


    def __getitem__(self, index):
        with np.load(self.paths[index]) as data:
            v = torch.from_numpy(data['x']).float()
            p = torch.from_numpy(data['y']).float()
        
        # Normalize value to [-1, +1].
        v /= self.v_max
        for i, ri in enumerate(self.p_range):
            p[i] = (p[i] - ri[0]) / (ri[1] - ri[0]) * 2 - 1
        
        return p, v
    
    def get_item_by_name(self, name):
        with np.load(name) as data:
            v = torch.from_numpy(data['x']).float()
            p = torch.from_numpy(data['y']).float()
        
        # Normalize value to [-1, +1].
        v /= self.v_max
        for i, ri in enumerate(self.p_range):
            p[i] = (p[i] - ri[0]) / (ri[1] - ri[0]) * 2 - 1
        
        return p, v
        
    def __len__(self):
        return len(self.paths)
