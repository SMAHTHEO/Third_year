import os
import torch
from torch.utils.data import Dataset
from diffusion.utils.mocap_v2 import MocapDM
import numpy as np
from collections import namedtuple


Batch = namedtuple('Batch', 'trajectories conditions')

class SpinkickDiffuserDataset(Dataset):
    def __init__(self, motion_src_path):
        self.motion_data = []
        self.mocap_dm = MocapDM()

        self.mocap_dm.load_mocap(filepath=motion_src_path)

        data_config = self.mocap_dm.data_config

        data_config = np.array(data_config)
        # size = data_config.shape[0]
        # data_config = np.concatenate([data_config, np.zeros(size).reshape(size, 1)], axis=1)
        data_config = data_config[:64, :]

        # for i in range(len(data_config)):
        #     prefix = data_config[i:]
        #     suffix = data_config[:i]
        #     motion = prefix + suffix
        #     motion = torch.from_numpy(np.array(motion)).float()
        #     self.motion_data.append(motion)

        for _ in range(100):
            self.motion_data.append(torch.from_numpy(data_config).float())

        self.motion_data = torch.stack(self.motion_data)

    def __len__(self):
        return len(self.motion_data)

    def __getitem__(self, idx):
        motion = self.motion_data[idx]
        condition = self.get_conditions(motion)
        return Batch(trajectories=motion, conditions=condition)
    
    def get_conditions(self, data_configs):
        '''
            condition on current observation for planning
        '''
        return {0: data_configs[0]}

if __name__ == '__main__':
    dataset = SpinkickDiffuserDataset("/home/kenji/Fyp/DeepMimic_mujoco/diffusion/data/motions/humanoid3d_spinkick.txt")
    print(dataset[0], dataset[0].trajectories.shape)