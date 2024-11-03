import os
import torch
from torch.utils.data import Dataset
from diffusion.utils.mocap_v2 import MocapDM
import numpy as np
from collections import namedtuple


Batch = namedtuple("Batch", "trajectories conditions")


class MotionDataset(Dataset):
    def __init__(self, motion_src_path, shuffle=False):
        self.motion_data = []
        self.mocap_dm = MocapDM()

        self.mocap_dm.load_mocap(filepath=motion_src_path)

        data_config = self.mocap_dm.data_config
        data_vel = self.mocap_dm.data_vel

        data_config = np.array(data_config)
        data_vel = np.array(data_vel)
        # size = data_config.shape[0]
        # data_config = np.concatenate([data_config, np.zeros(size).reshape(size, 1)], axis=1)

        num_frames = data_config.shape[0]
        if num_frames % 8 != 0: # temporal unet needs to be multiple of 8
            num_frames -= (num_frames % 8) # adjust to be maximum multiple of 8

        data_config = data_config[:num_frames, :]
        data_vel = data_vel[:num_frames, :]
        combined = np.concatenate([data_config, data_vel], axis=1)

        if shuffle:
            diff = combined[-1] - combined[0]
            print(diff[:3])
            print(diff[:2])
            for i in range(len(combined)):
                prefix = combined[i:].copy()
                suffix = combined[:i].copy()
                if i != 0 and i != len(combined) - 1:
                    suffix[:, :3] += diff[:3]
                    first_diff = prefix[0] - combined[0]
                    prefix[:, :3] -= first_diff[:3]
                    suffix[:, :3] -= first_diff[:3]
                prefix = torch.from_numpy(np.array(prefix)).float()
                suffix = torch.from_numpy(np.array(suffix)).float()
                motion = torch.cat([prefix, suffix], dim=0)
                self.motion_data.append(motion)
        else:
            for _ in range(100):
                self.motion_data.append(torch.from_numpy(combined).float())

        self.motion_data = torch.stack(self.motion_data)

    def __len__(self):
        return len(self.motion_data)

    def __getitem__(self, idx):
        motion = self.motion_data[idx]
        condition = self.get_conditions(motion)
        return Batch(trajectories=motion, conditions=condition)

    def get_conditions(self, data_configs):
        """
        condition on current observation for planning
        """
        return {0: data_configs[0]}


if __name__ == "__main__":
    dataset = MotionDataset(
        "/home/kenji/Fyp/DeepMimic_mujoco/diffusion/data/motions/humanoid3d_walk.txt"
    )
    print(dataset[0], dataset[0].trajectories.shape)
