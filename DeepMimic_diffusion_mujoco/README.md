# DiffMimic

This is Kenji's final year project with diffusion models.
Research idea: If a model learns how to do something, it should be able to adapt its skills to different conditions.

The project's goal is to get a model to learn a motion and then see how the motion adapts to different conditions we set to the model (ie interacting with objects, moving along trajectories, changes in environment etc).

In this repo, there is

1. Code for different architectures to turn a mocap motion as a diffusion model sampling process
   1. Vanilla DDPM with 1D diffusion (in the main branch)
   2. MDM (in the mdm branch)
   3. Diffuser (in the main branch)
2. A way to apply constraints to the sampling process to alter the generated motion. A collection of notebooks showcasing different applications of motion diffusion models have been preparred
   1. `0_temporal_unet_diffusion.ipynb` - The base motion generation code
   2. `1_motion_editing.ipynb` - A notebook experimenting with editing the motion by changing the joint positions
   3. `2_starting_with_motion.ipynb` - A notebook experimenting with starting the sampling process with a motion instead of random noise
   4. `3_long_projection.ipynb` - A notebook experimenting with sampling with containing more frames than what the model was trained on
   5. `4_motion_inbetweening.ipynb` - A notebook experimenting with inbetweening between two motions using the inbetween way
   6. `5_motion_blending.ipynb` - A notebook experimenting with blending between transition of two motions using the blending way

## Examples

### Example 1: Interacting with objects

For example, we make a model learn a walking motion.

https://github.com/agent-lab/DeepMimic_diffusion_mujoco/assets/56083944/dd859e03-9e46-4336-8aab-e6df896bd932

Then we give it a constraint that it will need to hold a box, it should still know how to walk with the box in hand.

https://github.com/agent-lab/DeepMimic_diffusion_mujoco/assets/56083944/03e0459a-4408-4f02-b830-2c00eab90f06

### Example 2: Limiting use of joints

For example, we make a model learn how to do a cartwheel

https://github.com/agent-lab/DeepMimic_diffusion_mujoco/assets/56083944/d0526f4b-bd6b-4b09-8183-36603a5745f8

Will it be able to do an aerial if we limit its usage of its hands?

https://github.com/agent-lab/DeepMimic_diffusion_mujoco/assets/56083944/c19acca1-06fe-4624-be8d-9b783c065669

### Limitations

These mujoco simulations were not done under a physics simulator, so its not physically accurate, the main goal is just to check if we can make a model retain its learning of a skill under a different environment

## Quickstart

### Vanilla DDPM Setup
This library clashes with the other packages for the 
```bash
pip install denoising_diffusion_pytorch
```

### MDM Setup
The setup instructions are exactly the same as described in the MDM repository for the environment and packages. See the MDM branch for more details

### Diffuser Setup

Setup the conda environment with the following command:

```bash
conda env create -f environment.yml

conda activate diffmimic
```

### Running the main motion generation code

All the relevant code is kept within `temporal_unet_diffusion.ipynb`. Headers and comments are left in the notebook to explain what each cell does.

### Loading motion datasets

One key thing to mention about motion datasets, is that the number of frames we load must be a multiple of 8 because of how the U-Net works. So if you have a spinkick motion with 78 frames, we can only load 72 frames. The `motion_dataset` loader automatically chooses the maximum number of frames for you.

### Playing the generated motions

There are a few experiments already under the `logs` directory.
I have included a script to play the generated motions. To use it, run the following command:

```bash
python3 mocap_player.py logs/{exp_name}/sampled_motions/motion1.npy

eg: python3 mocap_player.py logs/walk-motion/sampled_motions/motion1.npy
```

### Sampled with constraints

We can apply conditioning directly in the `models/sampling_config.py` file under the `apply_conditioning` function.

- There are 2 versions of the `apply_conditioning` function, one is the identity function and the other is the actual conditioning function. This is just an easy way to turn conditioning on and off.
- Right now its just hardcoded to change the joint positions to look like the human model is holding a box. But essentially you can change these position tensors to anything you want.

In the joint configs, index 13-15 and 17-19 are shoulders representing an euler tuple, 16 and 20 are elbows representing a scalar rotation value in radians.

### Evaluating constrained sampling

View original motion data, all of the original motion mocaps are stored in the `diffusion/data/motions` directory. There is no CLI command to play the file, but you run the code in the following file to select a mocap and play it

```bash
python3 utils/mocap_v2.py
```

View sampled motion, all sampled motion are stored in the `diffusion/logs` directory, under `{experiment_name}/sampled_motions`. You can play these sampled motions using the following command

```bash
python3 mocap_player.py logs/test-constrained-sampling-holding-a-box/sampled_motions/motion1.npy
```
