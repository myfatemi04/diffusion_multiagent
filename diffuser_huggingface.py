# Idea: Optimize a trajectory for deceptiveness via a diffusion model
# As a next step, we will create a simple reward function for a set of trajectories generated
# by robotics. We can log each step of the diffusion process as a decision variable generated
# by a policy network.

import torch
import torch.nn as nn

from diffusers.schedulers.scheduling_ddim import DDIMScheduler, DDIMSchedulerOutput
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler, DDPMSchedulerOutput

NUM_TRAIN_TIMESTEPS = 1000

scheduler = DDPMScheduler(num_train_timesteps=NUM_TRAIN_TIMESTEPS, clip_sample=False)
scheduler.set_timesteps(NUM_TRAIN_TIMESTEPS)

# Point (x, y) coordinates
# data = torch.randn((4, 2)) * 0.1
results = []

for test in range(100):
    data = torch.randn((1, 1))
    for timestep in range(NUM_TRAIN_TIMESTEPS):
        predicted_noise = (data - 1) # + 0.01 * torch.randn((4, 2))

        assert isinstance(predicted_noise, torch.FloatTensor)
        assert isinstance(data, torch.FloatTensor)

        out = scheduler.step(predicted_noise, timestep, data)

        assert isinstance(out, DDPMSchedulerOutput)
        prev_sample = out.prev_sample
        pred_original_sample = out.pred_original_sample


        data = prev_sample
    results.append(data.item())

import matplotlib.pyplot as plt

plt.hist(results)
plt.show()
