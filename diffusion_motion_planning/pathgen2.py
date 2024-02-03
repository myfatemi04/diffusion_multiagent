import torch
import torch.nn as nn
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler, DDPMSchedulerOutput
import tqdm
import matplotlib.pyplot as plt

# 1. Train a model to approximate a linear path between two points
# model = Model()
import condunet

device = torch.device('cuda')

model = condunet.ConditionalUnet1D(input_dim=2).to(device)
optim = torch.optim.Adam(model.parameters(), lr=1e-4)
lr_schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=256)

NUM_TRAIN_DIFFUSION_TIMESTEPS = 10

scheduler = DDPMScheduler(num_train_timesteps=NUM_TRAIN_DIFFUSION_TIMESTEPS, clip_sample=False)
scheduler.set_timesteps(NUM_TRAIN_DIFFUSION_TIMESTEPS,device=device)

train = False

if train:
    plot_trajectory = False

    steps = 256
    with tqdm.tqdm(total=steps) as pbar:
        for epoch in range(steps):
            # Generate a random trajectory
            n_samples = 16
            start_locations = torch.ones((n_samples, 2)) * 1
            target_locations = torch.ones((n_samples, 2)) * -1
            # start_locations = torch.randn((n_samples, 2)) * 10
            # target_locations = torch.randn((n_samples, 2)) * 10
            trajectories = torch.linspace(0, 1, 40).unsqueeze(-1).repeat(1, 1, 2)
            trajectories = start_locations.unsqueeze(1) + trajectories * (target_locations - start_locations).unsqueeze(1)
            trajectories=trajectories.to(device)

            noise = torch.randn_like(trajectories).to(device)
            diffusion_timestep = torch.randint(0, NUM_TRAIN_DIFFUSION_TIMESTEPS, (n_samples,))

            noisy_trajectory = scheduler.add_noise(trajectories, noise, diffusion_timestep) # type: ignore

            # Test: Plot current location, target location, and trajectory
            if plot_trajectory:
                plt.title("Example trajectory")
                plt.scatter(trajectories[0, :, 0], trajectories[0, :, 1], label='Sampled trajectory')
                plt.scatter(noisy_trajectory[0, :, 0], noisy_trajectory[0, :, 1], label='Noisy trajectory')
                plt.scatter(start_locations[0, 0], start_locations[0, 1], c='r', label='Current location')
                plt.scatter(target_locations[0, 0], target_locations[0, 1], c='g', label='Target location')
                plt.legend()
                plt.show()

                exit()

            predicted_noise = model.forward(
                # start_locations,
                # target_locations,
                noisy_trajectory.to(device),
                diffusion_timestep.to(device),
            )
            # mse loss baseline is variance
            loss = torch.nn.functional.mse_loss(predicted_noise, noise)

            # Backpropagate
            optim.zero_grad()
            loss.backward()
            optim.step()
            lr_schedule.step()

            pbar.update()
            pbar.set_description(f"loss: {loss.item():.2f}")

    # Save model.
    torch.save(model.state_dict(), "pathgen.pt")

model.load_state_dict(torch.load("pathgen.pt"))

# Attempt inference.
# 1. Set startpoint as (-1, -1) and endpoint as (1, 1)
# 2. Generate a random trajectory and denoise to a linear one.

# start = torch.tensor([-1, -1]).unsqueeze(0) * 1.0
# end = torch.tensor([1, 1]).unsqueeze(0) * 1.0
# start = torch.randn((1, 2)) * 4
# end = torch.randn((1, 2)) * 4

start = torch.ones((1, 2)) * 2
end = torch.ones((1, 2)) * -2
trajectories = torch.linspace(0, 1, 40).unsqueeze(-1).repeat(1, 1, 2)
trajectories = start.unsqueeze(1) + trajectories * (end - start).unsqueeze(1)
x = trajectories[0, :, 0]
y = trajectories[0, :, 1]
print(trajectories.shape)
plt.scatter(x, y)
plt.scatter(start[0, 0], start[0, 1], c='r', label='Current location')
plt.scatter(end[0, 0], end[0, 1], c='g', label='Target location')
plt.show()
trajectories = trajectories.to(device)
# exit()

xmin = min(start[0, 0].item(), end[0, 0].item()) - 5
xmax = max(start[0, 0].item(), end[0, 0].item()) + 5
ymin = min(start[0, 1].item(), end[0, 1].item()) - 5
ymax = max(start[0, 1].item(), end[0, 1].item()) + 5

noisy_trajectory = torch.randn((1, 40, 2)).to(device)

# for diffusion_timestep in range(100):
for diffusion_timestep in range(NUM_TRAIN_DIFFUSION_TIMESTEPS - 1, -1, -1):
    ts = torch.tensor([1]).to(device)
    predicted_noise = model.forward(
        # start,
        # end,
        noisy_trajectory.to(device),
        ts,
    )
    # noisy_trajectory = scheduler.step(predicted_noise.to('cpu'), ts.to('cpu'), noisy_trajectory.to('cpu')).prev_sample.to('cuda') # type: ignore
    noisy_trajectory = noisy_trajectory - predicted_noise

    plt.title("Example trajectory")
    plt.scatter(noisy_trajectory[0, :, 0].detach().cpu().numpy(), noisy_trajectory[0, :, 1].detach().cpu().numpy(), label='Sampled trajectory')
    plt.scatter(x, y)
    print("Loss:", torch.nn.functional.mse_loss(noisy_trajectory, trajectories).item())
    plt.scatter(start[0, 0], start[0, 1], c='r', label='Current location')
    plt.scatter(end[0, 0], end[0, 1], c='g', label='Target location')

    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    plt.legend()
    plt.show()
