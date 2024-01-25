import torch
import torch.nn as nn
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler, DDPMSchedulerOutput
import tqdm
from positionalencoding import SinusoidalPosEmb
import matplotlib.pyplot as plt

# Note: We can sample random diffusion timesteps for each item.
# We do not have to perform the diffusion loss on all steps.

"""
When training, you can just use straight up L2 loss.
When evaluating, you should use the noise scheduler step function.
"""
class Model(nn.Module):
    def __init__(self, dim=256):
        super().__init__()

        self.dim = dim
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=dim,
                nhead=8,
                batch_first=True,
            ),
            num_layers=8,
        )
        self.embedding_to_predicted_noise = nn.Linear(dim, 2)
        self.spatial_embedding = SinusoidalPosEmb(dim//2, max_period=4)
        self.diffusion_step_embedding = SinusoidalPosEmb(dim, max_period=4)
        self.special_tokens = nn.Embedding(1, dim)
        self.special_pos_tokens = nn.Embedding(2, dim//2)
        self.time_embedding = nn.Embedding(40, dim//2)
        self.XY = nn.Linear(2, dim//2)

    def forward(self, current_location, target_location, noisy_trajectory, diffusion_timestep):
        """
        target_location: (batch_size, 2)
        noisy_trajectory: (batch_size, prediction_horizon, 2)
        diffusion_timestep: (batch_size,)
        """
        B = target_location.shape[0]
        target_location_embedded = torch.cat([self.XY(target_location), self.special_pos_tokens(torch.zeros(B, dtype=torch.long))], dim=-1)
        current_location_embedded = torch.cat([self.XY(current_location), self.special_pos_tokens(torch.ones(B, dtype=torch.long))], dim=-1)
        noisy_trajectory_embedded = torch.cat([self.XY(noisy_trajectory), self.time_embedding(torch.arange(40).unsqueeze(0).repeat(B, 1))], dim=-1)

        # X = target_location[:, 0]
        # Y = target_location[:, 1]
        # target_location_embedded = torch.cat((self.spatial_embedding(X), self.spatial_embedding(Y)), dim=-1)

        # X = current_location[:, 0]
        # Y = current_location[:, 1]
        # current_location_embedded = torch.cat((self.spatial_embedding(X), self.spatial_embedding(Y)), dim=-1)

        # X = noisy_trajectory[:, :, 0]
        # Y = noisy_trajectory[:, :, 1]
        # noisy_trajectory_embedded = torch.cat((self.spatial_embedding(X), self.spatial_embedding(Y)), dim=-1)

        # Reuse the sequence embedding for target and diffusion time steps
        diffusion_timestep_embedded = self.diffusion_step_embedding(diffusion_timestep) + self.special_tokens(torch.tensor([0]))
        # current_location_embedded = current_location_embedded + self.sequence_embedding(torch.tensor([41]))
        # target_location_embedded = target_location_embedded + self.sequence_embedding(torch.tensor([42]))
        # Use 40-step prediction horizon
        # noisy_trajectory_embedded = noisy_trajectory_embedded + self.sequence_embedding(torch.arange(40))

        # Concatenate results across sequence dimension and predict denoising values
        to_concat = (noisy_trajectory_embedded, diffusion_timestep_embedded.unsqueeze(1), current_location_embedded.unsqueeze(1), target_location_embedded.unsqueeze(1))
        # print("Tensor shapes:")
        # for tensor in to_concat:
        #     print(tensor.shape)
        embedded = torch.cat(to_concat, dim=1)
        embedded = self.encoder(embedded)
        predicted_noise = self.embedding_to_predicted_noise(embedded[:, :40, :])

        return predicted_noise

# 1. Train a model to approximate a linear path between two points
# model = Model()
import condunet
model = condunet.ConditionalUnet1D(input_dim=2)
optim = torch.optim.Adam(model.parameters(), lr=1e-4)
lr_schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=256)

NUM_TRAIN_DIFFUSION_TIMESTEPS = 10

scheduler = DDPMScheduler(num_train_timesteps=NUM_TRAIN_DIFFUSION_TIMESTEPS, clip_sample=False)
scheduler.set_timesteps(NUM_TRAIN_DIFFUSION_TIMESTEPS)

train = True

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

            noise = torch.randn_like(trajectories)
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
                noisy_trajectory,
                diffusion_timestep,
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
# exit()

xmin = min(start[0, 0].item(), end[0, 0].item()) - 5
xmax = max(start[0, 0].item(), end[0, 0].item()) + 5
ymin = min(start[0, 1].item(), end[0, 1].item()) - 5
ymax = max(start[0, 1].item(), end[0, 1].item()) + 5

noisy_trajectory = torch.randn((1, 40, 2))

# for diffusion_timestep in range(100):
for diffusion_timestep in range(NUM_TRAIN_DIFFUSION_TIMESTEPS - 1, -1, -1):
    predicted_noise = model.forward(
        # start,
        # end,
        noisy_trajectory,
        torch.tensor([diffusion_timestep]),
    )
    noisy_trajectory = scheduler.step(predicted_noise, torch.tensor([diffusion_timestep]), noisy_trajectory).prev_sample # type: ignore

    plt.title("Example trajectory")
    plt.scatter(noisy_trajectory[0, :, 0].detach().numpy(), noisy_trajectory[0, :, 1].detach().numpy(), label='Sampled trajectory')
    plt.scatter(x, y)
    print("Loss:", torch.nn.functional.mse_loss(noisy_trajectory, trajectories).item())
    plt.scatter(start[0, 0], start[0, 1], c='r', label='Current location')
    plt.scatter(end[0, 0], end[0, 1], c='g', label='Target location')

    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    plt.legend()
    plt.show()
