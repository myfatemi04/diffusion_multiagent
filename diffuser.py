import torch
import matplotlib.pyplot as plt

n_steps = 1000
beta_start = 1e-4
beta_end = 0.2
# base_beta = 0.001
# Constant schedule, can be changed.
# betas = torch.ones(n_steps) * base_beta
betas = torch.linspace(beta_start, beta_end, n_steps)
alpha_bar = torch.cumprod(1 - betas, dim=0)

def noised_next(xt, betat):
    # Equation (2) of DDPMs
    return torch.sqrt(1 - betat) * xt + torch.sqrt(betat) * torch.randn_like(xt)

def noised_any_step(x0, alpha_bar_t):
    # Equation (4) of DDPMs
    return torch.sqrt(alpha_bar_t) * x0 + torch.sqrt(1 - alpha_bar_t) * torch.randn_like(x0)

def denoised(xt, noisepred, betat, alphabart, is_first_step):
    alphat = 1 - betat
    # 3.2: sigmat^2 = betat works fine
    # sigmat = torch.sqrt(betat)
    alphabart_prev = alphabart / alphat
    sigmat = torch.sqrt((1 - alphabart_prev) / (1 - alphabart) * betat)
    denoised = torch.rsqrt(alphat) * (xt - ((1 - alphat) / torch.sqrt(1 - alphabart)) * noisepred) + (sigmat * torch.randn_like(xt) if not is_first_step else 0)
    # denoised = torch.rsqrt(alphabart) * xt - (torch.rsqrt(alphabart) - 1) * noisepred
    # denoised = torch.rsqrt(alphat) * (xt - ((1 - alphat) / torch.sqrt(1 - alphabart)) * noisepred) + (sigmat * torch.randn_like(xt) if not is_first_step else 0)
    print(xt, noisepred, ((1 - alphat) / torch.sqrt(1 - alphabart)), alphat, torch.rsqrt(alphat))
    return denoised

# target: 0
reverse_process_results = []
original_values = []
for _ in range(1000):
    current = torch.randn(())
    original_values.append(current.item())
    history = [current.item()]
    history_noise = []
    for t in range(n_steps, 0, -1):
        betat = betas[t - 1]
        alphabart = alpha_bar[t - 1]
        # alphat = 1 - betat
        # alphabart_prev = alphabart / alphat
        # sigmat = torch.sqrt((1 - alphabart_prev) / (1 - alphabart) * betat)
        noisepred = current - 2
        # print(current, sigmat)
        history_noise.append(noisepred.item())
        history.append(current.item())
        current = denoised(current, noisepred, betat, alphabart, True) #t == 1)
    plt.plot(history, label='History')
    plt.plot(history_noise, label='Noise Prediction')
    plt.legend()
    plt.show()
    exit()
    reverse_process_results.append(current.item())

plt.hist(reverse_process_results, bins=100, alpha=0.5, label='reverse')
plt.hist(original_values, bins=100, alpha=0.5, label='original')
plt.legend()
plt.show()

# forward_process_samples = []
# instant_process_samples = []
# tests = 10000
# for _ in range(tests):
#     current = torch.tensor(0.0)
#     for t in range(n_steps):
#         current = noised_next(current, betas[t])
#     forward_process_samples.append(current)
#     instant_process_samples.append(noised_any_step(torch.tensor(0.0), alpha_bar[-1]))

# import matplotlib.pyplot as plt

# plt.hist(forward_process_samples, bins=100, alpha=0.5, label='forward')
# plt.hist(instant_process_samples, bins=100, alpha=0.5, label='instant')
# plt.legend()
# plt.show()