import torch

n_steps = 10
beta_start = 0.1
beta_end = 0.001
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

def denoised(xt, noisepred, betat, alphabart):
    alphat = 1 - betat
    # 3.2: sigmat^2 = betat works fine
    # sigmat = torch.sqrt(betat)
    alphabart_prev = alphabart / alphat
    sigmat = torch.sqrt((1 - alphabart_prev) / (1 - alphabart) * betat)
    denoised = torch.rsqrt(alphat) * (xt - ((1 - alphat) / torch.sqrt(1 - alphabart)) * noisepred) + sigmat * torch.randn_like(xt)
    return denoised

# target: 2
current = torch.randn(1)
for t in range(n_steps, 0, -1):
    betat = betas[t - 1]
    alphat = 1 - betat
    alphabart = alpha_bar[t - 1]
    alphabart_prev = alphabart / alphat
    noisepred = current
    sigmat = torch.sqrt((1 - alphabart_prev) / (1 - alphabart) * betat)
    print(current, sigmat)
    current = denoised(current, noisepred, betat, alphabart)

# current = torch.tensor(0.0)
# for t in range(n_steps):
#     current = noised_next(current, betas[t])
#     print(current)
