import torch

@torch.no_grad()
def ode_euler_integration(model, x_0, condition, device):
    model.eval()
    model.to(device)

    eps = 1e-8
    n_steps = 100
    t = torch.linspace(eps, 1 - eps, n_steps + 1).to(device)
    x_0.to(device)

    for i in range(1, len(t)):
        t_prev = t[i - 1].unsqueeze(0)
        f_eval = model(x_0, t_prev, condition)
        x = x_0 + (t[i] - t[i - 1]) * f_eval
        x_0 = x

    return x
