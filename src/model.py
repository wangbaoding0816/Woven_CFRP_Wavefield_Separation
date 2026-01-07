import torch
from torch import nn


class PlainMLP(nn.Module):
    def __init__(self, in_dim=4, width=256, depth=12, out_dim=2, p_drop=0.1):
        super().__init__()
        layers = []
        last = in_dim
        for _ in range(depth):
            layers += [nn.Linear(last, width), nn.SiLU(), nn.Dropout(p_drop)]
            last = width
        layers += [nn.Linear(last, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, xyt):
        return self.net(xyt)


class BPINN2DWave(nn.Module):
    """
    Two-head version:
    u_hat = u_prop + u_struct
    inputs:
      xytf: (N,4) = [x,y,t,F]
    but:
      f_prop   uses [x,y,t] (3D)
      f_struct uses [x,y,t,F] (4D)
      f_sigma  uses [x,y,t,F] (4D)
    """

    def __init__(self, p_drop=0.05):
        super().__init__()
        self.f_prop = PlainMLP(in_dim=3, width=200, depth=6, out_dim=1, p_drop=p_drop)
        self.f_struct = PlainMLP(in_dim=4, width=120, depth=4, out_dim=1, p_drop=p_drop)
        self.f_sigma = PlainMLP(in_dim=4, width=120, depth=4, out_dim=1, p_drop=p_drop)

        self.softplus = nn.Softplus()
        self.c_raw = nn.Parameter(torch.tensor(0.5))

    def c(self):
        return self.softplus(self.c_raw) + 1e-6

    def forward(self, xytf):
        xyt = xytf[:, :3]
        u_prop = self.f_prop(xyt)
        u_struct = self.f_struct(xytf)
        sigma = self.softplus(self.f_sigma(xytf)) + 1e-6
        u_hat = u_prop + u_struct
        return u_hat, u_prop, u_struct, sigma

    def u_sigma(self, xytf):
        u_hat, _, _, sigma = self.forward(xytf)
        return u_hat, sigma

    def u(self, xytf):
        u_hat, _ = self.u_sigma(xytf)
        return u_hat
