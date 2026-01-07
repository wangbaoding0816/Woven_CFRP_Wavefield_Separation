import torch


def nll_gaussian(residual, sigma):
    return (0.5 * (residual ** 2) / (sigma ** 2) + torch.log(sigma)).mean()


def tv2d_from_grid(u, grid_size, batch_time, eps=1e-6):
    u = u.view(batch_time, grid_size, grid_size)
    dx = u[:, :, 1:] - u[:, :, :-1]
    dy = u[:, 1:, :] - u[:, :-1, :]
    tv = torch.mean(torch.sqrt(dx * dx + eps)) + torch.mean(torch.sqrt(dy * dy + eps))
    return tv


def _zscore(x, eps=1e-12):
    return (x - x.mean()) / (x.std() + eps)


def corrcoef_1d(a, b, eps=1e-12):
    a = a - a.mean()
    b = b - b.mean()
    return (a * b).mean() / (a.std() + eps) / (b.std() + eps)


def soft_p95_abs(u_abs_bt, tau=10.0):
    """
    u_abs_bt: (Bs, Bt) >=0
    soft-quantile 近似：用 softmax 权重逼近高分位
    """
    w = torch.softmax(tau * u_abs_bt, dim=1)
    return torch.sum(w * u_abs_bt, dim=1, keepdim=True)


def struct_stat_and_freq_loss(
    u_struct,
    Bs,
    Bt,
    F_db_target_si,
    mode="meanabs",
    lambda_mse=5e-4,
    lambda_corr=5e-4,
    p95_soft=True,
):
    """
    u_struct: (Bs*Bt,1) 按 (spatial,time) 展平
    F_db_target_si: (Bs,) 目标结构先验（dB），与 si 对齐
    """
    u_struct_bt = u_struct.view(Bs, Bt)
    u_abs = torch.abs(u_struct_bt)

    if mode == "p95abs":
        if p95_soft:
            stat = soft_p95_abs(u_abs, tau=10.0)
        else:
            stat = torch.quantile(u_abs, 0.95, dim=1, keepdim=True)
    else:
        stat = torch.mean(u_abs, dim=1, keepdim=True)

    stat_z = _zscore(stat.squeeze(1))
    F_z = _zscore(F_db_target_si)

    loss_mse = torch.mean((stat_z - F_z) ** 2)
    corr = corrcoef_1d(stat_z, F_z)
    loss_corr = 1.0 - corr

    loss = lambda_mse * loss_mse + lambda_corr * loss_corr
    return loss, float(loss_mse.item()), float(corr.item())
