import os
from datetime import datetime

import matplotlib

matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import LightSource, LinearSegmentedColormap
from scipy.ndimage import gaussian_filter
from skimage import exposure

from .dataset import (
    build_output_dir,
    make_matrix_from_vector,
    make_roi_indices,
    pick_pkl,
    prepare_dataset_from_pkl,
    rc_to_serp_idx,
)
from .freq_prior import save_F_struct_db_map_and_sensitivity
from .losses import nll_gaussian, struct_stat_and_freq_loss, tv2d_from_grid
from .model import BPINN2DWave

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# ===========================
# 用户配置区（常改）
# ===========================
FS = 20e6
CROP = (450, 800)  # 训练用短窗；结构先验会用全时域
TRAIN_RATIO = 0.9
RAND_SEED = 42
SERPENTINE_VIS = True

w_data = 0.995
EPOCHS = 10000

# Dropout & MC 采样
DROPOUT_P = 0.05
MC_SAMPLES = 20

# 滤波配置（对原始时序数据）
FILTER_MODE = "none"
FREQ_LOW = 0.5e6
FREQ_HIGH = 1.2e6
FILTER_ORDER = 4

# 评估间隔
EVAL_INTERVAL = 5000

# 结构频域先验配置
FREQ_PRIOR_MODE = "band"
FREQ_F0_HZ = 0.01e6
FREQ_BAND_HZ = (0.2e6, 0.4e6)
FREQ_USE_HANN = True
FREQ_DB_REF_MODE = "p95"
FREQ_DB_EPS = 1e-12

# 结构先验损失权重
LAMBDA_FREQ_MSE = 5e-4
LAMBDA_FREQ_CORR = 5e-4
STRUCT_STAT_MODE = "meanabs"
P95_SOFT = True


# ===========================
# 采样器
# ===========================

def make_data_sampler_with_indices(bundle, train_idx, batch_spatial=512, batch_time=None, energy_weighted=True):
    U = bundle["U"]
    x = bundle["x_norm"]
    y = bundle["y_norm"]
    t = bundle["t_norm"]
    Ns, Nt = U.shape
    if batch_time is None:
        batch_time = Nt
    train_idx = torch.tensor(train_idx, dtype=torch.long)

    if energy_weighted and "E_t" in bundle:
        p = bundle["E_t"].cpu().numpy()
    else:
        p = None
    F_feat_vec = bundle.get("F_feat_vec", None)

    def sample():
        si = train_idx[torch.randint(0, len(train_idx), (batch_spatial,))]
        if p is None:
            ti = torch.randint(0, Nt, (batch_time,))
        else:
            ti = torch.tensor(np.random.choice(Nt, size=(batch_time,), p=p), dtype=torch.long)

        xs = x[si].unsqueeze(1).repeat(1, batch_time)
        ys = y[si].unsqueeze(1).repeat(1, batch_time)
        ts = t[ti].unsqueeze(0).repeat(batch_spatial, 1)

        if F_feat_vec is None:
            ffeat = torch.zeros((batch_spatial, batch_time), dtype=xs.dtype)
        else:
            ffeat = F_feat_vec[si].unsqueeze(1).repeat(1, batch_time)

        xytf = torch.stack([xs, ys, ts, ffeat], dim=-1).reshape(-1, 4)
        u = U[si][:, ti].reshape(-1, 1)
        return xytf, u, si, ti

    return sample


def make_reg_sampler(bundle, grid_size=32, batch_time=4):
    R, C, Nt = bundle["R"], bundle["C"], bundle["Nt"]
    t = bundle["t_norm"]
    F_feat_vec = bundle.get("F_feat_vec", None)

    def sample():
        y0 = np.random.randint(0, R - grid_size)
        x0 = np.random.randint(0, C - grid_size)
        ys = np.arange(y0, y0 + grid_size, dtype=np.int32)
        xs = np.arange(x0, x0 + grid_size, dtype=np.int32)

        XX, YY = np.meshgrid(xs, ys)

        x_norm = (XX.astype(np.float32) / (C - 1)) * 2.0 - 1.0
        y_norm = (YY.astype(np.float32) / (R - 1)) * 2.0 - 1.0
        if F_feat_vec is None:
            F_patch = np.zeros_like(x_norm, dtype=np.float32)
        else:
            si_patch = np.vectorize(lambda rr, cc: rc_to_serp_idx(int(rr), int(cc), C))(YY, XX)
            F_patch = F_feat_vec[torch.tensor(si_patch.reshape(-1), dtype=torch.long)].cpu().numpy()
            F_patch = F_patch.reshape(grid_size, grid_size).astype(np.float32)

        ti = torch.randint(0, Nt, (batch_time,))
        ts = t[ti].cpu().numpy().astype(np.float32)

        xytf_list = []
        for tt in ts:
            t_plane = np.full_like(x_norm, tt, dtype=np.float32)
            xytf = np.stack([x_norm.reshape(-1), y_norm.reshape(-1), t_plane.reshape(-1), F_patch.reshape(-1)], axis=1)
            xytf_list.append(xytf)

        xytf_all = np.concatenate(xytf_list, axis=0)
        return torch.tensor(xytf_all, dtype=torch.float32), grid_size, batch_time

    return sample


# ===========================
# 训练
# ===========================

def train_b_pinn(
    bundle,
    train_idx,
    epochs=10000,
    w_data=1.0,
    batch_spatial=1024,
    batch_time=None,
    lr=1e-3,
    device="cpu",
    dropout_p=DROPOUT_P,
    eval_interval=EVAL_INTERVAL,
    eval_bundle=None,
    eval_train_idx=None,
    eval_held_idx=None,
    eval_out_base_dir=None,
    eval_roi=None,
):
    if batch_time is None:
        batch_time = bundle["Nt"]

    data_sampler = make_data_sampler_with_indices(bundle, train_idx, batch_spatial, batch_time, energy_weighted=True)
    reg_sampler = make_reg_sampler(bundle, grid_size=32, batch_time=3)

    LAMBDA_PROP_SMOOTH = 1e-4
    LAMBDA_STRUCT_TV = 1e-4
    LAMBDA_STRUCT_L1 = 1e-5

    if "F_struct_db_vec" not in bundle:
        print("[WARN] bundle 内没有 F_struct_db_vec，将不会启用频域结构先验对齐损失。")
        use_freq_prior = False
    else:
        use_freq_prior = True
        F_struct_db_vec = bundle["F_struct_db_vec"].to(device)

    model = BPINN2DWave(p_drop=dropout_p).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-7)

    history = []

    for ep in range(1, epochs + 1):
        model.train()
        opt.zero_grad()

        xyt_d, u_d, si, ti = data_sampler()
        xyt_d = xyt_d.to(device)
        u_d = u_d.to(device)
        si = si.to(device)

        u_hat, u_prop, u_struct, sigma_pt = model(xyt_d)
        loss_data = nll_gaussian(u_d - u_hat, sigma_pt)

        xyt_r, gs, bt = reg_sampler()
        xyt_r = xyt_r.to(device)

        _, u_prop_r, u_struct_r, _ = model(xyt_r)
        loss_prop_smooth = tv2d_from_grid(u_prop_r, gs, bt)
        loss_struct_tv = tv2d_from_grid(u_struct_r, gs, bt)
        loss_struct_l1 = torch.mean(torch.abs(u_struct_r))

        loss = (
            w_data * loss_data
            + LAMBDA_PROP_SMOOTH * loss_prop_smooth
            + LAMBDA_STRUCT_TV * loss_struct_tv
            + LAMBDA_STRUCT_L1 * loss_struct_l1
        )

        freq_mse = np.nan
        freq_corr = np.nan
        if use_freq_prior and (LAMBDA_FREQ_MSE > 0 or LAMBDA_FREQ_CORR > 0):
            Bs = int(si.shape[0])
            Bt = int(ti.shape[0])
            F_target = F_struct_db_vec[si]

            loss_freq, freq_mse, freq_corr = struct_stat_and_freq_loss(
                u_struct=u_struct,
                Bs=Bs,
                Bt=Bt,
                F_db_target_si=F_target,
                mode=STRUCT_STAT_MODE,
                lambda_mse=LAMBDA_FREQ_MSE,
                lambda_corr=LAMBDA_FREQ_CORR,
                p95_soft=P95_SOFT,
            )
            loss = loss + loss_freq

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        scheduler.step()

        if ep % 10 == 0 or ep == 1:
            lr_now = scheduler.get_last_lr()[0]
            with torch.no_grad():
                rmse = torch.sqrt(((u_hat - u_d) ** 2).mean()).item()
                mae = torch.mean(torch.abs(u_hat - u_d)).item()
                sigma_mean = float(sigma_pt.mean().item())
                c_val = float(model.c().item())
                lp = float(loss_prop_smooth.item())
                lstv = float(loss_struct_tv.item())
                l1s = float(loss_struct_l1.item())

            print(
                f"[{ep}] total={loss.item():.4e} data(NLL)={loss_data.item():.4e} "
                f"propTV={lp:.2e} structTV={lstv:.2e} structL1={l1s:.2e} "
                f"freqMSE={freq_mse:.2e} freqCorr={freq_corr:.3f} "
                f"rmse={rmse:.4f} mae={mae:.4f} c={c_val:.3f} sigma_mean={sigma_mean:.3e} lr={lr_now:.2e}"
            )

            history.append(
                {
                    "epoch": ep,
                    "loss": float(loss.item()),
                    "loss_data": float(loss_data.item()),
                    "freq_mse_z": float(freq_mse) if np.isfinite(freq_mse) else np.nan,
                    "freq_corr": float(freq_corr) if np.isfinite(freq_corr) else np.nan,
                    "c": c_val,
                    "sigma_mean": sigma_mean,
                    "lr": float(lr_now),
                    "rmse_batch": float(rmse),
                    "mae_batch": float(mae),
                }
            )

        if (
            eval_interval > 0
            and ep % eval_interval == 0
            and eval_bundle is not None
            and eval_train_idx is not None
            and eval_held_idx is not None
            and eval_out_base_dir is not None
            and eval_roi is not None
        ):
            print(f"--- 开始进行第 {ep} 个 epoch 的评估 ---")
            eval_epoch_dir = os.path.join(eval_out_base_dir, f"eval_epoch_{ep:06d}")
            os.makedirs(eval_epoch_dir, exist_ok=True)

            save_combined_evaluation_gif(
                model,
                eval_bundle,
                device,
                eval_epoch_dir,
                ROI_X0=eval_roi[0],
                ROI_X1=eval_roi[1],
                ROI_Y0=eval_roi[2],
                ROI_Y1=eval_roi[3],
                fps=10,
            )

            print(f"--- 第 {ep} 个 epoch 的评估完成，结果已保存至: {eval_epoch_dir} ---")

    df_history = pd.DataFrame(history)
    return model, df_history


# ===========================
# 可视化与评估
# ===========================

def save_combined_evaluation_gif(model, bundle, device, out_dir, ROI_X0, ROI_X1, ROI_Y0, ROI_Y1, fps=10):
    print("\n>>> [GIF] 正在生成全时段四合一动态对比图 (这可能需要几分钟)...")
    save_path = os.path.join(out_dir, "monitoring_combined_4in1.gif")

    R, C, Nt = bundle["R"], bundle["C"], bundle["Nt"]
    U_full = bundle["U"].cpu().numpy()
    vmin_u = np.percentile(U_full, 1)
    vmax_u = np.percentile(U_full, 99)

    fig = plt.figure(figsize=(22, 5), constrained_layout=True)

    def update(t_idx):
        fig.clear()
        axs = fig.subplots(1, 4)

        t_us = float(bundle["t_grid"][t_idx].item() * 1e6)

        M_pred_mean_full, M_unc_full, _ = predict_time_slice_mc(model, bundle, t_idx, device=device, mc=MC_SAMPLES)

        gt_slice = U_full[:, t_idx]
        M_gt_full = make_matrix_from_vector(gt_slice, R, C, serpentine=SERPENTINE_VIS)

        M_gt = M_gt_full[ROI_Y0:ROI_Y1, ROI_X0:ROI_X1]
        M_pred = M_pred_mean_full[ROI_Y0:ROI_Y1, ROI_X0:ROI_X1]
        M_std = M_unc_full[ROI_Y0:ROI_Y1, ROI_X0:ROI_X1]

        M_data_resid = np.abs(M_gt - M_pred)

        sigma_safe = np.maximum(M_std, 1e-6)
        M_nll = 0.5 * ((M_gt - M_pred) ** 2 / sigma_safe**2 + 2 * np.log(sigma_safe) + np.log(2 * np.pi))

        extent = [ROI_X0, ROI_X1 - 1, ROI_Y0, ROI_Y1 - 1]

        im1 = axs[0].imshow(M_gt, origin="lower", extent=extent, cmap="viridis", vmin=vmin_u, vmax=vmax_u)
        axs[0].set_title("Ground Truth")
        plt.colorbar(im1, ax=axs[0], fraction=0.046, pad=0.04)

        im2 = axs[1].imshow(M_pred, origin="lower", extent=extent, cmap="viridis", vmin=vmin_u, vmax=vmax_u)
        axs[1].set_title("Pred Mean")
        plt.colorbar(im2, ax=axs[1], fraction=0.046, pad=0.04)

        vmax_dr = np.percentile(M_data_resid, 99)
        im3 = axs[2].imshow(
            M_data_resid,
            origin="lower",
            extent=extent,
            cmap="magma",
            vmin=0,
            vmax=max(1e-12, vmax_dr),
        )
        axs[2].set_title("Data Residual |Diff|")
        plt.colorbar(im3, ax=axs[2], fraction=0.046, pad=0.04)

        vmin_nll = np.percentile(M_nll, 1)
        vmax_nll = np.percentile(M_nll, 99)
        im4 = axs[3].imshow(
            M_nll,
            origin="lower",
            extent=extent,
            cmap="coolwarm",
            vmin=vmin_nll,
            vmax=max(vmin_nll + 1e-12, vmax_nll),
        )
        axs[3].set_title("NLL Heatmap")
        plt.colorbar(im4, ax=axs[3], fraction=0.046, pad=0.04)

        fig.suptitle(f"Time: {t_us:.2f} μs (Frame {t_idx}/{Nt})", fontsize=16)

        if t_idx % 5 == 0:
            print(f"   [GIF] 处理帧: {t_idx}/{Nt}", end="\r")

    anim = FuncAnimation(fig, update, frames=range(0, Nt), interval=100)

    try:
        anim.save(save_path, writer=PillowWriter(fps=fps))
        print(f"\n[GIF] 成功保存: {save_path}")
    except Exception as exc:
        print(f"\n[GIF] 保存失败: {exc}")

    plt.close(fig)


def save_training_loss_curves(df_history, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    fig1, ax1 = plt.subplots(figsize=(8, 4))
    ax1.plot(df_history["epoch"], df_history["loss_data"])
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Data Loss (NLL, log scale)")
    ax1.set_title("Data NLL vs. Epoch")
    ax1.set_yscale("log")
    ax1.grid(True, which="both", ls="--")
    fig1.tight_layout()
    fig1.savefig(os.path.join(out_dir, "curve_train_loss_data.png"), dpi=160)
    plt.close(fig1)

    fig3, ax3 = plt.subplots(figsize=(8, 4))
    ax3.plot(df_history["epoch"], df_history["loss"], label="Total Loss")
    ax3.plot(df_history["epoch"], df_history["loss_data"], label="Data Loss", linestyle="--", alpha=0.7)
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Loss (log scale)")
    ax3.set_title("Total Loss vs. Epoch")
    ax3.set_yscale("log")
    ax3.legend()
    ax3.grid(True, which="both", ls="--")
    fig3.tight_layout()
    fig3.savefig(os.path.join(out_dir, "curve_train_loss_total.png"), dpi=160)
    plt.close(fig3)

    if "freq_corr" in df_history.columns:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(df_history["epoch"], df_history["freq_corr"])
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Corr(u_struct_stat, F_struct_db)")
        ax.set_title("Frequency-prior alignment (corr) over training")
        ax.grid(True, alpha=0.4)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "curve_freq_prior_corr.png"), dpi=160)
        plt.close(fig)

    csv_path = os.path.join(out_dir, "training_history.csv")
    df_history.to_csv(csv_path, index=False, float_format="%.6g")
    print(f"\n训练历史曲线已保存到: {out_dir}")


def predict_time_slice_mc(model, bundle, t_idx, device="cpu", mc=MC_SAMPLES):
    R, C = bundle["R"], bundle["C"]
    Ns = bundle["Ns"]
    assert 0 <= t_idx < bundle["Nt"]
    F_feat_vec = bundle.get("F_feat_vec", None)

    x = bundle["x_norm"].unsqueeze(1)
    y = bundle["y_norm"].unsqueeze(1)
    t_val = bundle["t_norm"][t_idx].view(1, 1).repeat(Ns, 1)
    if F_feat_vec is None:
        f_val = torch.zeros((Ns, 1), dtype=x.dtype)
    else:
        f_val = F_feat_vec.view(-1, 1)

    xyt = torch.cat([x, y, t_val, f_val], dim=1).to(device)

    model.train()
    preds = []
    sigmas2 = []
    with torch.no_grad():
        for _ in range(mc):
            u_hat, sigma_pt = model.u_sigma(xyt)
            u_np = u_hat.cpu().numpy().reshape(R, C)
            s_np2 = (sigma_pt.cpu().numpy().reshape(R, C)) ** 2
            if SERPENTINE_VIS:
                u_np[1::2, :] = u_np[1::2, ::-1]
                s_np2[1::2, :] = s_np2[1::2, ::-1]
            preds.append(u_np)
            sigmas2.append(s_np2)

    preds = np.stack(preds, axis=0)
    sigmas2 = np.stack(sigmas2, axis=0)

    mean_u = preds.mean(axis=0)
    var_u = preds.var(axis=0)
    mean_s2 = sigmas2.mean(axis=0)

    total_std = np.sqrt(np.maximum(var_u + mean_s2, 1e-12))
    aleatoric_std = np.sqrt(np.maximum(mean_s2, 1e-12))
    return mean_u, total_std, aleatoric_std


def predict_time_slice_components(model, bundle, t_idx, device="cpu"):
    R, C = bundle["R"], bundle["C"]
    Ns = bundle["Ns"]
    F_feat_vec = bundle.get("F_feat_vec", None)

    x = bundle["x_norm"].unsqueeze(1)
    y = bundle["y_norm"].unsqueeze(1)
    t_val = bundle["t_norm"][t_idx].view(1, 1).repeat(Ns, 1)

    if F_feat_vec is None:
        f_val = torch.zeros((Ns, 1), dtype=x.dtype)
    else:
        f_val = F_feat_vec.view(-1, 1)

    xytf = torch.cat([x, y, t_val, f_val], dim=1).to(device)

    model.eval()
    with torch.no_grad():
        u_hat, u_prop, u_struct, _ = model(xytf)
        u_hat = u_hat.cpu().numpy().reshape(R, C)
        u_prop = u_prop.cpu().numpy().reshape(R, C)
        u_struct = u_struct.cpu().numpy().reshape(R, C)

    if SERPENTINE_VIS:
        u_hat[1::2, :] = u_hat[1::2, ::-1]
        u_prop[1::2, :] = u_prop[1::2, ::-1]
        u_struct[1::2, :] = u_struct[1::2, ::-1]
    return u_hat, u_prop, u_struct


def save_struct_maps(model, bundle, out_dir, ROI_X0=0, ROI_X1=80, ROI_Y0=0, ROI_Y1=80, t_show_idx=None):
    os.makedirs(out_dir, exist_ok=True)
    save_dir = os.path.join(out_dir, "struct_component_final")
    os.makedirs(save_dir, exist_ok=True)

    plt.rcParams.update({"font.size": 22})
    plt.rcParams["font.sans-serif"] = ["Times New Roman"]

    colors = ["#FFFF00", "#FF0000"]
    custom_cmap = LinearSegmentedColormap.from_list("struct_cmap", colors)

    Nt = bundle["Nt"]
    R, C = bundle["R"], bundle["C"]
    if t_show_idx is None:
        t_show_idx = Nt // 2

    print(">>> [Processing] 提取结构分量并应用平滑与自定义色标...")

    stack_abs_struct = []
    for t_idx in range(Nt):
        _, _, u_struct = predict_time_slice_components(model, bundle, t_idx, device=next(model.parameters()).device)
        roi_struct = u_struct[ROI_Y0:ROI_Y1, ROI_X0:ROI_X1]
        stack_abs_struct.append(np.abs(roi_struct))

    stack_abs_struct = np.stack(stack_abs_struct, axis=0)
    p95_map = np.percentile(stack_abs_struct, 95, axis=0)

    def enhance_and_smooth(M, sigma=0.2):
        M_smooth = gaussian_filter(M, sigma=sigma)
        M_centered = M_smooth - np.median(M_smooth)
        M_min, M_max = M_centered.min(), M_centered.max()
        M_norm = (M_centered - M_min) / (M_max - M_min + 1e-12)
        M_clahe = exposure.equalize_adapthist(M_norm, clip_limit=0.015)
        return M_clahe

    M_final = enhance_and_smooth(p95_map, sigma=0.2)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")

    im = ax.imshow(M_final, origin="lower", cmap=custom_cmap, vmin=0, vmax=1)

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Normalized Value")

    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "u_struct_p95_custom_style.png"), dpi=600)
    plt.close(fig)

    np.save(os.path.join(save_dir, "data_struct_p95_final.npy"), M_final)

    print(f"\n[Done] 图像已优化。平滑后的结构图和自定义色标已保存至：{save_dir}")


def start_interactive_exploration(model, bundle, device, model_save_dir="./outputs"):
    """
    增强版交互式评估工具：
    1. 全局字体 24（包含 Color bar）。
    2. 子图 6 为 3D 线框图 (Wireframe)。
    3. X + 点击导航栏：批量保存 6 个子图。
    4. 点击子图 1：更新截面；Ctrl + 点击：保存截面图。
    """
    plt.rcParams.update(
        {
            "font.size": 24,
            "axes.titlesize": 24,
            "axes.labelsize": 24,
            "xtick.labelsize": 20,
            "ytick.labelsize": 20,
            "font.family": "serif",
            "font.serif": ["Times New Roman"],
            "axes.unicode_minus": False,
        }
    )

    U_full_cpu = bundle["U"].cpu().numpy()
    t_grid = bundle["t_grid"].cpu().numpy() * 1e6
    Nt, R, C = bundle["Nt"], bundle["R"], bundle["C"]

    state = {
        "x_pressed": False,
        "ctrl_pressed": False,
        "mid_row": 49,
        "mid_col": 55,
        "current_idx": 0,
        "save_dir": None,
    }

    v_max_global = np.percentile(np.abs(U_full_cpu), 99.5)
    v_min_global = -v_max_global
    amplitude_curve = np.mean(np.abs(U_full_cpu), axis=0)

    fig_nav, ax_nav = plt.subplots(figsize=(12, 4))
    ax_nav.plot(t_grid, amplitude_curve, "b-", linewidth=2)
    ax_nav.set_title("Time Selector (X+Click to Save All)", fontsize=24)
    line_indicator = ax_nav.axvline(x=t_grid[0], color="r", linestyle="--", linewidth=3)

    fig_vis = plt.figure(figsize=(22, 12))

    def get_save_path():
        if state["save_dir"] is None:
            now = datetime.now().strftime("%Y%m%d_%H%M%S")
            state["save_dir"] = os.path.join(model_save_dir, f"capture_{now}")
            os.makedirs(state["save_dir"], exist_ok=True)
        return state["save_dir"]

    def save_ax_as_image(ax, name):
        path = get_save_path()
        extent = ax.get_tightbbox(fig_vis.canvas.get_renderer()).transformed(fig_vis.dpi_scale_trans.inverted())
        fig_vis.savefig(os.path.join(path, f"{name}.png"), bbox_inches=extent, dpi=300)

    def update_vis():
        idx = state["current_idx"]
        t_us = t_grid[idx]

        u_hat, u_prop, u_struct = predict_time_slice_components(model, bundle, idx, device=device)
        u_gt = make_matrix_from_vector(U_full_cpu[:, idx], R, C, serpentine=SERPENTINE_VIS)

        fig_vis.clear()
        extent = [0, C - 1, 0, R - 1]

        ax1 = fig_vis.add_subplot(2, 3, 1)
        im1 = ax1.imshow(u_gt, cmap="viridis", vmin=v_min_global, vmax=v_max_global, origin="lower", extent=extent)
        ax1.set_title("Raw Wavefield")
        ax1.set_xticks([])
        ax1.set_yticks([])
        ax1.axhline(state["mid_row"], color="white", linestyle=":", alpha=0.6)
        ax1.axvline(state["mid_col"], color="white", linestyle=":", alpha=0.6)
        cb1 = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
        cb1.ax.tick_params(labelsize=20)

        ax2 = fig_vis.add_subplot(2, 3, 2)
        ax2.plot(u_gt[state["mid_row"], :], "k-", alpha=0.3, label="Raw")
        ax2.plot(u_prop[state["mid_row"], :], "r--", linewidth=2, label="Prop")
        ax2.set_title(f"X-Section (Row {state['mid_row']})")
        ax2.set_ylim(v_min_global * 1.2, v_max_global * 1.2)
        ax2.set_box_aspect(1)
        ax2.grid(True, linestyle=":", alpha=0.5)

        ax3 = fig_vis.add_subplot(2, 3, 3)
        ax3.plot(u_gt[:, state["mid_col"]], "k-", alpha=0.3)
        ax3.plot(u_prop[:, state["mid_col"]], "r--", linewidth=2)
        ax3.set_title(f"Y-Section (Col {state['mid_col']})")
        ax3.set_ylim(v_min_global * 1.2, v_max_global * 1.2)
        ax3.set_box_aspect(1)
        ax3.grid(True, linestyle=":", alpha=0.5)

        ax4 = fig_vis.add_subplot(2, 3, 4)
        v_s = np.percentile(np.abs(u_struct), 99.5)
        im4 = ax4.imshow(u_struct, cmap="coolwarm", origin="lower", extent=extent, vmin=-v_s, vmax=v_s)
        ax4.set_title("Structural Field")
        ax4.set_xticks([])
        ax4.set_yticks([])
        cb4 = plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)
        cb4.ax.tick_params(labelsize=20)

        ax5 = fig_vis.add_subplot(2, 3, 5)
        im5 = ax5.imshow(u_prop, cmap="viridis", vmin=v_min_global, vmax=v_max_global, origin="lower", extent=extent)
        ax5.set_title("Separated Propagation")
        ax5.set_xticks([])
        ax5.set_yticks([])
        cb5 = plt.colorbar(im5, ax=ax5, fraction=0.046, pad=0.04)
        cb5.ax.tick_params(labelsize=20)

        ax6 = fig_vis.add_subplot(2, 3, 6, projection="3d")
        X_m, Y_m = np.meshgrid(np.arange(C), np.arange(R))
        ax6.plot_wireframe(X_m, Y_m, u_prop, rstride=4, cstride=4, color="red", linewidth=0.8, alpha=0.5)
        ax6.set_zlim(v_min_global * 1.5, v_max_global * 1.5)
        ax6.set_title("3D Propagation (Wireframe)")
        ax6.view_init(elev=35, azim=-45)

        fig_vis.suptitle(f"Time: {t_us:.2f} μs | Frame: {idx}", fontsize=28, fontweight="bold")
        fig_vis.tight_layout()
        fig_vis.canvas.draw()

        if state["x_pressed"]:
            print(">>> 正在批量导出子图...")
            save_ax_as_image(ax1, f"T{idx}_1_Raw")
            save_ax_as_image(ax2, f"T{idx}_2_XSec")
            save_ax_as_image(ax3, f"T{idx}_3_YSec")
            save_ax_as_image(ax4, f"T{idx}_4_Struct")
            save_ax_as_image(ax5, f"T{idx}_5_Prop")
            save_ax_as_image(ax6, f"T{idx}_6_3DWire")
            state["x_pressed"] = False

    def on_key_press(event):
        if event.key == "x":
            state["x_pressed"] = True
        if event.key == "control":
            state["ctrl_pressed"] = True

    def on_key_release(event):
        if event.key == "x":
            state["x_pressed"] = False
        if event.key == "control":
            state["ctrl_pressed"] = False

    def on_click_nav(event):
        if event.inaxes != ax_nav:
            return
        state["current_idx"] = (np.abs(t_grid - event.xdata)).argmin()
        line_indicator.set_xdata([t_grid[state["current_idx"]]])
        fig_nav.canvas.draw()
        update_vis()

    def on_click_vis(event):
        if event.inaxes is None:
            return
        spec = event.inaxes.get_subplotspec()
        if spec.rowspan.start == 0 and spec.colspan.start == 0:
            state["mid_col"] = int(round(event.xdata))
            state["mid_row"] = int(round(event.ydata))

            if state["ctrl_pressed"]:
                update_vis()
                axes = fig_vis.get_axes()
                save_ax_as_image(axes[2], f"T{state['current_idx']}_XSection_R{state['mid_row']}")
                save_ax_as_image(axes[3], f"T{state['current_idx']}_YSection_C{state['mid_col']}")
            else:
                update_vis()

    fig_nav.canvas.mpl_connect("button_press_event", on_click_nav)
    fig_nav.canvas.mpl_connect("key_press_event", on_key_press)
    fig_nav.canvas.mpl_connect("key_release_event", on_key_release)

    fig_vis.canvas.mpl_connect("button_press_event", on_click_vis)
    fig_vis.canvas.mpl_connect("key_press_event", on_key_press)
    fig_vis.canvas.mpl_connect("key_release_event", on_key_release)

    update_vis()
    plt.show()


# ===========================
# 训练/评估入口
# ===========================

def run_training_mode():
    print(">>> 进入【训练模式】...")
    p80 = pick_pkl("选择 PKL 数据文件（*_numxnum.pkl）")
    if p80 is None:
        print("未选择文件，退出。")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    bundle = prepare_dataset_from_pkl(
        p80,
        sampling_rate_hz=FS,
        crop_cols=CROP,
        filter_mode=FILTER_MODE,
        f_low=FREQ_LOW,
        f_high=FREQ_HIGH,
        filter_order=FILTER_ORDER,
    )

    R, C = bundle["R"], bundle["C"]
    ROI_X0, ROI_X1 = 0, C
    ROI_Y0, ROI_Y1 = 0, R
    print(f"已设置全图 ROI: X[0:{C}], Y[0:{R}]")

    out_dir = build_output_dir(
        pkl_path=p80,
        model_tag="B-PINN-hetero-NO_PDE+FREQ_PRIOR",
        roi=(ROI_X0, ROI_X1, ROI_Y0, ROI_Y1),
        train_ratio=TRAIN_RATIO,
        epochs=EPOCHS,
    )
    os.makedirs(out_dir, exist_ok=True)
    print(f"输出目录: {out_dir}")

    print(">>> [FreqPrior] 正在读取全时域数据用于频域结构先验 ...")
    bundle_full = prepare_dataset_from_pkl(
        p80,
        sampling_rate_hz=FS,
        crop_cols=None,
        filter_mode="none",
        f_low=None,
        f_high=None,
        filter_order=FILTER_ORDER,
    )

    F_db_vec = save_F_struct_db_map_and_sensitivity(
        bundle_full,
        out_dir=out_dir,
        f_band_main=FREQ_BAND_HZ,
        use_hann=FREQ_USE_HANN,
        ref_mode=FREQ_DB_REF_MODE,
        freq_db_eps=FREQ_DB_EPS,
        serpentine_vis=SERPENTINE_VIS,
    )

    F_db_t = torch.tensor(F_db_vec, dtype=torch.float32)
    F_feat = (F_db_t - F_db_t.mean()) / (F_db_t.std() + 1e-12)
    bundle["F_struct_db_vec"] = F_db_t
    bundle["F_feat_vec"] = F_feat

    roi_idx_all = make_roi_indices(bundle, ROI_X0, ROI_X1, ROI_Y0, ROI_Y1)
    rng = np.random.default_rng(RAND_SEED)
    roi_idx_all = np.array(roi_idx_all)
    rng.shuffle(roi_idx_all)
    n_train = int(round(TRAIN_RATIO * len(roi_idx_all)))
    train_idx = np.sort(roi_idx_all[:n_train]).tolist()
    held_idx = np.sort(roi_idx_all[n_train:]).tolist()

    model, df_history = train_b_pinn(
        bundle,
        train_idx,
        epochs=EPOCHS,
        w_data=w_data,
        batch_spatial=1024,
        batch_time=bundle["Nt"],
        lr=1e-3,
        device=device,
        dropout_p=DROPOUT_P,
        eval_interval=EVAL_INTERVAL,
        eval_bundle=bundle,
        eval_train_idx=train_idx,
        eval_held_idx=held_idx,
        eval_out_base_dir=out_dir,
        eval_roi=(ROI_X0, ROI_X1, ROI_Y0, ROI_Y1),
    )

    model_ckpt_path = os.path.join(out_dir, "b_pinn_hetero_model_ckpt_NO_PDE_FREQ_PRIOR.pt")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": {
                "FS": FS,
                "CROP": CROP,
                "TRAIN_RATIO": TRAIN_RATIO,
                "ROI_X0": ROI_X0,
                "ROI_X1": ROI_X1,
                "ROI_Y0": ROI_Y0,
                "ROI_Y1": ROI_Y1,
                "EPOCHS": EPOCHS,
                "DROPOUT_P": DROPOUT_P,
                "MC_SAMPLES": MC_SAMPLES,
                "FILTER_MODE": FILTER_MODE,
                "FREQ_BAND_HZ": FREQ_BAND_HZ,
                "STRUCT_STAT_MODE": STRUCT_STAT_MODE,
                "LAMBDA_FREQ_MSE": LAMBDA_FREQ_MSE,
                "LAMBDA_FREQ_CORR": LAMBDA_FREQ_CORR,
            },
        },
        model_ckpt_path,
    )
    print(f"模型已保存到: {model_ckpt_path}")

    save_training_loss_curves(df_history, out_dir)

    print("--- 开始进行最终评估 ---")
    final_eval_dir = os.path.join(out_dir, "final_evaluation")
    os.makedirs(final_eval_dir, exist_ok=True)

    save_combined_evaluation_gif(
        model,
        bundle,
        device,
        final_eval_dir,
        ROI_X0=ROI_X0,
        ROI_X1=ROI_X1,
        ROI_Y0=ROI_Y0,
        ROI_Y1=ROI_Y1,
        fps=10,
    )

    save_struct_maps(
        model,
        bundle,
        final_eval_dir,
        ROI_X0=ROI_X0,
        ROI_X1=ROI_X1,
        ROI_Y0=ROI_Y0,
        ROI_Y1=ROI_Y1,
        t_show_idx=bundle["Nt"] // 2,
    )

    print("--- 训练及最终评估全部完成（含频域结构先验） ---")


def run_evaluation_mode():
    print(">>> 进入【载入模型评估模式】...")

    pkl_path = pick_pkl("【步骤1】选择原始 PKL 数据文件 (用于坐标对齐)")
    if not pkl_path:
        return

    model_path = pick_pkl("【步骤2】选择已训练的模型 (b_pinn_...pt)")
    if not model_path:
        return

    model_dir = os.path.dirname(model_path)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    eval_out_dir = os.path.join(model_dir, f"ReEval_Output_FREQ_PRIOR_{ts}")
    os.makedirs(eval_out_dir, exist_ok=True)
    print(f"评估结果将保存至: {eval_out_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.zeros(1).to(device)

    print(f"正在读取数据: {os.path.basename(pkl_path)} ...")
    bundle = prepare_dataset_from_pkl(
        pkl_path,
        sampling_rate_hz=FS,
        crop_cols=CROP,
        filter_mode=FILTER_MODE,
        f_low=FREQ_LOW,
        f_high=FREQ_HIGH,
        filter_order=FILTER_ORDER,
    )

    bundle_full = prepare_dataset_from_pkl(
        pkl_path,
        sampling_rate_hz=FS,
        crop_cols=None,
        filter_mode="none",
        f_low=None,
        f_high=None,
        filter_order=FILTER_ORDER,
    )
    F_db_vec = save_F_struct_db_map_and_sensitivity(
        bundle_full,
        out_dir=eval_out_dir,
        f_band_main=FREQ_BAND_HZ,
        use_hann=FREQ_USE_HANN,
        ref_mode=FREQ_DB_REF_MODE,
        freq_db_eps=FREQ_DB_EPS,
        serpentine_vis=SERPENTINE_VIS,
    )

    F_db_t = torch.tensor(F_db_vec, dtype=torch.float32)
    F_feat = (F_db_t - F_db_t.mean()) / (F_db_t.std() + 1e-12)
    bundle["F_struct_db_vec"] = F_db_t
    bundle["F_feat_vec"] = F_feat
    R, C = bundle["R"], bundle["C"]
    ROI_X0, ROI_X1 = 0, C
    ROI_Y0, ROI_Y1 = 0, R
    print(f"已设置全图 ROI: X[0:{C}], Y[0:{R}]")

    print(f"正在加载模型: {os.path.basename(model_path)} ...")
    try:
        checkpoint = torch.load(model_path, map_location=device)

        if "config" in checkpoint:
            saved_dropout = checkpoint["config"].get("DROPOUT_P", DROPOUT_P)
            print(f"从 checkpoint 读取到 Dropout_P = {saved_dropout}")
        else:
            saved_dropout = DROPOUT_P

        model = BPINN2DWave(p_drop=saved_dropout).to(device)

        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)

        model.eval()
        print(f"模型加载成功。学习到的 c = {model.c().item():.4f}")

    except Exception as exc:
        print(f"模型加载失败: {exc}")
        return

    print("正在恢复 ROI 索引 (基于当前配置的随机种子)...")
    roi_idx_all = make_roi_indices(bundle, ROI_X0, ROI_X1, ROI_Y0, ROI_Y1)
    rng = np.random.default_rng(RAND_SEED)
    roi_idx_all = np.array(roi_idx_all)
    rng.shuffle(roi_idx_all)
    n_train = int(round(TRAIN_RATIO * len(roi_idx_all)))
    train_idx = np.sort(roi_idx_all[:n_train]).tolist()
    held_idx = np.sort(roi_idx_all[n_train:]).tolist()

    save_struct_maps(
        model,
        bundle,
        eval_out_dir,
        ROI_X0=ROI_X0,
        ROI_X1=ROI_X1,
        ROI_Y0=ROI_Y0,
        ROI_Y1=ROI_Y1,
        t_show_idx=bundle["Nt"] // 2,
    )

    print(">>> [1/5] 正在执行标准评估 (贝叶斯四联图 + 指标CSV)...")
    ans = input("\n是否启动交互式 3D 波场查看器? (y/n) [y]: ")
    if ans.lower() != "n":
        start_interactive_exploration(model, bundle, device, model_save_dir=eval_out_dir)

    print(">>> [Extra] 正在生成四合一动态 GIF ...")
    save_combined_evaluation_gif(
        model,
        bundle,
        device,
        eval_out_dir,
        ROI_X0=ROI_X0,
        ROI_X1=ROI_X1,
        ROI_Y0=ROI_Y0,
        ROI_Y1=ROI_Y1,
        fps=10,
    )

    print(f"\n所有重评估结果已保存至: {eval_out_dir}")
