import math
import os
import re
from datetime import datetime
import tkinter as tk
from tkinter import filedialog

import numpy as np
import pandas as pd
import torch
from scipy import signal


def pick_pkl(title="选择Pickle文件"):
    root = tk.Tk()
    root.withdraw()
    fp = filedialog.askopenfilename(title=title, filetypes=[("Pickle文件", "*.pkl")])
    return fp if fp else None


def serp_idx_to_rc(i, ncols):
    r = i // ncols
    k = i % ncols
    c = k if (r % 2 == 0) else (ncols - 1 - k)
    return r, c


def rc_to_serp_idx(r, c, ncols):
    return r * ncols + (c if r % 2 == 0 else (ncols - 1 - c))


def _apply_filter_to_raw_array(U_np, fs, mode="none", f_low=None, f_high=None, order=4):
    if mode is None:
        mode = "none"
    mode = mode.lower()

    if mode == "none":
        return U_np

    nyq = 0.5 * float(fs)

    try:
        if mode == "lowpass":
            if f_high is None or f_high <= 0 or f_high >= nyq:
                print("[WARN] 低通滤波: FREQ_HIGH 非法或未设置，跳过滤波。")
                return U_np
            wn = float(f_high) / nyq
            b, a = signal.butter(order, wn, btype="lowpass")

        elif mode == "highpass":
            if f_low is None or f_low <= 0 or f_low >= nyq:
                print("[WARN] 高通滤波: FREQ_LOW 非法或未设置，跳过滤波。")
                return U_np
            wn = float(f_low) / nyq
            b, a = signal.butter(order, wn, btype="highpass")

        elif mode == "bandpass":
            if (
                f_low is None
                or f_high is None
                or f_low <= 0
                or f_low >= nyq
                or f_high <= 0
                or f_high >= nyq
                or f_low >= f_high
            ):
                print("[WARN] 带通滤波: FREQ_LOW/FREQ_HIGH 非法或未设置，跳过滤波。")
                return U_np
            wn = [float(f_low) / nyq, float(f_high) / nyq]
            b, a = signal.butter(order, wn, btype="bandpass")

        else:
            print(f"[WARN] 未知 FILTER_MODE={mode}，跳过滤波。")
            return U_np

        U_filt = signal.filtfilt(b, a, U_np, axis=1)
        return U_filt.astype(np.float32)

    except Exception as exc:
        print(f"[WARN] 滤波时发生异常：{exc}，跳过滤波。")
        return U_np


def prepare_dataset_from_pkl(
    pkl_path,
    sampling_rate_hz,
    crop_cols=None,
    filter_mode="none",
    f_low=None,
    f_high=None,
    filter_order=4,
):
    df = pd.read_pickle(pkl_path)

    if crop_cols is not None:
        s, e = crop_cols
        df = df.iloc[:, s:e].copy()

    U_np = df.values.astype(np.float32)

    U_np = _apply_filter_to_raw_array(
        U_np,
        fs=sampling_rate_hz,
        mode=filter_mode,
        f_low=f_low,
        f_high=f_high,
        order=filter_order,
    )

    U = torch.tensor(U_np, dtype=torch.float32)

    Ns, Nt = U.shape
    fs = float(sampling_rate_hz)
    dt = 1.0 / fs
    t_grid = torch.linspace(0.0, (Nt - 1) * dt, Nt)

    m = re.search(r"_(\d+)x(\d+)\.pkl$", os.path.basename(pkl_path))
    if m:
        R, C = int(m.group(1)), int(m.group(2))
        assert R * C == Ns
    else:
        n = int(round(math.sqrt(Ns)))
        assert n * n == Ns
        R = C = n

    r_list, c_list = [], []
    for i in range(Ns):
        r, c = serp_idx_to_rc(i, C)
        r_list.append(r)
        c_list.append(c)
    r_orig = torch.tensor(r_list, dtype=torch.float32)
    c_orig = torch.tensor(c_list, dtype=torch.float32)

    x_norm = (c_orig / (C - 1)) * 2.0 - 1.0
    y_norm = (r_orig / (R - 1)) * 2.0 - 1.0

    scale = torch.max(torch.abs(U))
    U = U / (scale + 1e-12)

    t_norm = (t_grid / (t_grid[-1] + 1e-12)) * 2.0 - 1.0

    with torch.no_grad():
        E = torch.mean(torch.abs(U), dim=0) + 1e-6
        E = E / torch.sum(E)

    return {
        "U": U,
        "Ns": Ns,
        "Nt": Nt,
        "dt": dt,
        "x_norm": x_norm,
        "y_norm": y_norm,
        "t_norm": t_norm,
        "R": R,
        "C": C,
        "t_grid": t_grid,
        "r_orig": r_orig,
        "c_orig": c_orig,
        "scale": scale,
        "E_t": E,
        "fs": fs,
    }


def build_output_dir(pkl_path, model_tag="PINN", roi=(0, 80, 50, 80), train_ratio=0.7, epochs=10000):
    base_dir = os.path.dirname(pkl_path)
    file_stem = os.path.splitext(os.path.basename(pkl_path))[0]
    x0, x1, y0, y1 = roi
    roi_tag = f"ROIy{y0}-{y1}_x{x0}-{x1}"
    run_tag = f"{model_tag}_{roi_tag}_TR{train_ratio:.2f}_ep{epochs}"
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = os.path.join(base_dir, f"{file_stem}__{run_tag}__{ts}")
    return out_dir


def make_roi_indices(bundle, x0, x1, y0, y1):
    r = bundle["r_orig"].long()
    c = bundle["c_orig"].long()
    mask = (c >= x0) & (c < x1) & (r >= y0) & (r < y1)
    idx = torch.where(mask)[0].cpu().numpy().tolist()
    return idx


def make_matrix_from_vector(vec, R, C, serpentine=True):
    M = vec.reshape(R, C).copy()
    if serpentine:
        M[1::2, :] = M[1::2, ::-1]
    return M
