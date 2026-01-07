import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _fft_rfft_mag(U_np, fs, use_hann=True):
    """
    U_np: (Ns, Nt) float32
    return freqs (Nf,), mag (Ns, Nf)
    """
    Ns, Nt = U_np.shape
    if use_hann:
        w = np.hanning(Nt).astype(np.float32)[None, :]
        X = np.fft.rfft(U_np * w, axis=1)
    else:
        X = np.fft.rfft(U_np, axis=1)
    mag = np.abs(X).astype(np.float32)
    freqs = np.fft.rfftfreq(Nt, d=1.0 / fs).astype(np.float32)
    return freqs, mag


def compute_F_struct_db_from_fulltime(
    bundle_full,
    mode="band",
    f0_hz=0.01e6,
    f_band_hz=(0.01e6, 0.1e6),
    use_hann=True,
    ref_mode="p95",
    eps=1e-12,
):
    """
    返回：
      F_db_vec: (Ns,) —— 与原始蛇形索引一致，可直接用 si 索引
      freqs: (Nf,)
      meta: dict
    """
    U_full = bundle_full["U"].cpu().numpy().astype(np.float32)
    fs = float(bundle_full["fs"])
    freqs, mag = _fft_rfft_mag(U_full, fs, use_hann=use_hann)

    if mode == "single":
        idx = int(np.argmin(np.abs(freqs - float(f0_hz))))
        F_lin = mag[:, idx]
        meta = {"mode": "single", "f0_hz": float(freqs[idx]), "idx": idx}
    else:
        fL, fH = float(f_band_hz[0]), float(f_band_hz[1])
        band_mask = (freqs >= fL) & (freqs <= fH)
        if np.sum(band_mask) < 1:
            idx = int(np.argmin(np.abs(freqs - float(f0_hz))))
            F_lin = mag[:, idx]
            meta = {
                "mode": "band_fallback_to_single",
                "f0_hz": float(freqs[idx]),
                "idx": idx,
                "f_band_hz": (fL, fH),
            }
        else:
            F_lin = np.mean(mag[:, band_mask], axis=1)
            meta = {"mode": "band", "f_band_hz": (fL, fH), "n_bins": int(np.sum(band_mask))}

    if ref_mode == "max":
        ref = float(np.max(F_lin))
    elif ref_mode == "median":
        ref = float(np.median(F_lin))
    else:
        ref = float(np.percentile(F_lin, 95))

    ref = max(ref, eps)
    F_db_vec = 20.0 * np.log10(np.maximum(F_lin, eps) / ref).astype(np.float32)
    return F_db_vec, freqs, meta


def save_F_struct_db_map_and_sensitivity(
    bundle_full,
    out_dir,
    f_band_main=(0.01e6, 0.1e6),
    use_hann=True,
    ref_mode="p95",
    freq_db_eps=1e-12,
    serpentine_vis=True,
):
    """
    1) 保存参考频带的 F_struct(dB) 图
    2) 频带敏感性：扫多个子频带，与参考频带图做相关性曲线（1张图）
    """
    save_dir = os.path.join(out_dir, "freq_prior")
    os.makedirs(save_dir, exist_ok=True)

    R, C = bundle_full["R"], bundle_full["C"]

    F_ref_db_vec, freqs, meta = compute_F_struct_db_from_fulltime(
        bundle_full,
        mode="band",
        f_band_hz=f_band_main,
        use_hann=use_hann,
        ref_mode=ref_mode,
        eps=freq_db_eps,
    )

    M_ref = F_ref_db_vec.reshape(R, C).copy()
    if serpentine_vis:
        M_ref[1::2, :] = M_ref[1::2, ::-1]

    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    v = np.percentile(np.abs(M_ref), 99)
    im = ax.imshow(M_ref, origin="lower", cmap="coolwarm", vmin=-v, vmax=v)
    ax.set_title(f"F_struct (dB), band={f_band_main[0]/1e6:.3f}-{f_band_main[1]/1e6:.3f} MHz")
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="dB (ref=95th)")
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "F_struct_db_refband.png"), dpi=250)
    plt.close(fig)

    f_start, f_end = 0.01e6, 0.1e6
    centers = np.linspace(f_start, f_end, 12).astype(np.float32)
    half_bw = 0.005e6

    ref_flat = M_ref.reshape(-1)
    ref_flat = ref_flat - np.mean(ref_flat)
    ref_flat = ref_flat / (np.std(ref_flat) + 1e-12)

    corrs = []
    bands = []
    for fc in centers:
        band = (float(max(f_start, fc - half_bw)), float(min(f_end, fc + half_bw)))
        F_db_vec, _, _ = compute_F_struct_db_from_fulltime(
            bundle_full, mode="band", f_band_hz=band, use_hann=use_hann, ref_mode=ref_mode, eps=freq_db_eps
        )
        M = F_db_vec.reshape(R, C).copy()
        if serpentine_vis:
            M[1::2, :] = M[1::2, ::-1]
        x = M.reshape(-1)
        x = x - np.mean(x)
        x = x / (np.std(x) + 1e-12)
        corr = float(np.mean(ref_flat * x))
        corrs.append(corr)
        bands.append(band)

    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    ax.plot(centers / 1e6, corrs, marker="o")
    ax.set_xlabel("Center frequency (MHz)")
    ax.set_ylabel("Corr with reference band map")
    ax.set_title("Band selection sensitivity (corr vs. reference 0.01–0.1 MHz)")
    ax.grid(True, alpha=0.4)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "band_sensitivity.png"), dpi=250)
    plt.close(fig)

    df = pd.DataFrame(
        {
            "fc_MHz": centers / 1e6,
            "band_low_MHz": [b[0] / 1e6 for b in bands],
            "band_high_MHz": [b[1] / 1e6 for b in bands],
            "corr_with_ref": corrs,
        }
    )
    df.to_csv(os.path.join(save_dir, "band_sensitivity.csv"), index=False, float_format="%.6g")

    np.save(os.path.join(save_dir, "F_struct_db_refband_vec.npy"), F_ref_db_vec)
    return F_ref_db_vec
