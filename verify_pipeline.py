"""
Paper Fig. 11 doğrulama betiği.

Çıktı: my_fig11.png — paper Fig. 11(a)/(b) ile doğrudan karşılaştırma
  - Kırmızı (Synthetic): σ_P=0 deterministik, R0 sabit (HELI=2.1 m, QUAD=1.8 m)
  - Mavi (Augmented): σ_P=0.20 ile 4 örnek (paper'daki ölçüm yerine)
  - X-ekseni: metre (R = c0·f/(2μ))

Ek çıktı: my_spectrum_log.png — log-ölçek ortalama spektrum + beklenen frekans tepe konumları
"""
import numpy as np
import matplotlib.pyplot as plt

from main import (
    HELI_REF_PARAMS, QUAD_REF_PARAMS,
    generate_single_profile, generate_profile_mti,
    range_axis_around, R0_HELI, R0_QUAD,
    FS, N_FFT, BIN_HZ, PROFILE_LEN,
)

N_AUGMENTED = 4
SEED = 0


def fig11_panel(ax, params, R0, title, xlim):
    """Paper Fig. 11 stilinde tek panel: σ_P=0 kırmızı + σ_P=0.20 mavi."""
    rng = np.random.default_rng(SEED)
    x_m = range_axis_around(R0)

    # Mavi: σ_P=0.20 augmented (paper'daki gerçek ölçümlerin sentetik karşılığı)
    for i in range(N_AUGMENTED):
        prof = generate_single_profile(params, sigma_p=0.20, R0=R0, rng=rng)
        label = f"Augmented #{i+1:02d}" if i < N_AUGMENTED else None
        ax.plot(x_m, prof, color="steelblue", alpha=0.6, linewidth=1, label=label)

    # Kırmızı: σ_P=0 deterministik (Table II birebir parametreler)
    rng_clean = np.random.default_rng(SEED)
    clean = generate_single_profile(params, sigma_p=0.0, R0=R0, rng=rng_clean)
    ax.plot(x_m, clean, color="red", linewidth=2.0, label="Synthetic (σ_P=0)")

    ax.set_title(title)
    ax.set_xlabel("Range (m)")
    ax.set_ylabel("Normalized Units")
    ax.set_xlim(xlim)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)


def expected_freq_peaks(params_dict):
    """Table II'den beklenen baskın vibrasyon frekansları (Hz)."""
    peaks = []
    for name, p in params_dict.items():
        if p.get("Gamma", 0) == 0:
            continue
        w_a, w_b = p["omega_alpha"], p["omega_beta"]
        peaks.append((name, "α", w_a / (2 * np.pi)))
        peaks.append((name, "β", w_b / (2 * np.pi)))
        peaks.append((name, "α+β", p["C_omega_plus"] * abs(w_a + w_b) / (2 * np.pi)))
        peaks.append((name, "α-β", p["C_omega_minus"] * abs(w_a - w_b) / (2 * np.pi)))
    return peaks


def main():
    # === Fig. 11 karşılaştırma ===
    fig, axes = plt.subplots(2, 1, figsize=(9, 8))
    fig11_panel(axes[0], HELI_REF_PARAMS, R0_HELI,
                "(a) Single-engine UAV, RadioFLY helicopter — paper Fig. 11(a)",
                xlim=(1.1, 2.95))
    fig11_panel(axes[1], QUAD_REF_PARAMS, R0_QUAD,
                "(b) Quadcopter UAV, DJI Drone Mavic Mini — paper Fig. 11(b)",
                xlim=(0.85, 2.75))
    plt.tight_layout()
    plt.savefig("my_fig11.png", dpi=140)
    print("✅ my_fig11.png — paper_fig11.png ile yan yana karşılaştır")

    # === Beklenen frekans tepeleri (Table II) ===
    print(f"\n--- Beklenen vibrasyon frekansları ---")
    print(f"FS={FS:.0f} Hz, N_FFT={N_FFT}, bin = {BIN_HZ:.2f} Hz\n")
    for cls_name, params in [("HELI", HELI_REF_PARAMS), ("QUAD", QUAD_REF_PARAMS)]:
        print(f"[{cls_name}]")
        for engine, comp, freq_hz in expected_freq_peaks(params):
            print(f"  {engine:10s} {comp:5s}: {freq_hz:7.1f} Hz  (bin {freq_hz/BIN_HZ:6.1f})")
        print()

    # === Log spektrum ===
    rng = np.random.default_rng(SEED)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    for ax, params, R0, name in [
        (axes[0], HELI_REF_PARAMS, R0_HELI, "HELI"),
        (axes[1], QUAD_REF_PARAMS, R0_QUAD, "QUAD"),
    ]:
        clean = generate_single_profile(params, sigma_p=0.0, R0=R0, rng=rng)
        x_m = range_axis_around(R0)
        ax.semilogy(x_m, clean + 1e-6)
        ax.set_title(f"{name} σ_P=0 (log) — R0={R0} m")
        ax.set_xlabel("Range (m)")
        ax.set_ylabel("Normalized (log)")
        ax.grid(True, alpha=0.3, which="both")
    plt.tight_layout()
    plt.savefig("my_spectrum_log.png", dpi=140)
    print("✅ my_spectrum_log.png")

    # === MTI'lı çıktı (Fig. 6 ölçüm zinciri) ek doğrulama ===
    rng = np.random.default_rng(SEED)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    for ax, params, R0, name in [
        (axes[0], HELI_REF_PARAMS, R0_HELI, "HELI"),
        (axes[1], QUAD_REF_PARAMS, R0_QUAD, "QUAD"),
    ]:
        no_mti = generate_single_profile(params, sigma_p=0.0, R0=R0, rng=np.random.default_rng(SEED))
        mti = generate_profile_mti(params, sigma_p=0.0, R0=R0, rng=np.random.default_rng(SEED))
        x_m = range_axis_around(R0)
        ax.plot(x_m, no_mti, color="steelblue", linewidth=1.2, label="without MTI (Fig. 2)")
        ax.plot(x_m, mti, color="red", linewidth=1.5, label="with MTI 1:10 (Fig. 6)")
        ax.set_title(f"{name} — paper Fig. 10(e/f) tarzı MTI karşılaştırma")
        ax.set_xlabel("Range (m)")
        ax.set_ylabel("Normalized Units")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("my_fig10.png", dpi=140)
    print("✅ my_fig10.png — paper Fig. 10(e/f) MTI vs raw")


if __name__ == "__main__":
    main()
