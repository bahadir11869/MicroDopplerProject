"""
Pipeline doğrulama betiği:
  1) Görsel: HELI ve QUAD için 4 ölçüm + sentetik ortalama (paper Fig. 11'e paralel)
  2) İstatistiksel: beklenen ω_α, ω_β frekanslarının FFT'de görünüp görünmediği
"""
import numpy as np
import matplotlib.pyplot as plt

from main import (
    HELI_REF_PARAMS, QUAD_REF_PARAMS,
    generate_single_profile, FS, N_SAMPLES, N_FFT,
)

N_EXAMPLES = 4
SEED = 0


def sample_profiles(params, n, rng):
    return np.stack([generate_single_profile(params, rng=rng) for _ in range(n)])


def plot_class(ax, profiles, title, color):
    for i, p in enumerate(profiles):
        ax.plot(p, color="steelblue", alpha=0.6,
                label=f"Measurement #{i+1:02d}" if i < N_EXAMPLES else None)
    ax.plot(profiles.mean(axis=0), color=color, linewidth=2, label="Synthetic (mean)")
    ax.set_title(title)
    ax.set_xlabel("Range profile index (0-400)")
    ax.set_ylabel("Normalized amplitude")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)
    ax.set_xlim(0, 400)


def expected_fft_peaks(params_dict):
    """Table II'den beklenen baskın frekansları (FFT bin olarak) döndürür."""
    peaks = []
    bin_hz = FS / N_FFT  # Hz per bin
    for name, p in params_dict.items():
        if p.get("Gamma", 0) == 0:
            continue
        # α, β, ve mix± frekansları (rad/s → Hz → bin)
        w_a, w_b = p["omega_alpha"], p["omega_beta"]
        freqs_hz = [w_a / (2 * np.pi), w_b / (2 * np.pi),
                    p["C_omega_plus"] * abs(w_a + w_b) / (2 * np.pi),
                    p["C_omega_minus"] * abs(w_a - w_b) / (2 * np.pi)]
        for f in freqs_hz:
            peaks.append((name, f, f / bin_hz))
    return peaks


def main():
    rng = np.random.default_rng(SEED)

    heli = sample_profiles(HELI_REF_PARAMS, N_EXAMPLES, rng)
    quad = sample_profiles(QUAD_REF_PARAMS, N_EXAMPLES, rng)

    # --- Görsel karşılaştırma ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7))
    plot_class(ax1, heli, "HELI (Single-engine, RadioFLY) — paper Fig. 11(a)", "red")
    plot_class(ax2, quad, "QUAD (DJI Mavic Mini) — paper Fig. 11(b)", "red")
    plt.tight_layout()
    plt.savefig("my_fig11.png", dpi=130)
    print("✅ my_fig11.png kaydedildi (paper Fig. 11 ile kıyasla)")

    # --- İstatistiksel kontrol ---
    print("\n--- Beklenen FFT tepe bin'leri (Table II'den) ---")
    print(f"FS={FS:.0f} Hz, N_FFT={N_FFT}, bin çözünürlüğü = {FS/N_FFT:.2f} Hz\n")

    for cls_name, params in [("HELI", HELI_REF_PARAMS), ("QUAD", QUAD_REF_PARAMS)]:
        print(f"[{cls_name}]")
        for engine, freq_hz, bin_ in expected_fft_peaks(params):
            print(f"  {engine:10s}: {freq_hz:7.1f} Hz  → bin {bin_:6.1f}")
        print()

    # --- Ortalama spektrumda tepeler görünüyor mu? ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, data, name in [(axes[0], heli, "HELI"), (axes[1], quad, "QUAD")]:
        mean_prof = data.mean(axis=0)
        ax.semilogy(mean_prof + 1e-6)
        ax.set_title(f"{name} — ortalama profil (log ölçek)")
        ax.set_xlabel("Bin")
        ax.set_ylabel("Normalize genlik (log)")
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("my_spectrum_log.png", dpi=130)
    print("✅ my_spectrum_log.png kaydedildi")


if __name__ == "__main__":
    main()
