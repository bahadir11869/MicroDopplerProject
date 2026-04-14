import numpy as np

# --- MAKALE TABLE II — MODEL PARAMETER VECTORS FOR EACH CLASS ---
# Rojhani vd., IEEE TMTT, 2023 (doi: 10.1109/TMTT.2023.xxxx) — Table II
# HELI tek motorlu: yalnızca engine_1 aktif (Γ₂=Γ₃=Γ₄=0).
# QUAD dört motorlu: tüm engine'ler aktif.

HELI_REF_PARAMS = {
    "engine_1": {
        "Gamma": 0.044, "A_alpha": 2.05e-3, "omega_alpha": 2.69e3, "phi_alpha": 1.04,
        "A_beta": 2.08e-3, "omega_beta": 3.98e3, "phi_beta": 0.81,
        "A_mix_plus": 6.68e-4, "C_omega_plus": 1.04, "phi_mix_plus": 1.90,
        "A_mix_minus": 2.35e-3, "C_omega_minus": 0.92, "phi_mix_minus": -0.04
    },
}

QUAD_REF_PARAMS = {
    "engine_1": {
        "Gamma": 0.001, "A_alpha": 2.24e-3, "omega_alpha": 1.77e3, "phi_alpha": -1.70,
        "A_beta": 2.19e-3, "omega_beta": 4.84e3, "phi_beta": -0.51,
        "A_mix_plus": 1.64e-3, "C_omega_plus": 1.00, "phi_mix_plus": 3.88,
        "A_mix_minus": 3.60e-3, "C_omega_minus": 1.00, "phi_mix_minus": 0.01
    },
    "engine_2": {
        "Gamma": 0.003, "A_alpha": 3.24e-3, "omega_alpha": 4.79e3, "phi_alpha": 1.06,
        "A_beta": 1.63e-3, "omega_beta": 2.61e3, "phi_beta": 1.93,
        "A_mix_plus": 1.42e-3, "C_omega_plus": 1.00, "phi_mix_plus": -2.64,
        "A_mix_minus": 2.61e-3, "C_omega_minus": 0.99, "phi_mix_minus": -1.54
    },
    "engine_3": {
        "Gamma": 0.001, "A_alpha": 1.67e-3, "omega_alpha": 2.82e3, "phi_alpha": 1.52,
        "A_beta": 3.02e-3, "omega_beta": 5.06e3, "phi_beta": 3.35,
        "A_mix_plus": 1.31e-3, "C_omega_plus": 1.00, "phi_mix_plus": 4.31,
        "A_mix_minus": 4.76e-3, "C_omega_minus": 1.00, "phi_mix_minus": -1.70
    },
    "engine_4": {
        "Gamma": 0.011, "A_alpha": 1.82e-3, "omega_alpha": 5.07e3, "phi_alpha": -0.68,
        "A_beta": 1.44e-3, "omega_beta": 2.62e3, "phi_beta": -2.58,
        "A_mix_plus": 8.29e-4, "C_omega_plus": 1.00, "phi_mix_plus": -3.92,
        "A_mix_minus": 3.88e-3, "C_omega_minus": 1.00, "phi_mix_minus": -1.71
    },
}

# --- RADAR PARAMETRELERİ (Table I) ---
C = 3e8                 # Işık hızı
FC = 77e9               # Taşıyıcı frekansı [Hz]
LAMBDA = C / FC
FS = 200e3              # IF örnekleme [Hz]
N_SAMPLES = 504         # Chirp başına örnek
N_FFT = 4096            # FFT noktası
MU = 1.56e12            # Chirp slope μ [Hz/s] (4 GHz / 2.56 ms ≈ 1.56e12)

PROFILE_LEN = 400
R0_RANGE = (1.2, 2.8)   # Hedef menzil dağılımı [m] — paper Fig. 11'deki aralık


def _augment_params(params, sigma_p, rng):
    """Denklem 12: p_n = p + 𝒩(0, (σ_p·|p|)²)  (oranlı Gauss gürültü)."""
    return {k: v + rng.normal(0.0, sigma_p * abs(v)) for k, v in params.items()}


def _engine_vibration(p, t):
    """Denklem 5: α, β ve karışım (w_a±w_b) bileşenlerinin toplamı."""
    w_a, w_b = p["omega_alpha"], p["omega_beta"]
    vib = p["A_alpha"] * np.cos(w_a * t + p["phi_alpha"])
    vib += p["A_beta"] * np.cos(w_b * t + p["phi_beta"])
    vib += p["A_mix_plus"] * np.cos(p["C_omega_plus"] * abs(w_a + w_b) * t + p["phi_mix_plus"])
    vib += p["A_mix_minus"] * np.cos(p["C_omega_minus"] * abs(w_a - w_b) * t + p["phi_mix_minus"])
    return vib


def generate_single_profile(base_params, sigma_p=0.20, R0=None, snr_db=None, rng=None):
    """
    Tek bir sentetik menzil profili üretir.
      1) Denklem 12 — parametreleri σ_p oranlı Gauss gürültüyle artırır (augmentation)
      2) Denklem 6  — IF sinyalini sentezler: s = Σ Γ·exp(j·4π/λ·(R₀ + vib(t)))
      3) Denklem 7  — NFFT=4096 FFT → tepe etrafında 400 örneklik profil → max-normalize

    snr_db: None ise alıcı gürültüsü eklenmez; verilirse karmaşık AWGN ile
    hedeflenen SNR sağlanır (makaledeki ölçüm gürültüsü modeli).
    """
    if rng is None:
        rng = np.random.default_rng()
    if R0 is None:
        R0 = rng.uniform(*R0_RANGE)

    t = np.arange(N_SAMPLES) / FS
    s_if = np.zeros(N_SAMPLES, dtype=complex)

    # FMCW beat frekansı — menzile karşılık gelen tepe bin'ini sağlar (paper Denk. 6)
    f_beat = 2 * R0 * MU / C

    for params in base_params.values():
        if params.get("Gamma", 0) == 0:
            continue
        p = _augment_params(params, sigma_p, rng)
        vib = _engine_vibration(p, t)
        phase = 2 * np.pi * f_beat * t + (4 * np.pi / LAMBDA) * (R0 + vib)
        s_if += p["Gamma"] * np.exp(1j * phase)

    if snr_db is not None:
        sig_power = np.mean(np.abs(s_if) ** 2)
        noise_power = sig_power / (10 ** (snr_db / 10))
        noise = (rng.normal(0, 1, N_SAMPLES) + 1j * rng.normal(0, 1, N_SAMPLES))
        s_if = s_if + noise * np.sqrt(noise_power / 2)

    range_profile = np.abs(np.fft.fft(s_if, n=N_FFT))
    peak_idx = int(np.argmax(range_profile))
    start = max(0, peak_idx - PROFILE_LEN // 2)
    slice_ = range_profile[start:start + PROFILE_LEN]
    if slice_.size < PROFILE_LEN:
        slice_ = np.pad(slice_, (0, PROFILE_LEN - slice_.size))

    m = slice_.max()
    return slice_ / m if m > 0 else slice_


def build_dataset(num_samples_per_class=16000, sigma_p=0.20, snr_db=None, seed=42):
    """HELI (0) ve QUAD (1) için dengeli dataset üretir ve .npy olarak kaydeder."""
    rng = np.random.default_rng(seed)
    print(f"Veri seti üretiliyor… her sınıf için {num_samples_per_class} örnek (σ_p={sigma_p}, SNR={snr_db}).")

    classes = [(HELI_REF_PARAMS, 0), (QUAD_REF_PARAMS, 1)]
    X, y = [], []
    for params, label in classes:
        for _ in range(num_samples_per_class):
            X.append(generate_single_profile(params, sigma_p=sigma_p, snr_db=snr_db, rng=rng))
            y.append(label)

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)

    np.save("X_data.npy", X)
    np.save("y_labels.npy", y)
    print(f"✅ Kaydedildi. X={X.shape}, y={y.shape}")


if __name__ == "__main__":
    # Makaleyle aynı: 16.000 / sınıf, σ_p=0.20. snr_db=None → temiz sinyal.
    build_dataset(num_samples_per_class=16000, sigma_p=0.20, snr_db=None, seed=42)