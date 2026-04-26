"""
Rojhani vd., IEEE TMTT 2023 — "Model-Based Data Augmentation..."
Sentetik micro-Doppler menzil profili üretici.

Paper yapısı:
  - Fig. 2 (sentetik zincir): PHYSICAL MODEL → s_IF(t) → FFT → SYNTHETIC DATASET
  - Fig. 6 (ölçüm zinciri):   RX → 2D-FFT → MTI → açısal entegrasyon → menzil profili
  - Fig. 11: sentetik (kırmızı, σ_P=0) vs gerçek ölçümler (mavi, σ_P=0.20 augmentation için referans)

Bu modül her iki zinciri de destekler:
  - generate_single_profile(...) → Fig. 2 sentetik çıkış (CNN eğitim veri seti için)
  - generate_profile_mti(...)    → Fig. 6 MTI'lı çıkış (ölçüm benzeri, doğrulama için)
"""
import numpy as np


# ============================================================
# TABLE I — FMCW MIMO RADAR PLATFORM PARAMETERS
# ============================================================
C = 3e8                     # Işık hızı [m/s]
FC = 77e9                   # Taşıyıcı frekansı [Hz]
LAMBDA = C / FC             # Dalga boyu [m]
FS = 200e3                  # IF örnekleme frekansı [Hz]
N_SAMPLES = 504             # Chirp başına örnek
N_FFT = 4096                # FFT noktası
MU = 1.56e12                # Chirp slope μ [Hz/s]
N_RX = 8                    # RX anten sayısı
N_MTI = 10                  # MTI frame depth (Table I)
CRI = 200e-3                # Chirp repetition interval [s] — Table I
RX_SPACING = LAMBDA / 2     # Anten aralığı (λ/2)

# Açısal entegrasyon penceresi (Eq. 16, s. 2227)
DELTA_THETA = np.deg2rad(7.0)

# Giriş CNN profili uzunluğu (paper: "400 samples centered around target range")
PROFILE_LEN = 400

# Menzil ekseni çözünürlüğü: R = c0·f/(2μ),  f_bin = FS/N_FFT
BIN_HZ = FS / N_FFT
BIN_M = BIN_HZ * C / (2 * MU)   # metre/bin (~4.69 mm)

# ============================================================
# Paper Fig. 11 ölçüm menzilleri — Section IV-A-2
# ============================================================
R0_HELI = 2.1               # RadioFLY helikopter [m]
R0_QUAD = 1.8               # DJI Mavic Mini [m]
R0_RANGE = (1.2, 2.8)       # Rastgele dataset için hedef menzil dağılımı

# ============================================================
# TABLE II — MODEL PARAMETER VECTORS FOR EACH CLASS
# ============================================================
# HELI tek motorlu (Γ₂=Γ₃=Γ₄=0); QUAD dört motorlu (tümü aktif).
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


# ============================================================
# Sentez yardımcıları
# ============================================================

def bin_to_range_m(bin_idx):
    """FFT bin indeksi → menzil (m). R = c0·f/(2μ)."""
    return np.asarray(bin_idx) * BIN_M


def range_axis_around(r0_m, n=PROFILE_LEN):
    """r0_m etrafında n-örneklik metre ekseni (tepe merkeze hizalı)."""
    return r0_m + (np.arange(n) - n // 2) * BIN_M


def _augment_params(params, sigma_p, rng):
    """Eq. 12: p_n = p + 𝒩(0, (σ_p·|p|)²). σ_p=0 → birebir parametre."""
    if sigma_p == 0:
        return dict(params)
    return {k: v + rng.normal(0.0, sigma_p * abs(v)) for k, v in params.items()}


def _engine_vibration(p, t):
    """Eq. 10: α + β + karışım (ω_a±ω_b) bileşenlerinin toplamı."""
    w_a, w_b = p["omega_alpha"], p["omega_beta"]
    vib = p["A_alpha"] * np.cos(w_a * t + p["phi_alpha"])
    vib += p["A_beta"] * np.cos(w_b * t + p["phi_beta"])
    vib += p["A_mix_plus"] * np.cos(p["C_omega_plus"] * abs(w_a + w_b) * t + p["phi_mix_plus"])
    vib += p["A_mix_minus"] * np.cos(p["C_omega_minus"] * abs(w_a - w_b) * t + p["phi_mix_minus"])
    return vib


def _synthesize_if(base_params, sigma_p, R0, rng, t_offset=0.0):
    """
    Eq. 6 — FMCW IF sinyali (paper formülasyonu, birebir):
      s_IF(t) = Σ_k Γ_k · cos[(8πμ/c₀) · (R₀ + Σ_j A_kj cos(ω_kj t + φ_kj)) · t]

    Kompleks exp ile uyguluyoruz (|·| alındığında cos'a eşdeğer spektrum).
    NOT: Paper bu formülde carrier-phase round-trip terimini (4π/λ)·R(t) açıkça
    yazmıyor; modülasyon yalnızca chirp-slope üzerinden geliyor (modülasyon
    indeksi ~0.65 rad → temiz tepe + zayıf sidebandlar — Fig. 11 ile uyumlu).

    t_offset: frame'ler arası mutlak zaman ofseti (MTI için, Eq. 15).
    """
    t = np.arange(N_SAMPLES) / FS + t_offset
    s_if = np.zeros(N_SAMPLES, dtype=complex)

    # Paper Eq. 6 yazılı haliyle: s_IF = Σ Γ cos[(8πμ/c₀)·R(t)·t]
    # Eq. 14 (R = c·f/(2μ)) ile tutarlılık için katsayıyı 4πμ/c₀ alıyoruz
    # (yazılı "8" yerine — aksi takdirde peak menzil 2× kayar).
    # Modülasyon yalnızca chirp slope üzerinden:
    #   β = (4πμ/c₀) · A_vib · T_chirp ≈ 0.33 rad
    # Bu rejimde J_0 dominanttır; J_1 sidebandları zayıf (~16%) ve sinc
    # yan loblarıyla iç içe görünür. Carrier round-trip terimi (4π/λ)·R(t)
    # eklenirse β≈6.4'e fırlayıp Bessel comb ile spektrum dağılır — Fig. 11
    # ile uyumsuz. Paper [12]'nin Passafiume modeli muhtemelen carrier
    # terimini çıkarıyor; biz de aynısını yapıyoruz.
    K = 4 * np.pi * MU / C

    for params in base_params.values():
        if params.get("Gamma", 0) == 0:
            continue
        p = _augment_params(params, sigma_p, rng)
        vib = _engine_vibration(p, t)
        phase = K * (R0 + vib) * t
        s_if += p["Gamma"] * np.exp(1j * phase)

    return s_if


def _add_awgn(s_if, snr_db, rng):
    """Karmaşık AWGN, hedef SNR'a göre ölçekli."""
    sig_power = np.mean(np.abs(s_if) ** 2)
    noise_power = sig_power / (10 ** (snr_db / 10))
    noise = rng.normal(0, 1, s_if.size) + 1j * rng.normal(0, 1, s_if.size)
    return s_if + noise * np.sqrt(noise_power / 2)


def _extract_profile(range_spectrum, center_bin=None, profile_len=PROFILE_LEN):
    """4096-bin spektrumdan tepe etrafı 400 örnek al + max-normalize."""
    rp = np.abs(range_spectrum)
    if center_bin is None:
        center_bin = int(np.argmax(rp))
    start = max(0, center_bin - profile_len // 2)
    slice_ = rp[start:start + profile_len]
    if slice_.size < profile_len:
        slice_ = np.pad(slice_, (0, profile_len - slice_.size))
    m = slice_.max()
    return slice_ / m if m > 0 else slice_


# ============================================================
# Fig. 2 sentetik zincir — CNN eğitim veri seti için
# ============================================================

def generate_single_profile(base_params, sigma_p=0.20, R0=None, snr_db=None,
                            apply_window=False, rng=None):
    """
    Paper Fig. 2: PHYSICAL MODEL → s_IF → FFT → range profile.

    sigma_p=0     → Fig. 11 kırmızı çizgisi (deterministik).
    sigma_p=0.20  → Section IV-A-2 CNN dataset varsayılanı.
    apply_window  → Hann; sinc yan loblarını azaltır (varsayılan False, paper uyumu).
    """
    if rng is None:
        rng = np.random.default_rng()
    if R0 is None:
        R0 = rng.uniform(*R0_RANGE)

    s_if = _synthesize_if(base_params, sigma_p, R0, rng)
    if snr_db is not None:
        s_if = _add_awgn(s_if, snr_db, rng)
    if apply_window:
        s_if = s_if * np.hanning(N_SAMPLES)

    spectrum = np.fft.fft(s_if, n=N_FFT)

    # Sentetik zincirde peak target R0'a sabit olduğundan bin'i doğrudan hesaplayabiliriz
    target_bin = int(round(2 * R0 * MU / C / BIN_HZ))
    return _extract_profile(spectrum, center_bin=target_bin)


# ============================================================
# Fig. 6 ölçüm zinciri — MTI + açısal entegrasyon
# ============================================================

def generate_profile_mti(base_params, sigma_p=0.0, R0=None, n_mti=N_MTI,
                         snr_db=25.0, apply_window=False, rng=None,
                         r0_jitter_std=3e-3):
    """
    Paper Fig. 6 ölçüm zinciri (boresight hedef için 1D basitleştirilmiş).

    Eq. 15:  I_MTI(r) = I_current(r) − (1/N_MTI) · Σ_{i=1..N_MTI} I_{-i}(r)

    N_MTI=10 frame ardışık CRI=200 ms aralıkla üretilir. Rotor vibrasyonu
    mutlak zamana bağlı olduğundan frame'ler arası faz evrilir → MTI, statik
    bileşeni (gövde, zemin yansıması) bastırır; rotor AC bileşeni kalır.

    Ölçüm benzeri "measurement proxy" için iki eklenti (paperın Fig. 10e/f
    MTI çıktısına uyum için gerekli):

      r0_jitter_std  — hovering drone konum jitter'ı [m]. Sabit R0'da carrier
                       tam iptal olup profil yapısal olarak sentetiğe
                       benzemiyordu; ~3 mm (<1 bin) jitter ile carrier kısmen
                       korunur, profil Fig. 11 konturuna yaklaşır.
      snr_db         — per-chirp AWGN (varsayılan 25 dB, paperın 1.5 m hover
                       ölçümü için makul).
    """
    if rng is None:
        rng = np.random.default_rng()
    if R0 is None:
        R0 = rng.uniform(*R0_RANGE)

    frames = []
    for f in range(n_mti + 1):
        t_offset = f * CRI
        R0_frame = R0 + rng.normal(0.0, r0_jitter_std) if r0_jitter_std > 0 else R0
        s_if = _synthesize_if(base_params, sigma_p, R0_frame, rng, t_offset=t_offset)
        if snr_db is not None:
            s_if = _add_awgn(s_if, snr_db, rng)
        if apply_window:
            s_if = s_if * np.hanning(N_SAMPLES)
        frames.append(np.fft.fft(s_if, n=N_FFT))

    # Non-coherent (magnitude) MTI — paper Eq. 15, I(r,ϑ) "radar image" magnitudinal.
    # Kompleks (coherent) MTI denendi; R0 sabit iken carrier tam iptal olup Fig.
    # 10e/f davranışı çıkmıyor. Paperın MTI çıktısı pozitif magnitude domaininde
    # peak koruyor → magnitude subtraction tutarlı yorum.
    mags = np.abs(np.stack(frames))              # (N_MTI+1, N_FFT)
    reference = mags[:-1].mean(axis=0)
    mti_spectrum = np.maximum(mags[-1] - reference, 0.0)

    target_bin = int(round(2 * R0 * MU / C / BIN_HZ))
    return _extract_profile(mti_spectrum, center_bin=target_bin)


# ============================================================
# Ölçüm proxy'si — paper Fig. 13 davranışı için
# ============================================================

# Paper Fig. 11 incelemesi: ölçüm profilleri sentetik (σ_P=0) ile AYNI yapıda
# (sinc peak + vibrasyonal yan loblar), sadece inherent bir σ_P_meas ve ölçüm
# gürültüsü taşıyor. Paperın Section IV-C-1 bulgusu: eğitim σ_P = σ_P_meas
# olduğunda accuracy zirveye çıkıyor (σ_P_meas ≈ 0.20, Fig. 13).
#
# Biz gerçek ölçümü sentetize etmek için:
#   profile = generate_single_profile(params, sigma_p=SIGMA_P_MEAS, ..)
#          + additional measurement AWGN
# eğitim σ_P'sini 0..0.6 arasında süpürdüğümüzde accuracy SIGMA_P_MEAS
# civarında tepe yapacak — paper Fig. 13 ile aynı trend.
SIGMA_P_MEAS = 0.20        # paperın ölçüm setinde kestirdiği "true" σ_P
MEAS_SNR_DB = 20.0         # ölçüm gürültüsü (ek AWGN)


def generate_profile_measurement(base_params, R0=None, rng=None,
                                  sigma_p_meas=SIGMA_P_MEAS, snr_db=MEAS_SNR_DB):
    """
    Paperın ölçüm setinin sentetik eşdeğeri (Fig. 11 davranışı):
      - yapısal olarak Fig. 2 sentetik ile aynı
      - sabit inherent σ_P (= 0.20, paperın kestirdiği değer)
      - ek ölçüm AWGN (~20 dB)
    Eğitim σ_P'si bu değere yaklaştıkça sınıflandırıcı accuracy'si tepe yapar.
    """
    return generate_single_profile(base_params, sigma_p=sigma_p_meas,
                                    R0=R0, snr_db=snr_db, rng=rng)


# ============================================================
# Dataset üretici
# ============================================================

def build_dataset(num_samples_per_class=16000, sigma_p=0.20, snr_db=None,
                  use_mti=False, seed=42, out_X="X_data.npy", out_y="y_labels.npy"):
    """HELI (0) ve QUAD (1) için dengeli dataset üretir ve .npy olarak kaydeder."""
    rng = np.random.default_rng(seed)
    mode = "MTI (Fig. 6)" if use_mti else "sentetik (Fig. 2)"
    print(f"[{mode}] σ_p={sigma_p}, SNR={snr_db}, N/sınıf={num_samples_per_class}")

    gen = generate_profile_mti if use_mti else generate_single_profile
    classes = [(HELI_REF_PARAMS, 0), (QUAD_REF_PARAMS, 1)]
    X, y = [], []
    for params, label in classes:
        for i in range(num_samples_per_class):
            X.append(gen(params, sigma_p=sigma_p, snr_db=snr_db, rng=rng))
            y.append(label)
            if (i + 1) % 2000 == 0:
                print(f"  sınıf {label}: {i+1}/{num_samples_per_class}")

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)

    np.save(out_X, X)
    np.save(out_y, y)
    print(f"✅ {out_X} X={X.shape}, {out_y} y={y.shape}")


if __name__ == "__main__":
    # Paper Section IV-A-2 varsayılanı: σ_p=0.20, 16k/sınıf, temiz (SNR=None).
    build_dataset(num_samples_per_class=16000, sigma_p=0.20, snr_db=None,
                  use_mti=False, seed=42)
