"""HELI vs QUAD sentetik profiller üst üste: CNN'in öğrenebileceği fark var mı?"""
import numpy as np
import matplotlib.pyplot as plt
from main import (HELI_REF_PARAMS, QUAD_REF_PARAMS, generate_single_profile,
                  R0_HELI, R0_QUAD, range_axis_around)

rng = np.random.default_rng(0)

# Aynı R0'da (karşılaştırılabilir ekseneler için) σ_P=0 deterministik
R0 = 2.0
heli = generate_single_profile(HELI_REF_PARAMS, sigma_p=0.0, R0=R0, rng=rng)
quad = generate_single_profile(QUAD_REF_PARAMS, sigma_p=0.0, R0=R0, rng=rng)

x = range_axis_around(R0)
fig, axes = plt.subplots(2, 1, figsize=(10, 7))

axes[0].plot(x, heli, color="blue", label="HELI (σ_P=0)")
axes[0].plot(x, quad, color="red", label="QUAD (σ_P=0)")
axes[0].set_title("HELI vs QUAD — aynı R0=2.0m, σ_P=0")
axes[0].set_xlabel("Range (m)")
axes[0].set_ylabel("Normalized")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Fark
axes[1].plot(x, quad - heli, color="purple")
axes[1].set_title("QUAD − HELI (farkı gösterir)")
axes[1].set_xlabel("Range (m)")
axes[1].set_ylabel("Δ")
axes[1].grid(True, alpha=0.3)

# Dominant farkın hangi bin'lerde olduğu
peak_diff_bins = np.argsort(np.abs(quad - heli))[-5:][::-1]
print("En büyük farklı 5 bin (x-m | Δ):")
for b in peak_diff_bins:
    print(f"  bin {b:3d}  range={x[b]:.3f} m  Δ={quad[b]-heli[b]:+.4f}")

plt.tight_layout()
plt.savefig("class_diff.png", dpi=130)
print("✅ class_diff.png")
