import numpy as np
import json
import os

# Makale Tablo II: HELI (Tek Motorlu İHA) Referans Parametreleri
HELI_REF_PARAMS = {
    "engine_1": {
        "Gamma": 0.044,
        "A_alpha": 2.05e-3, "omega_alpha": 2.69e3, "phi_alpha": 1.04,
        "A_beta": 2.08e-3,  "omega_beta": 3.98e3,  "phi_beta": 0.81,
        "A_mix_plus": 6.68e-4, "C_omega_plus": 1.04, "phi_mix_plus": 1.90,
        "A_mix_minus": 2.35e-3, "C_omega_minus": 0.92, "phi_mix_minus": -0.04
    }
}

# Makale Tablo II: QUAD (Dört Motorlu İHA) Referans Parametreleri
QUAD_REF_PARAMS = {
    "engine_1": {
        "Gamma": 0.001,
        "A_alpha": 2.24e-3, "omega_alpha": 1.77e3, "phi_alpha": -1.70,
        "A_beta": 2.19e-3,  "omega_beta": 4.84e3,  "phi_beta": -0.51,
        "A_mix_plus": 1.64e-3, "C_omega_plus": 1.00, "phi_mix_plus": 3.88,
        "A_mix_minus": 3.60e-3, "C_omega_minus": 1.00, "phi_mix_minus": 0.01
    },
    "engine_2": {
        "Gamma": 0.003,
        "A_alpha": 3.24e-3, "omega_alpha": 4.79e3, "phi_alpha": 1.06,
        "A_beta": 1.63e-3,  "omega_beta": 2.61e3,  "phi_beta": 1.93,
        "A_mix_plus": 5.42e-3, "C_omega_plus": 1.00, "phi_mix_plus": -2.64,
        "A_mix_minus": 2.61e-3, "C_omega_minus": 0.99, "phi_mix_minus": -1.54
    },
    "engine_3": {
        "Gamma": 0.001,
        "A_alpha": 1.67e-3, "omega_alpha": 2.82e3, "phi_alpha": 1.52,
        "A_beta": 3.02e-3,  "omega_beta": 5.06e3,  "phi_beta": 3.35,
        "A_mix_plus": 1.31e-2, "C_omega_plus": 1.00, "phi_mix_plus": 4.31,
        "A_mix_minus": 4.76e-3, "C_omega_minus": 1.00, "phi_mix_minus": -1.70
    },
    "engine_4": {
        "Gamma": 0.011,
        "A_alpha": 1.82e-3, "omega_alpha": 5.07e3, "phi_alpha": -0.68,
        "A_beta": 1.44e-3,  "omega_beta": 2.62e3,  "phi_beta": -2.58,
        "A_mix_plus": 8.29e-4, "C_omega_plus": 1.00, "phi_mix_plus": -3.92,
        "A_mix_minus": 3.88e-4, "C_omega_minus": 1.00, "phi_mix_minus": -1.71
    }
}

def generate_synthetic_radar_data(target_class: str, sigma_p: float = 0.20, num_samples: int = 1, output_filename: str = "synthetic_data.json") -> str:
    """
    Belirtilen İHA sınıfı için sentetik parametreler üretir ve doğrudan dosyaya kaydeder.
    """
    target_class = target_class.upper()
    if target_class not in ["QUAD", "HELI"]:
        return "Hata: Hedef sınıfı 'QUAD' veya 'HELI' olmalıdır."

    base_params = QUAD_REF_PARAMS if target_class == "QUAD" else HELI_REF_PARAMS
    generated_samples = []
    
    for _ in range(num_samples):
        synthetic_sample = {}
        for engine_name, params in base_params.items():
            synthetic_sample[engine_name] = {}
            for param_name, base_value in params.items():
                std_dev = sigma_p * abs(base_value)
                noise = np.random.normal(0, std_dev)
                synthetic_value = base_value + noise
                synthetic_sample[engine_name][param_name] = float(np.format_float_scientific(synthetic_value, precision=4))
                
        generated_samples.append(synthetic_sample)

    # Veriyi dosyaya yaz
    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(generated_samples, f, indent=2)

    # Claude'a sadece başarılı olduğuna dair kısa bir bilgi dön
    return f"✅ Başarılı: {num_samples} adet {target_class} sentetik verisi üretildi ve '{os.path.abspath(output_filename)}' konumuna kaydedildi."

# Claude'un kullanacağı güncel araç şeması
RADAR_SKILL_DESCRIPTIONS = [
    {
        "name": "generate_synthetic_radar_data",
        "description": "QUAD veya HELI sınıfları için sentetik FMCW radar mikro-Doppler parametreleri üretir ve diske kaydeder.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "target_class": {
                    "type": "string",
                    "description": "İHA hedef sınıfı ('QUAD' veya 'HELI')",
                    "enum": ["QUAD", "HELI"]
                },
                "sigma_p": {
                    "type": "number",
                    "description": "Yayılım parametresi.",
                    "default": 0.20
                },
                "num_samples": {
                    "type": "integer",
                    "description": "Üretilecek örnek sayısı.",
                    "default": 1
                },
                "output_filename": {
                    "type": "string",
                    "description": "Verilerin kaydedileceği JSON dosyasının adı (örn: quad_data.json).",
                    "default": "synthetic_data.json"
                }
            },
            "required": ["target_class"]
        }
    }
]