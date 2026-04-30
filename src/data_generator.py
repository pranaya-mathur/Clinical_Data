import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.signal import fftconvolve
import os

# =========================
# GLOBAL CONFIGURATION
# =========================
np.random.seed(42)
NUM_SAMPLES = 1200
NUM_POINTS = 1024
x = np.linspace(0, 1, NUM_POINTS)

# =========================
# HELPER FUNCTIONS
# =========================
def gaussian(x, mu, sigma, amp):
    return amp * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

def emg_peak(x, mu, sigma, tau, amp):
    g = norm.pdf(x, mu, sigma)
    exp_tail = np.exp(-(x - mu) / tau) * (x >= mu)
    exp_tail /= np.sum(exp_tail) + 1e-12
    return amp * fftconvolve(g, exp_tail, mode='same')

def baseline_drift(x):
    freq1 = np.random.uniform(0.2, 1.5)
    freq2 = np.random.uniform(0.3, 2.0)
    drift = (
        0.006 * np.sin(2 * np.pi * x * freq1) +
        0.004 * np.cos(2 * np.pi * x * freq2) +
        np.polyval(np.random.uniform(-0.012, 0.012, 3), x)
    )
    return drift

def add_noise(signal):
    noise_level = np.random.uniform(0.002, 0.012)
    return signal + np.random.normal(0, noise_level, size=signal.shape)

def jitter_positions(positions):
    return positions * np.random.uniform(0.98, 1.02, size=len(positions))

def normalize_curve(signal):
    max_val = np.max(signal)
    return signal / (max_val + 1e-8) if max_val > 0 else signal

def generate_hdl(has_plaque):
    base_positions = np.array([0.22, 0.30, 0.39, 0.48, 0.57])
    positions = jitter_positions(base_positions)
    sigmas = np.array([0.014, 0.017, 0.019, 0.021, 0.024])

    if has_plaque:
        base_amps = np.array([0.65, 0.75, 0.85, 0.95, 1.05])
    else:
        base_amps = np.array([0.95, 0.85, 0.75, 0.65, 0.55])
    
    amps = base_amps * np.random.uniform(0.75, 1.25, size=5)
    amps += np.random.normal(0, 0.08, size=5)

    clean_peaks = []
    areas = []
    for mu, sigma, amp in zip(positions, sigmas, amps):
        peak = gaussian(x, mu, sigma, amp) if np.random.rand() < 0.6 else emg_peak(x, mu, sigma, tau=0.013, amp=amp)
        clean_peaks.append(peak)
        # Using trapezoid for NumPy 2.0+ compatibility
        areas.append(np.trapezoid(peak, x))

    signal = np.sum(clean_peaks, axis=0)
    signal += baseline_drift(x)
    signal = add_noise(signal)
    signal = np.clip(signal, 0, None)
    signal = normalize_curve(signal)

    fractions = np.array(areas) / np.sum(areas)
    return signal, fractions

def generate_ldl(has_plaque):
    base_positions = np.array([0.62, 0.71, 0.79, 0.86])
    positions = jitter_positions(base_positions)
    sigmas = np.array([0.028, 0.023, 0.019, 0.016])

    if has_plaque:
        base_amps = np.array([0.85, 1.05, 0.95, 0.75])
    else:
        base_amps = np.array([1.05, 0.75, 0.65, 0.55])
    
    amps = base_amps * np.random.uniform(0.8, 1.2, size=4)
    amps += np.random.normal(0, 0.07, size=4)

    clean_peaks = []
    areas = []
    for mu, sigma, amp in zip(positions, sigmas, amps):
        peak = gaussian(x, mu, sigma, amp) if np.random.rand() < 0.6 else emg_peak(x, mu, sigma, tau=0.014, amp=amp)
        clean_peaks.append(peak)
        # Using trapezoid for NumPy 2.0+ compatibility
        areas.append(np.trapezoid(peak, x))

    signal = np.sum(clean_peaks, axis=0)
    signal += baseline_drift(x)
    signal = add_noise(signal)
    signal = np.clip(signal, 0, None)
    signal = normalize_curve(signal)

    fractions = np.array(areas) / np.sum(areas)
    sdldl_percent = fractions[1]
    return signal, sdldl_percent

def run_generation(output_dir="data"):
    print(f"🚀 Starting synthetic data generation (N={NUM_SAMPLES})...")
    os.makedirs(output_dir, exist_ok=True)
    
    hdl_curves = []
    ldl_curves = []
    metadata = []

    for i in range(NUM_SAMPLES):
        has_plaque = 1 if np.random.rand() < 0.65 else 0
        hdl_curve, hdl_frac = generate_hdl(has_plaque)
        ldl_curve, sdldl = generate_ldl(has_plaque)

        age = int(np.clip(np.random.normal(55 if has_plaque else 50, 13), 30, 85))
        sex_male = 1 if np.random.rand() < (0.58 if has_plaque else 0.48) else 0
        bmi = round(np.clip(np.random.normal(27.8 if has_plaque else 26.2, 5.0), 18.0, 40.0), 1)
        diabetes = 1 if np.random.rand() < (0.32 if has_plaque else 0.18) else 0
        hypertension = 1 if np.random.rand() < (0.55 if has_plaque else 0.38) else 0
        smoking = 1 if np.random.rand() < (0.33 if has_plaque else 0.22) else 0

        record = {
            "sample_id": i, "has_plaque": has_plaque, "age": age, "sex_male": sex_male,
            "bmi": bmi, "diabetes": diabetes, "hypertension": hypertension, "smoking": smoking,
            "hdl2b_percent": hdl_frac[0], "hdl2a_percent": hdl_frac[1], "hdl3a_percent": hdl_frac[2],
            "hdl3b_percent": hdl_frac[3], "hdl3c_percent": hdl_frac[4], "sdldl_percent": sdldl,
            "total_hdl": np.sum(hdl_curve), "total_ldl": np.sum(ldl_curve),
            "hdl_ldl_ratio": np.sum(hdl_curve) / (np.sum(ldl_curve) + 1e-8)
        }

        hdl_curves.append(hdl_curve)
        ldl_curves.append(ldl_curve)
        metadata.append(record)

    metadata_df = pd.DataFrame(metadata)
    
    # Save files
    metadata_df.to_csv(os.path.join(output_dir, "synthetic_lipospec_dataset_metadata.csv"), index=False)
    np.save(os.path.join(output_dir, "hdl_curves.npy"), np.array(hdl_curves))
    np.save(os.path.join(output_dir, "ldl_curves.npy"), np.array(ldl_curves))
    
    print(f"✅ Dataset generated and saved in '{output_dir}/'")

if __name__ == "__main__":
    run_generation()
