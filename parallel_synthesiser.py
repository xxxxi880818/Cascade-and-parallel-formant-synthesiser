
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import sawtooth
import soundfile as sf



SAMPLE_RATE = 10000
DURATION = 0.3
F0 = 100

VOWELS = {
    'a': {'F': [730, 1090, 2440], 'B': [130, 70, 160], 'A': [1.0, 0.8, 0.5]},
    'i': {'F': [270, 2290, 3010], 'B': [60, 90, 100], 'A': [1.0, 0.8, 0.4]},
    'u': {'F': [300, 870, 2240], 'B': [60, 100, 150], 'A': [1.0, 0.7, 0.5]}
}

def generate_impulse_train(F0, fs, duration):
    N = int(fs * duration)
    signal = np.zeros(N)
    for i in range(0, N, int(fs // F0)):
        signal[i] = 1.0
    return signal

def formant_resonator(signal, F, fs, B):
    T_s = 1 / fs
    B_coef = 2 * np.exp(-np.pi * B * T_s) * np.cos(2 * np.pi * F * T_s)
    C_coef = -1 * np.exp(-2 * np.pi * B * T_s)
    A_coef = 1 - B_coef - C_coef
    y = np.zeros_like(signal)
    if len(signal) > 1:
        y[1] = A_coef * signal[0] + B_coef * y[0]
    for n in range(2, len(signal)):
        y[n] = A_coef * signal[n-1] + B_coef * y[n-1] + C_coef * y[n-2]
    return y

def parallel_synth(signal, F_list, B_list, A_list, fs):
    out = np.zeros_like(signal)
    for F, B, A in zip(F_list, B_list, A_list):
        out += A * formant_resonator(signal, F, fs, B)
    return out

def normalize(sig):
    return sig / np.max(np.abs(sig)) * 0.9

def synthesize_parallel():
    src = generate_impulse_train(F0, SAMPLE_RATE, DURATION)
    t = np.arange(len(src)) / SAMPLE_RATE
    for v, p in VOWELS.items():
        print(f"Generating vowel /{v}/ (parallel)...")
        y_p = normalize(parallel_synth(src, p['F'], p['B'], p['A'], SAMPLE_RATE))
        sf.write(f"{v}_parallel.wav", y_p, SAMPLE_RATE)
        plt.figure()
        plt.plot(t, y_p, label="Parallel", alpha=0.8)
        plt.title(f"Waveform: /{v}/ (Parallel)")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{v}_parallel_waveform.png")
        plt.close()
        print(f"Saved {v}_parallel.wav and waveform plot.")

if __name__ == "__main__":
    synthesize_parallel()
