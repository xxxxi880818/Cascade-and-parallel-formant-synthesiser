import numpy as np
import math
from Generate_files import *

# amplitude response

def _private_cascadeAmplitudeResponse(A, B, C, omega_Hz, array, formant, t_s):
    assert isinstance(array, np.ndarray)
    for i in range(5000):
        cos_term = (1 - C[formant]) * np.cos(omega_Hz[i] * t_s) - B[formant]
        sin_term = (1 + C[formant]) * np.sin(omega_Hz[i] * t_s)
        amplitude = A[formant] / np.sqrt(cos_term**2 + sin_term**2)
        array[0][i] = 20 * np.log10(amplitude)
    return array

# phase response

def _private_cascadePhaseResponse(A, B, C, omega_Hz, array, formant, t_s):
    assert isinstance(array, np.ndarray)
    for i in range(5000):
        cos_term = (1 - C[formant]) * np.cos(omega_Hz[i] * t_s) - B[formant]
        sin_term = (1 + C[formant]) * np.sin(omega_Hz[i] * t_s)
        phase = -np.arctan2(sin_term, cos_term)
        array[0][i] = (phase * 180 / np.pi)
    return array

# frequency response

def cascade_frequency_response(vowel, f_n, b_n):
    assert isinstance(f_n, list) and isinstance(b_n, list)
    f_step = 1
    f_s = 10000
    f_Hz = np.arange(0, 5000, f_step)
    omega_Hz = 2 * math.pi * f_Hz
    t_s = 1 / f_s

    sigma, omega = [], []
    for i in range(5):
        sigma.append(-np.pi * b_n[i])
        omega.append(2 * np.pi * f_n[i])

    A, B, C = [], [], []
    for i in range(5):
        C.append(-math.exp(2 * sigma[i] * t_s))
        B.append(2 * math.exp(sigma[i] * t_s) * np.cos(omega[i] * t_s))
        A.append(1 - B[i] - C[i])

    amp_1 = np.empty([1, 5000], dtype=float)
    amp_2 = np.empty([1, 5000], dtype=float)
    amp_3 = np.empty([1, 5000], dtype=float)
    amp_4 = np.empty([1, 5000], dtype=float)
    amp_5 = np.empty([1, 5000], dtype=float)

    for i, amp in enumerate([amp_1, amp_2, amp_3, amp_4, amp_5]):
        _private_cascadeAmplitudeResponse(A, B, C, omega_Hz, amp, i, t_s)

    amp_sum = np.empty([1, 5000], dtype=float)
    for i in range(5000):
        amp_sum[0][i] = amp_1[0][i] + amp_2[0][i] + amp_3[0][i] + amp_4[0][i] + amp_5[0][i]

    pha_1 = np.empty([1, 5000], dtype=float)
    pha_2 = np.empty([1, 5000], dtype=float)
    pha_3 = np.empty([1, 5000], dtype=float)
    pha_4 = np.empty([1, 5000], dtype=float)
    pha_5 = np.empty([1, 5000], dtype=float)

    for i, pha in enumerate([pha_1, pha_2, pha_3, pha_4, pha_5]):
        _private_cascadePhaseResponse(A, B, C, omega_Hz, pha, i, t_s)

    pha_sum = np.empty([1, 5000], dtype=float)
    for i in range(5000):
        pha_sum[0][i] = pha_1[0][i] + pha_2[0][i] + pha_3[0][i] + pha_4[0][i] + pha_5[0][i]

    return amp_sum, pha_sum
