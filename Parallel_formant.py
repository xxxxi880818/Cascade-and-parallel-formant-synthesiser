import numpy as np
import math

# # amplitude response
def _privateAmplitudeResponse(A, B, C, omega_Hz, array, formant, t_s):
    # Args:
    # A, B, C: list, 3 coefficients
    # omega_Hz: float, angular frequency, = 2*pi*frequency
    # array: array, to store result
    # formant: int, formant number, 1-2
    # t_s: float, sampling period Ts

    # Retruns: int, amplitude values in linear scale for every frequency
    assert isinstance(array, np.ndarray)
    for i in range(5000):
        amp = A[formant] / math.sqrt(
            math.pow((1 - C[formant]) * math.cos(omega_Hz[i] * t_s) - B[formant], 2) + math.pow(
                (1 + C[formant]) * math.sin(omega_Hz[i] * t_s), 2))
        array[0][i] = amp
    return array


# phase response
def _privatePhaseResponse(A, B, C, omega_Hz, array, formant, t_s):
    
    # Args:
    # A, B, C: list, 3 coefficients
    # omega_Hz: float, angular frequency, = 2*pi*frequency
    # array: array, to store result
    # formant: int, formant number, 1-3
    # t_s: float, sampling period Ts

    # Retruns: int, phase values in radians for every frequency
    
    assert isinstance(array, np.ndarray)
    for i in range(5000):
        pha = -math.atan2((1 + C[formant]) * math.sin(omega_Hz[i] * t_s),
                          (1 - C[formant]) * math.cos(omega_Hz[i] * t_s) - B[formant])
        array[0][i] = pha
    return array

# frequency response
def parallel_frequency_response(gain, f_n, b_n):
    '''
    Args:
    gain: list, amplitude control for each formant
    f_n: list, cutoff frequency (F)
    b_n: list, bandwidth (BW)

    Retruns:
    amplitude response graph, phase response graph, amplitude response array, phase response array
    '''
    assert isinstance(f_n, list) and isinstance(b_n, list) and isinstance(gain, list)
    f_step = 1  # int, frequency step
    f_s = 10000  # int, sampling frequency (Fs): 10 kHz
    f_Hz = np.arange(0, 5000, f_step)  # array, frequency range(0Hz,5000Hz), with frequency step 1 Hz, altogether 5000 samples
    omega_Hz = 2 * math.pi * f_Hz  # array, angular frequency = 2*pi*f
    t_s = 1 / f_s  # float, sampling period (Ts) = 1/sampling frequency = 0.1 ms

    # compute sigma & omega
    sigma, omega = [], []
    for i in range(len(f_n)):
        sigma.append(-np.pi * b_n[i])  # σ = -pi*BW
        omega.append(2 * np.pi * f_n[i])  # ω = 2*pi*F

    # compute A,B,C
    A, B, C = [], [], []
    for i in range(len(f_n)):
        C.append(-math.exp(2 * sigma[i] * t_s))  # C = -exp(-2*pi*BW*Ts) = -exp(2*σ*Ts)
        B.append(2 * math.exp(sigma[i] * t_s) * np.cos(
            omega[i] * t_s))  # B = 2*exp(-pi*BW*Ts)*cos(2*pi*F*Ts) = 2*exp(σ*Ts)*cos(ω*Ts)
        A.append(1 - B[i] - C[i])  # A = 1- B - C

    # # first order low-pass filter @mal
    B.append(math.exp(-2 * math.pi * 100 * t_s))
    A.append(1 - B[3])

    # create arrays for storing amplitude response of 3 formants
    amp_1 = np.empty([1, 5000], dtype=float)
    amp_2 = np.empty([1, 5000], dtype=float)
    amp_3 = np.empty([1, 5000], dtype=float)

    # compute amplitude response of 3 formants, adjust amplitude for each formant
    amps = [amp_1, amp_2, amp_3]
    for i in range(len(gain)):
        amps[i] = gain[i] * _privateAmplitudeResponse(A, B, C, omega_Hz, amps[i], i, t_s)
    amp_1, amp_2, amp_3 = amps

    # create arrays for storing phase response of 3 formants
    pha_1 = np.empty([1, 5000], dtype=float)
    pha_2 = np.empty([1, 5000], dtype=float)
    pha_3 = np.empty([1, 5000], dtype=float)

    # compute phase response of 3 formants
    phas = [pha_1, pha_2, pha_3]
    for i in range(len(phas)):
        phas[i] = _privatePhaseResponse(A, B, C, omega_Hz, phas[i], i, t_s)
    pha_1, pha_2, pha_3 = phas

    # frequency response
    # add up 3 sets of amplitude response (subtract the second resonator due to polarity reversal)
    amp_sum = np.empty([1, 5000], dtype=float)
    for i in range(5000):
        result = math.sqrt((amp_1[0][i] ** 2) + (amp_2[0][i] ** 2) + (amp_3[0][i] ** 2)
                           - 2 * amp_1[0][i] * amp_2[0][i] * math.cos(pha_1[0][i] - pha_2[0][i])
                           + 2 * amp_1[0][i] * amp_3[0][i] * math.cos(pha_1[0][i] - pha_3[0][i])
                           - 2 * amp_2[0][i] * amp_3[0][i] * math.cos(pha_2[0][i] - pha_3[0][i]))
        amp_sum[0][i] = 20 * np.log10(result)  # convert to dB scale

    # phase response
    # sum up 3 sets of phase response
    pha_sum = np.empty([1, 5000], dtype=float)
    for i in range(5000):
        result = math.atan2(
            amp_1[0][i] * math.sin(pha_1[0][i]) - amp_2[0][i] * math.sin(pha_2[0][i]) + amp_3[0][i] * math.sin(
                pha_3[0][i]),
            amp_1[0][i] * math.cos(pha_1[0][i]) - amp_2[0][i] * math.cos(pha_2[0][i]) + amp_3[0][i] * math.cos(
                pha_3[0][i]))
        pha_sum[0][i] = np.degrees(result)  # convert to degree scale
    pha_sum[0] = np.unwrap(pha_sum[0])

    return amp_sum, pha_sum
