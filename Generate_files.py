import matplotlib.pyplot as plt
from scipy.io import wavfile
import numpy as np


def plot_responses(f_Hz, cas_amp, amp, cas_pha, pha, vowel):
    # amplitude response
    plt.figure(f"Amplitude response of [{vowel}]")
    plt.xlabel('frequency (Hz)')
    plt.ylabel('amplitude (dB)')
    plt.plot(f_Hz, cas_amp[0], label='cascade model', color="green")
    plt.plot(f_Hz, amp[0], '--', color='red', label='parallel model')
    plt.legend()
    plt.title(f'Amplitude Response for vowel [{vowel}]')
    plt.grid(linestyle='-.')
    plt.savefig('amplitude_response_' + str(vowel) + '.jpg', dpi=300)

    # phase response
    plt.figure(f"Phase response of [{vowel}]")
    plt.xlabel('frequency (Hz)')
    plt.ylabel('Phase (radians)')
    plt.plot(f_Hz, cas_pha[0], label='cascade model', color="green")
    plt.plot(f_Hz, pha[0], '--', color='red', label='parallel model')
    plt.legend()
    plt.title(f'Phase Response for vowel [{vowel}]')
    plt.grid(linestyle='-.')
    plt.savefig('phase_response_' + str(vowel) + '.jpg', dpi=300)

    plt.show()

def plot_waveform(time, cas_output, output, vowel):
    plt.figure("Output waveform")
    plt.xlabel('time (ms)')
    plt.ylabel('amplitude')
    plt.plot(time[2:300], cas_output[2:300], label='cascade model', color='green')
    plt.plot(time[2:300], output[2:300], label='parallel model', color='red')
    plt.legend()
    plt.grid(linestyle='-.')
    plt.savefig('Output_wavefor_' + str(vowel) + '.jpg', dpi=300)
    plt.title('Output waveform[' + str(vowel) + ']')

    plt.show()



def save_audio_file(output, f_s, vowel):
    
    # Saves a waveform as an audio file.

    # Arguments:
    # output -- array containing the waveform data
    # f_s -- the sampling frequency of the waveform
    # fileName -- the name of the file to be saved
    
    wav_data = np.array(output)
    fileName = "{}_output.wav".format(vowel)
    wavfile.write(fileName, f_s, wav_data.astype(np.int16))