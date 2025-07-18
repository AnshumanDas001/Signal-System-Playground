import numpy as np
from scipy.signal import butter, lfilter

class SignalProcessor:
    @staticmethod
    def add_noise(signal, noise_type='gaussian', snr_db=20):
        if noise_type == 'gaussian':
            signal_power = np.mean(signal ** 2)
            snr = 10 ** (snr_db / 10)
            noise_power = signal_power / snr
            noise = np.random.normal(0, np.sqrt(noise_power), signal.shape)
            return signal + noise
        else:
            raise ValueError('Unsupported noise type')

    @staticmethod
    def lowpass_filter(signal, cutoff, fs, order=5):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        return lfilter(b, a, signal)

    @staticmethod
    def highpass_filter(signal, cutoff, fs, order=5):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='high', analog=False)
        return lfilter(b, a, signal)

    @staticmethod
    def bandpass_filter(signal, lowcut, highcut, fs, order=5):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band', analog=False)
        return lfilter(b, a, signal) 