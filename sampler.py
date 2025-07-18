import numpy as np
from scipy.signal import resample

class Sampler:
    @staticmethod
    def sample(signal, fs_orig, fs_sample):
        t_orig = np.arange(len(signal)) / fs_orig
        duration = t_orig[-1]
        n_samples = int(duration * fs_sample) + 1
        t_sampled = np.linspace(0, duration, n_samples)
        sampled_signal = np.interp(t_sampled, t_orig, signal)
        return sampled_signal, t_sampled

    @staticmethod
    def reconstruct(sampled_signal, t_sampled, fs_recon, duration=None):
        if duration is None:
            duration = t_sampled[-1]
        t_recon = np.linspace(0, duration, int(fs_recon * duration) + 1)
        recon_signal = np.zeros_like(t_recon)
        for n, t_n in enumerate(t_sampled):
            recon_signal += sampled_signal[n] * np.sinc(fs_recon * (t_recon - t_n))
        return recon_signal, t_recon

    @staticmethod
    def detect_aliasing(signal, fs_orig, fs_sample, freq_components):
        nyquist = fs_sample / 2
        for f in freq_components:
            if f > nyquist:
                return True
        return False 