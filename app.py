import streamlit as st
import numpy as np
from scipy.signal import square, sawtooth
from system_properties import SystemProperties
from signal_processor import SignalProcessor
from sampler import Sampler
import sympy as sp
import numbers

st.title('Signal System Playground')

def safe_numeric_array(arr, t):
    out = []
    for val in arr:
        try:
            if isinstance(val, numbers.Number):
                out.append(float(val))
            elif hasattr(val, 'evalf'):
                out.append(float(val.evalf()))
            else:
                out.append(0.0)
        except Exception:
            out.append(0.0)
    return np.array(out, dtype=float)

# Input mode selection
time_or_laplace = st.radio('Input Mode', ['Time-domain', 'Laplace-domain'])

# Common signal parameters
duration = st.number_input('Duration (s)', 0.1, 5.0, 1.0)
fs = st.number_input('Sampling Rate (Hz)', 100.0, 5000.0, 1000.0)
t = np.linspace(0, duration, int(fs*duration), endpoint=False)

if time_or_laplace == 'Time-domain':
    st.header('1. Generate Signal (Time-domain)')
    signal_type = st.selectbox('Signal type', ['Sine', 'Ramp', 'Unit Step', 'Square', 'Impulse', 'Triangular'])
    freq = st.number_input('Frequency (Hz)', 1.0, 1000.0, 5.0)
    ampl = st.number_input('Amplitude', 0.1, 10.0, 1.0)
    if signal_type == 'Sine':
        signal = ampl * np.sin(2 * np.pi * freq * t)
        freq_components = [freq]
    elif signal_type == 'Ramp':
        signal = ampl * t
        freq_components = [0]
    elif signal_type == 'Unit Step':
        signal = ampl * np.ones_like(t)
        signal[:int(0.1*len(t))] = 0
        freq_components = [0]
    elif signal_type == 'Square':
        signal = ampl * square(2 * np.pi * freq * t)
        freq_components = [freq]
    elif signal_type == 'Impulse':
        signal = np.zeros_like(t)
        signal[0] = ampl
        freq_components = [0]
    else:  # Triangular
        signal = ampl * sawtooth(2 * np.pi * freq * t, 0.5)
        freq_components = [freq]
    signal = safe_numeric_array(signal, t)
    st.line_chart(signal)
else:
    st.header('1. Generate Signal (Laplace-domain)')
    st.info("Note: Use '*' for multiplication in Laplace expressions (e.g., '1/(2*s+1)').")
    laplace_expr = st.text_input('Enter Laplace-domain expression (in s):', '1/(s+1)')
    ampl = st.number_input('Amplitude', 0.1, 10.0, 1.0, key='ampl_laplace')
    s, t_sym = sp.symbols('s t', real=True)
    try:
        laplace_sym = ampl * sp.sympify(laplace_expr)
        time_expr = sp.inverse_laplace_transform(laplace_sym, s, t_sym)
        # Custom DiracDelta handler
        def dirac_delta_numeric(x):
            # Returns 1 at t=0, 0 elsewhere (discrete impulse)
            return np.isclose(x, 0, atol=1e-8).astype(float)
        modules = [{'DiracDelta': dirac_delta_numeric}, 'numpy']
        time_func = sp.lambdify(t_sym, time_expr, modules=modules)
        signal = time_func(t)
        if isinstance(signal, float) or np.isscalar(signal):
            signal = np.ones_like(t) * signal
        freq_components = [0]  # Not easily determined from Laplace, so set to 0
        st.write('Time-domain expression:', time_expr)
        signal = safe_numeric_array(signal, t)
        st.line_chart(signal)
    except Exception as e:
        st.error(f'Error in Laplace expression: {e}')
        signal = np.zeros_like(t)
        freq_components = [0]

# --- Ensure signal is a float array for all further processing ---
try:
    signal = np.asarray(signal, dtype=float)
except Exception:
    # Try to convert each value individually
    try:
        signal = safe_numeric_array(signal, t)
        st.warning('Some values in the signal were symbolic and have been numerically evaluated. If you see unexpected results, try a simpler Laplace expression.')
    except Exception as e:
        st.error(f'Could not convert signal to float array: {e}. Using zero signal.')
        signal = np.zeros_like(t)

# SystemProperties demo
st.header('2. System Properties')
def example_system(x):
    return 2 * x  # Example: scaling system
x1 = signal
x2 = np.roll(signal, 1)
a1 = 1.5
a2 = -0.7
try:
    st.write('Linearity:', SystemProperties.is_linear(example_system, x1, x2, a1, a2))
    st.write('Time-Invariance:', SystemProperties.is_time_invariant(example_system, x1, 2))
    st.write('Causality:', SystemProperties.is_causal(example_system, x1))
    st.write('Stability:', SystemProperties.is_stable(example_system, x1))
except Exception as e:
    st.error(f'Error in system property checks: {e}')

# SignalProcessor demo
st.header('3. Add Noise & Filter')
noise_snr = st.slider('Noise SNR (dB)', 0, 40, 20)
noisy_signal = SignalProcessor.add_noise(signal, snr_db=noise_snr)
st.line_chart(noisy_signal)
filter_type = st.selectbox('Filter type', ['Low-pass', 'High-pass', 'Band-pass'])
if filter_type == 'Low-pass':
    cutoff = st.slider('Low-pass cutoff (Hz)', 1, int(fs//2), 5)
    filtered = SignalProcessor.lowpass_filter(noisy_signal, cutoff, fs)
elif filter_type == 'High-pass':
    cutoff = st.slider('High-pass cutoff (Hz)', 1, int(fs//2), 5)
    filtered = SignalProcessor.highpass_filter(noisy_signal, cutoff, fs)
else:
    lowcut = st.slider('Band-pass low cutoff (Hz)', 1, int(fs//2)-1, 2)
    highcut = st.slider('Band-pass high cutoff (Hz)', lowcut+1, int(fs//2), 10)
    filtered = SignalProcessor.bandpass_filter(noisy_signal, lowcut, highcut, fs)
st.line_chart(filtered)

# Sampler demo
st.header('4. Sampling & Aliasing')
fs_sample = st.slider('Sampling Frequency (Hz)', 10, int(fs), int(fs//4))
sampled_signal, t_sampled = Sampler.sample(signal, fs, fs_sample)
st.line_chart(sampled_signal)
aliased = Sampler.detect_aliasing(signal, fs, fs_sample, freq_components)
st.write('Aliasing Detected:' if aliased else 'No Aliasing Detected')
fs_recon = st.slider('Reconstruction Rate (Hz)', int(fs_sample), int(fs), int(fs))
recon_signal, t_recon = Sampler.reconstruct(sampled_signal, t_sampled, fs_recon, duration)
st.line_chart(recon_signal) 