import numpy as np
from scipy.fft import fft, fftfreq

def power_features(voltage_samples,
           current_samples,
           sampling_rate=30e3,
           expected_freq_range=(40, 70)):

    N = len(voltage_samples)
     # Calculate RMS values for voltage and current
    voltage_rms = np.sqrt(np.mean(np.square(voltage_samples)))
    current_rms = np.sqrt(np.mean(np.square(current_samples)))

    voltage_fft = np.fft.fft(voltage_samples)
    current_fft = np.fft.fft(current_samples)

    # Fundamental frequency
    frequencies = fftfreq(N, 1 / sampling_rate)
    # Index
    # fundamental_index = np.argmax(np.abs(voltage_fft[:N // 2]))
    # fundamental_frequency = round(frequencies[fundamental_index])
    magnitudes = np.abs(voltage_fft)

    valid_indices = np.where((frequencies >= expected_freq_range[0]) & (frequencies <= expected_freq_range[1]))[0]
    fundamental_index = valid_indices[np.argmax(magnitudes[valid_indices])]
    fundamental_frequency: None = frequencies[fundamental_index]

    # Frequency resolution
    frequency_resolution = sampling_rate / N

    # Index of the fundamental frequency (assuming 50 Hz)
    fundamental_index = int(fundamental_frequency / frequency_resolution)

    # Extract the fundamental frequency component
    voltage_fundamental = voltage_fft[fundamental_index]
    current_fundamental = current_fft[fundamental_index]

    # Calculate phase angles and phase difference
    voltage_phase = np.angle(voltage_fundamental)
    current_phase = np.angle(current_fundamental)
    phase_difference = voltage_phase - current_phase  # np.degrees(phase_difference)

    # Calculate Active Power (P), Reactive Power (Q), and Apparent Power (S)
    active_power = voltage_rms * current_rms * np.cos(phase_difference)
    reactive_power = voltage_rms * current_rms * np.sin(phase_difference)
    apparent_power = np.sqrt(active_power ** 2 + reactive_power**2)

    # Calculate Power factor (PF)
    if apparent_power == 0:
        return 0  # To avoid division by zero
    power_factor = active_power / apparent_power

    # Calcualte power factor angle (PFA)
    power_factor_angle = np.degrees(np.arccos(power_factor))

    return active_power, reactive_power, apparent_power, power_factor, power_factor_angle


def harmonic_features(voltage_samples,
                               current_samples,
                               sampling_rate,
                               expected_freq_range=(40, 70)):

    N = len(current_samples)

    # Amplitude (Peak Value)
    amplitude = np.max(np.abs(current_samples))

    # RMS Value
    rms = np.sqrt(np.mean(np.square(current_samples)))

    # Crest factor
    crest_factor = amplitude / rms

    peak_area = np.sum(np.abs(current_samples)) / sampling_rate

    dc_component = np.mean(current_samples)

    current_fft = fft(current_samples)
    voltage_fft = fft(voltage_samples)

    magnitudes_current = np.abs(current_fft)
    magnitudes_voltage =  np.abs(voltage_fft)

    frequencies = np.round(fftfreq(N, 1 / sampling_rate))

    # fundamental_index = np.argmax(np.abs(current_fft[1:N // 2])) + 1
    # fundamental_frequency = round(frequencies[fundamental_index])
    valid_indices = np.where((frequencies >= expected_freq_range[0]) & (frequencies <= expected_freq_range[1]))[0]
    fundamental_index = valid_indices[np.argmax(magnitudes_voltage[valid_indices])]
    fundamental_frequency = frequencies[fundamental_index]

    fundamental_amplitude = np.abs(current_fft[fundamental_index])

    # harmonics
    harmonics = {}
    for n in [1, 3, 5, 7, 9, 11]:
        harmonic_index = np.where(frequencies == n * fundamental_frequency)[0][0]
        harmonic_amplitude = np.abs(current_fft[harmonic_index]) / (N / 2)
        harmonics[n] = harmonic_amplitude

    # THD
    total_harmonics = np.sqrt(np.sum(np.square(np.abs(current_fft[:N // 2]))) - np.square(fundamental_amplitude))
    thd = total_harmonics / fundamental_amplitude

    # Generate a time array
    current_waveform = current_samples[:int(sampling_rate/fundamental_frequency)]
    # print(current_waveform.shape)
    # current_waveform = current_samples
    t = np.arange(len(current_waveform)) / sampling_rate

    sine_wave = amplitude * np.sin(2 * np.pi * fundamental_frequency * t)

    # Compute the correlation coefficient
    correlation_matrix = np.corrcoef(current_waveform, sine_wave)
    correlation_coefficient = correlation_matrix[0, 1]

    return amplitude, rms, crest_factor, peak_area, dc_component, correlation_coefficient, harmonics, thd


def statis_features(current, voltage, fs):

    P, Q, S, PF, PFA = power_features(voltage, current, sampling_rate=fs, expected_freq_range=(40, 70))
    Iamp, Irms, Icf, w_s, dc, r, har, thd = harmonic_features(voltage, current, fs)

    # return [P, Q]
    # return [P, Q, Iamp]
    # return [P, Q, Iamp, Irms]
    # return [P, Q, Iamp, Irms, thd]
    # return [P, Q, Iamp, Irms, har[1], har[3], thd]
    # return [har[1], har[3], har[5], har[7], thd]
    # return [P, Irms, har[1], har[3], har[5], har[7], thd]
    return [P, Q, S, Iamp, Irms, har[1], har[3], har[5], har[7], thd]
