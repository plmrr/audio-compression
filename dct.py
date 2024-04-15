import numpy as np
import matplotlib.pyplot as plt


def generate_signal(signal_type='sine', frequency=5, amplitude=1, sample_rate=100):
    t = np.linspace(0, 1, num=sample_rate, endpoint=False)
    t = np.linspace(0, 1, num=sample_rate, endpoint=False)
    if signal_type == 'sine':
        return amplitude * np.sin(2 * np.pi * frequency * t)
    elif signal_type == 'square':
        return amplitude * np.sign(np.sin(2 * np.pi * frequency * t))
    elif signal_type == 'triangle':
        return amplitude * 2 * np.abs(2 * (t * frequency - np.floor(t * frequency + 0.5))) - 1
    elif signal_type == 'sawtooth':
        return amplitude * (2 * (t * frequency - np.floor(t * frequency)) - 1)
    elif signal_type == 'random':
        return np.random.randn(sample_rate)
    else:
        raise ValueError(
            "Unsupported signal type. Available types: 'sine', 'square', 'triangle', 'sawtooth', 'random'.")


def measure_parameters(signal):
    mean = np.mean(signal)
    variance = np.var(signal)
    peak_to_peak = np.ptp(signal)
    return mean, variance, peak_to_peak


def dct_from_scratch(signal):
    N = len(signal)
    X = np.zeros(N)
    for k in range(N):
        sum_val = 0
        for n in range(N):
            sum_val += signal[n] * np.cos(np.pi * k * (n + 0.5) / N)
        X[k] = sum_val
    return X


def plot_signals(original, transformed, sample_rate):
    fig, axs = plt.subplots(2, 1, figsize=(10, 8))

    # Plot the original signal
    axs[0].plot(np.linspace(0, 1, num=sample_rate, endpoint=False), original)
    axs[0].set_title('Original Signal')
    axs[0].set_xlabel('Time [s]')
    axs[0].set_ylabel('Amplitude')

    # Plot the DCT of the signal
    axs[1].plot(transformed)
    axs[1].set_title('DCT of the Signal')
    axs[1].set_xlabel('Frequency Bins')
    axs[1].set_ylabel('Magnitude')

    plt.tight_layout()
    plt.show()


def main():
    # Parameters
    sample_rate = 1000  # samples per second
    signal_type = 'square'  # or 'sine' 'triangle' 'square' 'sawtooth' 'random'
    frequency = 200  # Hz, only used for  wave
    amplitude = 1  # Amplitude of the  wave
    
    sample_rate = int(input("Please enter sample rate : "))
    signal_type = input(
        "Please enter signal type ('sine' 'triangle' 'square' 'sawtooth' 'random') : ")
    frequency = int(input("Please enter frequency : "))
    amplitude = int(input("Please enter amplitude : "))
 
    # Generate signal
    signal = generate_signal(signal_type,
                             frequency, amplitude, sample_rate)

    # Perform DCT using the custom function
    signal_dct = dct_from_scratch(signal)

    # Measure parameters before DCT
    original_params = measure_parameters(signal)
    print(
        "Original Signal Parameters: Mean = {:.2f}, Variance = {:.2f}, Peak-to-Peak = {:.2f}".format(*original_params))

    # Measure parameters after DCT
    dct_params = measure_parameters(signal_dct)
    print(
        "DCT Signal Parameters: Mean = {:.2f}, Variance = {:.2f}, Peak-to-Peak = {:.2f}".format(*dct_params))

    # Plot the signals
    plot_signals(signal, signal_dct, sample_rate)


if __name__ == "__main__":
    main()
