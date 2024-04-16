import numpy as np
import matplotlib.pyplot as plt


def generate_signal(signal_type='sine', frequency=5, amplitude=1, sample_rate=100):
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


def plot_signals(original, transformed, compressed, reconstructed, sample_rate, method):
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))  # Ustawienie 4 wykresów w pionie
    axs = axs.flatten()

    t = np.linspace(0, 1, num=sample_rate, endpoint=False)  # Czas dla oryginalnego sygnału

    # Wykres oryginalnego sygnału
    axs[0].plot(t, original, linewidth=0.7)
    axs[0].set_title('Original Signal', fontsize=10)
    axs[0].set_xlabel('Time [s]', fontsize=8)
    axs[0].set_ylabel('Amplitude', fontsize=8)

    # Wykres transformowanego sygnału (DCT/DFT)
    axs[1].plot(np.abs(transformed) if method == 'DFT' else transformed)
    axs[1].set_title(f'{method} of the Signal')
    axs[1].set_xlabel('Frequency Bins', fontsize=8)
    axs[1].set_ylabel('Magnitude', fontsize=8)

    # Wykres sygnału po kompresji
    axs[2].stem(np.abs(compressed) if method == 'DFT' else compressed)
    axs[2].set_title(f'{method} Spectrum after Compression', fontsize=10)
    axs[2].set_xlabel('Frequency Bins', fontsize=8)
    axs[2].set_ylabel('Magnitude', fontsize=8)

    # Wykres sygnału po rekonstrukcji
    axs[3].plot(t, reconstructed, linewidth=0.7)
    axs[3].set_title('Reconstructed Signal', fontsize=10)
    axs[3].set_xlabel('Time [s]', fontsize=8)
    axs[3].set_ylabel('Amplitude', fontsize=8)

    plt.tight_layout()
    plt.show()


def compress_signal(transformed_data, threshold):
    compressed_data = transformed_data.copy()
    compressed_data[np.abs(compressed_data) < threshold] = 0
    return compressed_data


def mean_squared_error(original, reconstructed):
    return np.mean((original - reconstructed) ** 2)


if __name__ == "__main__":
    print("Nothing to show here...")
