import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import librosa


def generate_signal(signal_type='sine', frequency=5, amplitude=1, sample_rate=1000, duration=1):
    t = np.linspace(0, duration, num=int(sample_rate * duration), endpoint=False)
    if signal_type == 'sine':
        return amplitude * np.sin(2 * np.pi * frequency * t)
    elif signal_type == 'square':
        return amplitude * np.sign(np.sin(2 * np.pi * frequency * t))
    elif signal_type == 'triangle':
        return amplitude * 2 * np.abs(2 * (t * frequency - np.floor(t * frequency + 0.5))) - 1
    elif signal_type == 'sawtooth':
        return amplitude * (2 * (t * frequency - np.floor(t * frequency)) - 1)
    elif signal_type == 'random':
        return np.random.randn(int(sample_rate * duration))
    else:
        raise ValueError(
            "Unsupported signal type. Available types: 'sine', 'square', 'triangle', 'sawtooth', 'random'.")


def measure_parameters(signal):
    mean = np.mean(signal)
    variance = np.var(signal)
    peak_to_peak = np.ptp(signal)
    return mean, variance, peak_to_peak


def plot_signals(original=None, transformed=None, compressed=None, reconstructed=None, sample_rate=None, method=None, mse=None, compression=None):
    if method in ('DCT', 'DFT'):
        fig, axs = plt.subplots(2, 2, figsize=(12, 10))
        axs = axs.flatten()

        t_original = np.linspace(0, len(original)/sample_rate, num=len(original), endpoint=False)
        t_reconstructed = np.linspace(0, len(reconstructed)/sample_rate, num=len(reconstructed), endpoint=False)

        # Original signal
        axs[0].plot(t_original, original, linewidth=0.7)
        axs[0].set_title('Original Signal', fontsize=10)
        axs[0].set_xlabel('Time [s]', fontsize=8)
        axs[0].set_ylabel('Amplitude', fontsize=8)

        # First half of the transformed signal (DCT/DFT)
        half_len = len(transformed) // 2
        axs[1].stem(np.abs(transformed[:half_len]))
        axs[1].set_title(f'{method} of the Signal')
        axs[1].set_xlabel('Frequency Bins', fontsize=8)
        axs[1].set_ylabel('Magnitude', fontsize=8)

        # First half of the compressed signal
        axs[2].stem(np.abs(compressed[:half_len]))
        axs[2].set_title(f'{method} Spectrum after Compression', fontsize=10)
        axs[2].set_xlabel('Frequency Bins', fontsize=8)
        axs[2].set_ylabel('Magnitude', fontsize=8)

        # Reconstructed signal after compression
        axs[3].plot(t_reconstructed, reconstructed, linewidth=0.7)
        axs[3].set_title(f'Reconstructed Signal, MSE: {mse}, Compression: {int(compression*100)}%', fontsize=10)
        axs[3].set_xlabel('Time [s]', fontsize=8)
        axs[3].set_ylabel('Amplitude', fontsize=8)

        plt.tight_layout()
        plt.savefig(f'{method}_plot.png')
        plt.show()
    else:
        fig, axs = plt.subplots(1, 2, figsize=(12, 10))
        axs = axs.flatten()

        t_original = np.linspace(0, len(original)/sample_rate, num=len(original), endpoint=False)
        t_reconstructed = np.linspace(0, len(reconstructed)/sample_rate, num=len(reconstructed), endpoint=False)

        # Original signal
        axs[0].plot(t_original, original, linewidth=0.7)
        axs[0].set_title('Original Signal', fontsize=10)
        axs[0].set_xlabel('Time [s]', fontsize=8)
        axs[0].set_ylabel('Amplitude', fontsize=8)

        # Interpolated signal after lagrange polynomials
        axs[1].plot(t_reconstructed, reconstructed, linewidth=0.7)
        axs[1].set_title(f'Interpolated signal, MSE: {mse}, Compression: {int(compression*100)}%', fontsize=10)
        axs[1].set_xlabel('Time [s]', fontsize=8)
        axs[1].set_ylabel('Amplitude', fontsize=8)

        plt.tight_layout()
        plt.savefig(f'{method}_plot.png')
        plt.show()


# Compressing signal
# A threshold specifies the minimum magnitude value that a signal component must have to be retained in the compressed signal.
def compress_signal(transformed_data, threshold):
    compressed_data = transformed_data.copy()
    compressed_data[np.abs(compressed_data) < threshold] = 0
    return compressed_data

# MSE
def mean_squared_error(original, reconstructed):
    return np.mean((original - reconstructed) ** 2)

# Loading wave file
def load_signal_from_file(file_path):
    signal, sample_rate = librosa.load(file_path, sr=None, mono=True)
    return sample_rate, signal

# Saving compressed signal
def save_compressed_signal(file_path, sample_rate, reconstructed_signal):
    max_val = np.max(np.abs(reconstructed_signal))
    if max_val == 0:
        print("Warning: Max value of the signal is 0. The file will contain only silence.")
        reconstructed_signal_normalized = np.zeros_like(reconstructed_signal, dtype=np.int16)
    else:
        # 16-bit int normalization
        reconstructed_signal_normalized = np.int16(reconstructed_signal / max_val * 32767)
    
    sf.write(file_path, reconstructed_signal_normalized, sample_rate)
    print(f"Compressed audio has been saved to {file_path}")


if __name__ == "__main__":
    print("Nothing to show here...")
