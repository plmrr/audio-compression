from tools import generate_signal, measure_parameters, plot_signals, compress_signal, mean_squared_error
from algorithms import dct_from_scratch, idct_from_scratch, dft, idft
import numpy as np


def main():
    # Parameters
    sample_rate = 1000  # samples per second
    signal_type = 'square'  # or 'sine' 'triangle' 'square' 'sawtooth' 'random'
    frequency = 200  # Hz, only used for wave
    amplitude = 1  # Amplitude of the wave
    method = 'DFT' # Method for processing signal
    threshold_percent = 0.3 # Threshold for compression
    
    # Generate signal
    signal = generate_signal(signal_type, frequency, amplitude, sample_rate)

    # For later, now use constant values for testing
    """
    sample_rate = int(input("Please enter sample rate : "))
    signal_type = input("Please enter signal type ('sine' 'triangle' 'square' 'sawtooth' 'random') : ")
    frequency = int(input("Please enter frequency : "))
    amplitude = int(input("Please enter amplitude : "))
    method = input("Please choose method ('DCT', 'DFT'): ").upper()
    threshold_percent = float(input("Please enter compression threshold in %: ")) / 100
    """

    # Process signal
    if method == 'DCT':
        transformed_signal = dct_from_scratch(signal)
        threshold_value = threshold_percent * np.max(np.abs(transformed_signal))
        compressed_signal = compress_signal(transformed_signal, threshold_value)
        reconstructed_signal = idct_from_scratch(compressed_signal)
    elif method == 'DFT':
        transformed_signal = dft(signal)
        threshold_value = threshold_percent * np.max(np.abs(transformed_signal))
        compressed_signal = compress_signal(transformed_signal, threshold_value)
        reconstructed_signal = idft(compressed_signal)
        pass
    else:
        raise ValueError("Invalid method! Available methods: 'DCT', 'DFT'.")

    # Measure parameters before transformation
    original_params = measure_parameters(signal)
    print("Original signal parameters: Mean = {:.2f}, Variance = {:.2f}, Peak-to-Peak = {:.2f}".format(*original_params))

    # Measure parameters after transformation
    transformed_params = measure_parameters(transformed_signal)
    print(f"{method} signal parameters: Mean = {transformed_params[0]:.2f}, Variance = {transformed_params[1]:.2f}, Peak-to-Peak = {transformed_params[2]:.2f}")

    # Measure parameters after compression
    compressed_params = measure_parameters(compressed_signal)
    print(f"Compressed {method} signal parameters: Mean = {compressed_params[0]:.2f}, Variance = {compressed_params[1]:.2f}, Peak-to-Peak = {compressed_params[2]:.2f}")

    # Measure parameters after reconstruction
    reconstructed_params = measure_parameters(reconstructed_signal)
    print(f"Reconstruced signal parameters: Mean = {reconstructed_params[0]:.2f}, Variance = {reconstructed_params[1]:.2f}, Peak-to-Peak = {reconstructed_params[2]:.2f}")

    # Measure MSE 
    mse_value = mean_squared_error(signal, reconstructed_signal)
    print(f"Mean Squared Error (MSE): {mse_value:.4f}")

    # Plot the signals
    plot_signals(signal, transformed_signal, compressed_signal, reconstructed_signal, sample_rate, method)


if __name__ == "__main__":
    main()