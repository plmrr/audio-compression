from tools import generate_signal, measure_parameters, plot_signals, compress_signal, mean_squared_error, load_signal_from_file, save_compressed_signal
from algorithms import dct_from_scratch, idct_from_scratch, dft, idft, lagrange_approximation, lagrange_interpolation
import numpy as np

def main():
    choice = input("Do you want to 'load' a signal from a file or 'generate' one? (load/generate): ").lower()
    if choice == 'generate':
        sample_rate = int(input("Please enter sample rate : "))
        signal_type = input("Please enter signal type ('sine' 'triangle' 'square' 'sawtooth' 'random') : ")
        frequency = int(input("Please enter frequency : "))
        amplitude = int(input("Please enter amplitude : "))
        signal = generate_signal(signal_type, frequency, amplitude, sample_rate)
    elif choice == 'load':
        file_path = input("Please enter the path to the audio file: ")
        sample_rate, signal = load_signal_from_file(file_path)
        if len(signal.shape) == 2:
            signal = signal.mean(axis=1)

    # Process signal
    method = input("Please choose method ('DCT', 'DFT', 'LAGRANGE'): ").upper()
    threshold_percent = float(input("Please enter compression threshold in %: ")) / 100
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
    elif method == 'LAGRANGE':
        degree = int(input("Please enter the degree of the Lagrange polynomial: "))
        transformed_signal = signal
        compressed_signal = lagrange_approximation(signal, degree)
        reconstructed_signal = compressed_signal
    else:
        raise ValueError("Invalid method! Available methods: 'DCT', 'DFT', 'LAGRANGE'.")

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

    # Save audio file if one was loaded
    if choice == 'load':
        save_compressed_signal('compressed_sample.wav', sample_rate, reconstructed_signal)

if __name__ == "__main__":
    main()

    # TODO
    # POPRAWKI:
    # - zaimplementowac wielomiany lagranga
    # - usunac aliasing (wykres do 1/2 sample rate) - Done
    # - opisac OFDM lepiej
    # - napisac cos o aliasingu
    # - dodac jeszcze jakis opis o kompresji
    # - dodac przypisy do bibliografii

    # SPRAWKO:
    # - wstep - nieformalny opis rozwazanego problemu, wraz z odniesieniemai literaturowymi
    # - opis konstrukcji i interpretacja algorytmow rozwazanych w realizacji projektu
    # - dyskusja przykladowych zastosowan omawianych technik
    # - realizacja i opis przykladow numerycznych algorytmow (wyniki)
    # - dyskusja wad i zalet rozwazanych technik

    # PREZKA:
    # - na temat technik i algorytmow omawianych w projekcie (dct, dft, wielomiany lagrangea)