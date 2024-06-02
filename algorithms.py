import numpy as np


def dct_from_scratch(signal):
    N = len(signal)
    X = np.zeros(N)
    for k in range(N):
        sum_val = 0
        for n in range(N):
            sum_val += signal[n] * np.cos(np.pi * k * (n + 0.5) / N)
        X[k] = sum_val
    return X


def idct_from_scratch(X):
    N = len(X)
    x = np.zeros(N)

    for n in range(N):
        sum_val = X[0] / 2
        for k in range(1, N):
            sum_val += X[k] * np.cos(np.pi * k * (n + 0.5) / N)
        x[n] = sum_val * 2 / N
    return x

def dft(signal):
    N = len(signal)
    X = np.zeros(N, dtype=complex)
    
    for k in range(N):
        for n in range(N):
            angle = -2j * np.pi * k * n / N
            X[k] += signal[n] * np.exp(angle)
    return X


def idft(X):
    N = len(X)
    x = np.zeros(N, dtype=complex)
    
    for n in range(N):
        for k in range(N):
            angle = 2j * np.pi * k * n / N
            x[n] += X[k] * np.exp(angle)
    return x / N


def lagrange_approximation(signal, degree):
    N = len(signal)
    indices = np.linspace(0, N-1, degree+1).astype(int)
    approx_signal = lagrange_interpolation(indices, signal[indices], np.arange(N))
    return approx_signal

def lagrange_interpolation(x, y, xi):
    n = len(x)
    yi = np.zeros_like(xi, dtype=np.float64)
    for i in range(n):
        L = np.ones_like(xi, dtype=np.float64)
        for j in range(n):
            if i != j:
                L *= (xi - x[j]) / (x[i] - x[j])
        yi += y[i] * L
    return yi


if __name__ == "__main__":
    print("Nothing to show here...")
