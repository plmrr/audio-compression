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


if __name__ == "__main__":
    print("Nothing to show here...")
