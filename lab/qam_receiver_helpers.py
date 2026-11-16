"""
Helper Code
Students may use these as black-box helpers. Their implementation is
not part of the assessment.
Functions:
- recover_symbols(rx_passband, fc, fs, samples_per_symbol)
- add_passband_awgn(signal, tx_symbols, samples_per_symbol, esn0_db, seed=2026)
"""
import numpy as np


def recover_symbols(rx_passband, fc, fs, samples_per_symbol):
    """Recover complex I+jQ symbol estimates from a real passband waveform.
    Assumes perfectly aligned symbols/carrier, and a whole number of
    carrier cycles per symbol. Returns one complex value per symbol.
    """
    signal = np.asarray(rx_passband, dtype=float)
    ns = int(samples_per_symbol)
    if ns <= 0 or signal.size % ns != 0:
        raise ValueError("Passband length must be a multiple of samples_per_symbol")
    t = np.arange(ns, dtype=float) / fs
    frames = signal.reshape(-1, ns)
    i = 2 * np.mean(frames * np.cos(2*np.pi*fc*t), axis=1)
    q = -2 * np.mean(frames * np.sin(2*np.pi*fc*t), axis=1)
    return i + 1j*q


def add_passband_awgn(signal, tx_symbols, samples_per_symbol, esn0_db, seed=2026):
    """Add real passband AWGN at a target symbol-energy-to-noise ratio.
    E_s/N_0 (dB) refers to the recovered complex constellation after
    coherent demodulation, not raw sample-level SNR.
    """
    signal = np.asarray(signal, dtype=float)
    symbols = np.asarray(tx_symbols, dtype=complex)
    if symbols.size == 0:
        raise ValueError("tx_symbols must not be empty")
    ns = int(samples_per_symbol)
    if ns <= 0 or signal.size != symbols.size*ns:
        raise ValueError("Expect exactly samples_per_symbol samples per tx symbol")
    es = float(np.mean(np.abs(symbols)**2))
    snr = 10 ** (float(esn0_db) / 10)
    variance = ns * es / (4 * snr)
    rng = np.random.default_rng(seed)
    return signal + rng.normal(0, np.sqrt(variance), size=signal.shape)
