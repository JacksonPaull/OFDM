"""
This file holds implementations for an OFDM transmitter receiver pair


"""
from scipy.fft import fft, ifft
from scipy.linalg import toeplitz, eigvals
import numpy as np
import utils

class OFDM_transmitter():

    def __init__(self, 
                 v):
        """
        v: cycle_prefix length
        fc: carrier frequency (0=baseband) (unused)
        B: total bandwidth (unused)
        """

        self.v = v
        
    def encode_bits(self, X:np.array):
        # Note: x[0], x[N/2] --> baseband (for fc=0)
        #       x[1:N/2] --> complex positive indices
        #       x[N/2:] --> complex negative indices
        # ==> i = N/2 - i if i > N/2
        N = len(X)
        real_bits = X[1:N//2]
        imag_bits = X[-1:N//2 :-1]

        complex_bits = np.column_stack((real_bits, imag_bits)).flatten()
        complex_symbols = utils.encode_QPSK(complex_bits)
        
        baseband = utils.encode_2PAM(X[0])
        nyquist = utils.encode_2PAM(X[N//2])

        complex_symbols = np.insert(complex_symbols, 0, baseband)
        complex_symbols = np.append(complex_symbols, nyquist)
        return complex_symbols

    def add_cp(self, X):
        return np.concatenate([X[-self.v:], X])

    def __call__(self, b):
        """
        b: bits
        """
        x = self.encode_bits(b)
        x = ifft(x)


        x = self.add_cp(x)  # Add CP
       
        # (Convert digital to analog, this can be skipped because we assume perfect sampling i.e. the pulse shaping and then sampling is ideal )
        # (Multiply by complex carrier, ignored because we are at baseband and assuming perfect pulse shaping)
        return x



class OFDM_receiver():
    
    def __init__(self, v, H, N):
        self.v = v
        self.N = N
        P = utils.construct_P(H, N)
        Pofdm = utils.P_to_Pofdm(P)
        self.eigvals = np.sort(np.abs(eigvals(Pofdm)))[::-1]

    def remove_cp(self, X):
        return X[self.v:]
    
    def decode_bits(self, X):
        assert len(X) == self.N
        pam = utils.decode_2PAM(X[[0, -1]] / self.eigvals[[0, -1]])
        qpsk = utils.decode_QPSK(X[1:-1] / self.eigvals[1:-1]) # Lambda_i

        real_bits = qpsk[0::2]
        imag_bits = qpsk[-1:0:-2]

        bits = np.concatenate([pam[[0]], real_bits, pam[[1]], imag_bits])

        return bits

    def __call__(self, X):
        x = self.remove_cp(X)                       # Remove CP
        z = fft(x)                                  # DFT
        b_hat = self.decode_bits(z)                 # Equalize and estimate

        return b_hat
