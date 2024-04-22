"""
This file holds implementations for an OFDM transmitter receiver pair


"""
from scipy.fft import fft, ifft
from scipy.linalg import eigvals
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
        
        baseband = utils.encode_2PAM(X[[0]])
        nyquist = utils.encode_2PAM(X[[1]])

        complex_symbols = utils.encode_QPSK(X[2:])

        signal = np.concatenate([baseband, 
                                 complex_symbols, 
                                 nyquist, 
                                 np.conj(complex_symbols[::-1])])
        return signal

    def add_cp(self, X):
        return np.concatenate([X[-self.v:], X])

    def __call__(self, b):
        """
        b: bits
        """
        x = self.encode_bits(b)
        x = ifft(x)
        x = self.add_cp(x)  # Add CP
       
        return x



class OFDM_receiver():
    
    def __init__(self, v, H, N):
        self.v = v
        self.N = N
        #P = utils.construct_P(H, N)
        #Pofdm = utils.P_to_Pofdm(P)
        #self.eigvals = np.sort(np.abs(eigvals(Pofdm)))[::-1]

    def remove_cp(self, X):
        return X[self.v:]
    
    def decode_bits(self, X):
        assert len(X) == self.N

        # Note: Because the equalizer is only a scalar multiplication, 
        # it doesn't change the decision boundary for any symbol detection for
        # the given encoding scheme
        pam = utils.decode_2PAM(X[[0, self.N//2]]) # / self.eigvals[[0, self.N-1]])
        qpsk = utils.decode_QPSK(X[1:self.N//2]) # / self.eigvals[1:-1:2]) # Lambda_i
        bits = np.concatenate([pam, qpsk])

        return bits

    def __call__(self, X):
        x = self.remove_cp(X)                       # Remove CP
        z = fft(x)                                  # DFT
        b_hat = self.decode_bits(z)                 # Equalize and estimate

        return b_hat
