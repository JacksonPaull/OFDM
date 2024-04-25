"""
This file holds implementations for an OFDM transmitter receiver pair


"""
from scipy.fft import fft, ifft
import numpy as np
import utils
import encoders

class OFDM_transmitter():
    """
    An implementation for an OFDM transmitter.

    Current limitations:
        baseband only
        stationary channel assumption
        limited by the limited encoder implementations
    """

    def __init__(self, 
                 N,
                 v):
        """
        Parameters:
            N: Number of subcarriers
            v: cycle_prefix length
        """

        self.N = N
        self.v = v
        self.pam = encoders.MPAM(2)
        self.qpsk = encoders.MQAM(4)
        
    def encode_bits(self, bits:np.array):
        assert(len(bits) == self.N) # An assumption dependent on using QPSK/2PAM
        signal = np.zeros(self.N, dtype=np.complex128)
        
        real_symbols = self.pam.encode_bits(bits[:2])
        complex_symbols = self.qpsk.encode_bits(bits[2:])
        
        signal[[0, self.N//2]] = real_symbols
        signal[1:self.N//2] = complex_symbols

        # Enforce baseband complex conjugate
        signal[self.N//2+1:] = np.conj(signal[1:self.N//2])[::-1]
        return signal

    def add_cp(self, X):
        return np.concatenate([X[-self.v:], X])

    def __call__(self, b, return_cache=False):
        """
        b: bits
        """
        symbols = self.encode_bits(b)       # Encode bits as symbols
        x = ifft(symbols)                   # Take the fft
        signal = self.add_cp(x)             # Add CP
        
        # The outputted signal should be real
        assert np.isclose(np.sum(np.imag(signal)), 0)

        if return_cache:
            return signal, {'symbols':symbols, 'x':x, 'signal':signal}
        return signal



class OFDM_receiver():
    """
    An implementation for a simple baseband OFDM receiver.

    Current limitations:
        - only baseband
        - limited by implemented PAM and QAM encoders
        ! stationary and known channel assumption (bad)
    """
    
    def __init__(self, N, v, p):
        self.v = v
        self.N = N

        Pofdm = utils.construct_Pofdm(p, N)
        self.eigvals = np.diag(fft(np.eye(N)) @ Pofdm @ ifft(np.eye(N)))
        self.pam = encoders.MPAM(2)
        self.qpsk = encoders.MQAM(4)


    def remove_cp(self, X):
        return X[self.v:]
    
    def equalize(self, z):
        return z / (self.eigvals + 1e-15) # Equalize and avoid divide by zero errors
    
    def decode_bits(self, X):
        assert len(X) == self.N
        
        real_bits = self.pam.decode_symbols(X[[0, self.N//2]])
        complex_bits = self.qpsk.decode_symbols(X[1:self.N//2])

        return np.concatenate((real_bits, complex_bits))


    def __call__(self, X, return_cache=False):
        x = self.remove_cp(X)                       # Remove CP
        z = fft(x)                                  # DFT
        z_eq = self.equalize(z)                     # Equalize
        b_hat = self.decode_bits(z_eq)              # estimate

        if return_cache:
            return b_hat, {'x':x, 'z':z, 'z_eq':z_eq,'b_hat':b_hat}

        return b_hat
