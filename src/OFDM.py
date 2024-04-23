"""
This file holds implementations for an OFDM transmitter receiver pair


"""
from scipy.fft import fft, ifft
from scipy.linalg import eigvals
import numpy as np
import utils
import encoders

class OFDM_transmitter():

    def __init__(self, 
                 v):
        """
        v: cycle_prefix length
        fc: carrier frequency (0=baseband) (unused)
        B: total bandwidth (unused)
        """

        self.v = v
        self.pam = encoders.MPAM(2)
        self.qpsk = encoders.MQAM(4)
        
    def encode_bits(self, X:np.array):
        # Note: x[0], x[N/2] --> baseband (for fc=0)
        #       x[1:N/2] --> complex positive indices
        #       x[N/2:] --> complex negative indices
        # ==> i = N/2 - i if i > N/2
        

        baseband = self.pam.encode_bits(X[[0]])
        nyquist = self.pam.encode_bits(X[[1]])

        complex_symbols = self.qpsk.encode_bits(X[2:])

        signal = np.concatenate([baseband, 
                                 complex_symbols, 
                                 nyquist, 
                                 np.conj(complex_symbols[::-1])])
        return signal

    def add_cp(self, X):
        return np.concatenate([X[-self.v:], X])

    def __call__(self, b, return_cache=False):
        """
        b: bits
        """
        symbols = self.encode_bits(b)
        x = ifft(symbols)
        signal = self.add_cp(x)  # Add CP
       
        assert(np.sum(np.abs(np.imag(signal))) == 0)

        if return_cache:
            return signal, {'symbols':symbols, 'x':x, 'signal':signal}
        return signal



class OFDM_receiver():
    
    def __init__(self, v, H, N):
        self.v = v
        self.N = N
        #TODO
        # Shouldn't the eigenvalues for Pofdm be the equalizer values?
        # Empirically, using the eigenvalues associated with each channel skyrockets Pe
        
        # P = utils.construct_P(H, N)
        # Pofdm = utils.P_to_Pofdm(P)
        # e = np.sort(np.abs(eigvals(Pofdm)))[::-1]
        # self.eigvals = np.concatenate([e[[0]], e[1:N-1:2], e[[N-1]], e[N-2:1:-2]])
        self.eigvals = fft(H, N)+1e-10 # 1 tap frequency EQ, avoid  divide by 0
        self.pam = encoders.MPAM(2)
        self.qpsk = encoders.MQAM(4)

    def remove_cp(self, X):
        return X[self.v:]
    
    def equalize(self, z):
        return z / self.eigvals
    
    def decode_bits(self, X):
        assert len(X) == self.N

        pam = self.pam.decode_symbols(X[[0, self.N//2]])
        qpsk = self.qpsk.decode_symbols(X[1:self.N//2])
        bits = np.concatenate([pam, qpsk])

        return bits

    def __call__(self, X, return_cache=False):
        x = self.remove_cp(X)                       # Remove CP
        z = fft(x)                                  # DFT
        # Note: Because the equalizer is only a scalar multiplication, 
        # it doesn't change the decision boundary for any symbol detection for
        # the given encoding scheme???
        z_eq = self.equalize(z)
        b_hat = self.decode_bits(z_eq)                 # estimate

        if return_cache:
            return b_hat, {'x':x, 'z':z, 'z_eq':z_eq,'b_hat':b_hat}

        return b_hat
