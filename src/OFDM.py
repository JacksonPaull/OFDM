"""
This file holds implementations for an OFDM transmitter receiver pair


"""
from scipy.fft import fft, ifft
import waterfilling
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
                 v,
                 power_allocations):
        """
        Parameters:
            N: Number of subcarriers
            v: cycle_prefix length
            power_allocations: power allocated to each subchannel
                indexing is assumed to be done correctly 
        """

        self.N = N
        self.v = v
        self.pam = encoders.MPAM(2)
        self.qpsk = encoders.MQAM(4)
        self.power_allocations = power_allocations
        self.n_zero_power = (power_allocations == 0).sum()
        self.zp_idx = np.arange(N)[self.power_allocations == 0]
        
    def encode_bits(self, bits:np.array):
        assert(len(bits) == self.N - self.n_zero_power)

        signal = np.zeros(self.N) + 0j
        b = 0
        for i in range(self.N//2 + 1): # For every channel index
            if i in [0, self.N//2] and i not in self.zp_idx:
                # Real channel, consume 1 bit
                signal[i] = self.pam.encode_bits(bits[b]).squeeze()
                b += 1
            elif i not in self.zp_idx:
                signal[i] = self.qpsk.encode_bits(bits[[b, b+1]]).squeeze()
                b += 2

        # Note: Because there are exactly as many bits as should be transmitted, and the nyquist is encoded last
        #       We are left with the nice property that the "real" bits in the actual bitstream are those on the
        #       edge, and all other bits are complex symbol pairs that are right next to each other
        
        # Enforce baseband complex conjugate
        signal[self.N//2+1:] = np.conj(signal[1:self.N//2])[::-1]
        return signal # * self.power_allocations

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
    
    def __init__(self, v, p, N, optimal_power):
        self.v = v
        self.N = N

        Pofdm = utils.construct_Pofdm(p, N)
        e = np.diag(fft(np.eye(N)) @ Pofdm @ ifft(np.eye(N)))
        self.eigvals = np.concatenate((e[N//2:], e[:N//2]))
        self.pam = encoders.MPAM(2)
        self.qpsk = encoders.MQAM(4)
        
        self.opt_tx_power = optimal_power
        self.n_zero_power = (self.opt_tx_power == 0).sum()

        self.zp_idx = np.arange(N)[self.opt_tx_power == 0]


    def remove_cp(self, X):
        return X[self.v:]
    
    def equalize(self, z):
        return z / (self.eigvals + 1e-15) # Equalize and avoid divide by zero errors
    
    def decode_bits(self, X):
        assert len(X) == self.N


        bits = []
        for i in range(self.N//2 + 1): # For every channel index
            if i in [0, self.N//2] and i not in self.zp_idx:
                # Real channel, consume 1 bit
                bits = bits + list(self.pam.decode_symbols(X[[i]]))

            elif i not in self.zp_idx:
                bits = bits + list(self.qpsk.decode_symbols(X[[i]]))


        return np.array(bits)


    def __call__(self, X, return_cache=False):
        x = self.remove_cp(X)                       # Remove CP
        z = fft(x)                                  # DFT
        z_eq = self.equalize(z)                     # Equalize
        b_hat = self.decode_bits(z_eq)              # estimate

        if return_cache:
            return b_hat, {'x':x, 'z':z, 'z_eq':z,'b_hat':b_hat}

        return b_hat
