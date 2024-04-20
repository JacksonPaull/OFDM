"""
This file holds implementations for an OFDM transmitter receiver pair


"""
from scipy.fft import fft, ifft
import numpy as np

class OFDM_transmitter():

    def __init__(self, 
                 v,
                 fc):
        """
        pulse_shape: function handle for the pulse shape to be used in the D/A converter
        v: cycle_prefix length
        """
        pass

    def __call__(self, X):
        """
        X: Complex exponential symbols (2PAM or QPSK)
        """
        x = ifft(X)
        # Note: x[0], x[N/2] --> baseband (for fc=0)
        #       x[1:N/2] --> complex positive indices
        #       x[N/2:] --> complex negative indices
        # ==> i = N/2 - i if i > N/2

        # Add CP
        # (Convert digital to analog, this can be skipped because we assume perfect sampling i.e. the pulse shaping and then sampling is ideal )
        # Multiply by complex carrier
        pass

    pass


class OFDM_receiver():
    
    pass