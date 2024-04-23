
import numpy as np

class Encoder():
    def __init__(self):
        # TODO Add general initialization (anything that is shared between all encoders)
        raise NotImplementedError()

    def encode_bits(bits):
        raise NotImplementedError()

    def decode_symbols(symbols):
        raise NotImplementedError()

    def plot_constellation():
        raise NotImplementedError()

    def plot_decoding():
        # Plot a decoding of the constellation into the relevant bits
        raise NotImplementedError()


class MPAM(Encoder):
    def __init__(self, M):
        if M != 2:
            raise NotImplementedError("Haven't implemented anything except 2PAM (for hw9)")

    def encode_bits(self, bits):
        return (bits-0.5) * 2

    def decode_symbols(self, symbols):
        return (symbols > 0).astype(int)

    def plot_constellation(self,):
        pass

    def plot_decoding(self,):
        # Plot a decoding of the constellation into the relevant bits
        pass


class MQAM(Encoder):
    def __init__(self, M):
        if M != 4:
            raise NotImplementedError("Haven't implemented anything except QPSK (for hw9)")
        
        self.M = M

    def encode_bits(self, bits):
        assert len(bits) % 2 == 0
        bits = bits.reshape((-1,2))

        # Multiplied by 2 Es to account for energy in both dimensions
        return (bits-0.5) @ np.array([1, 1j]) * 2

    def decode_symbols(self, symbols):
        bits = np.apply_along_axis(lambda s: np.array([np.real(s) > 0, np.imag(s) > 0]), 0, symbols).astype(int)
        return bits.T.flatten()

    def plot_constellation(self,):
        pass

    def plot_decoding(self,):
        # Plot a decoding of the constellation into the relevant bits
        pass