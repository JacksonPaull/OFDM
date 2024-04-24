
import numpy as np
import matplotlib.pyplot as plt

class Encoder():
    def __init__(self):
        # TODO Add general initialization (anything that is shared between all encoders)
        raise NotImplementedError()

    def encode_bits(self, bits):
        raise NotImplementedError()

    def decode_symbols(self, symbols):
        raise NotImplementedError()

    def plot_constellation(self, fsize=(5,5), show=True):
        fw, fh = fsize
        fig, ax = plt.subplots()
        fig.suptitle(f'{self.name} Constellation (assume gray coding)')
        fig.set_figwidth(fw)
        fig.set_figheight(fh)
        ax.set_xlabel('Real Part (I)')
        ax.set_ylabel('Imaginary Part (Q)')
        ax.grid(True, ls='--', lw=0.3)

        # Construct sequence of all bits we need
        bits = ''
        symbol_len = np.log2(self.M)
        for i in np.arange(self.M):
            b = bin(i)[2:]
            b = '0'*int(symbol_len-len(b)) + b
            bits += b

        symbols = self.encode_bits(np.array([int(i) for i in bits]))

        x, y = np.real(symbols), np.imag(symbols)
        for i, bits in enumerate(symbols):
            ax.plot(x[i], y[i],'bo')
            ax.text(x[i], y[i]+0.1, f'$X_{i}$', ha='center')
        
        if show:
            fig.show()
            return
        
        return fig, ax

    def plot_decoding(self, symbols, fsize=(5,5), show=True):
        # Plot a decoding of the constellation into the relevant bits
        fw, fh = fsize
        fig, ax = plt.subplots()
        fig.suptitle(f'{self.name} Constellation (assume gray coding)')
        fig.set_figwidth(fw)
        fig.set_figheight(fh)
        ax.set_xlabel('Real Part (I)')
        ax.set_ylabel('Imaginary Part (Q)')
        ax.grid(True, ls='--', lw=0.3)

        # Construct sequence of all bits we need
        bits = ''
        symbol_len = np.log2(self.M)
        for i in np.arange(self.M):
            b = bin(i)[2:]
            b = '0'*int(symbol_len-len(b)) + b
            bits += b

        x, y = np.real(symbols), np.imag(symbols)
        ax.scatter(x, y, c='orange')

        # Plot reference of constellation
        symbols = self.encode_bits(np.array([int(i) for i in bits]))
        x, y = np.real(symbols), np.imag(symbols)
        for i, bits in enumerate(symbols):
            ax.plot(x[i], y[i],'bo')
            ax.text(x[i], y[i]+0.1, f'$X_{i}$', ha='center')
        
        if show:
            fig.show()
            return

        raise NotImplementedError()


class MPAM(Encoder):
    def __init__(self, M):
        if M != 2:
            raise NotImplementedError("Haven't implemented anything except 2PAM (for hw9)")
        self.M = M
        self.name = f'{M} PAM'

    def encode_bits(self, bits):
        return (bits-0.5) * 2

    def decode_symbols(self, symbols):
        return (symbols > 0).astype(int)
    
    def plot_constellation(self, fsize=(5,5), show=True):
        fig, ax = super().plot_constellation(fsize, False)
        ax.set_ylim((-1, 1))

        if show:
            fig.show()
            return

        return fig, ax

    def plot_decoding(self,):
        # Plot a decoding of the constellation into the relevant bits
        pass


class MQAM(Encoder):
    def __init__(self, M):
        if M != 4:
            raise NotImplementedError("Haven't implemented anything except QPSK (for hw9)")
        
        self.M = M
        self.name = f'{M} QAM'

    def encode_bits(self, bits):
        assert len(bits) % 2 == 0
        bits = bits.reshape((-1,2))

        # Multiplied by root 2 to normalize signal energy
        return (bits-0.5) @ np.array([1, 1j]) * np.sqrt(2)

    def decode_symbols(self, symbols):
        bits = np.apply_along_axis(lambda s: np.array([np.real(s) > 0, np.imag(s) > 0]), 0, symbols).astype(int)
        return bits.T.flatten()
