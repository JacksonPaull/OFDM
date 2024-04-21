import numpy as np
from scipy.linalg import toeplitz
from scipy import special

def generate_bits(N, ret='array'):
    """
    Generate N random bits with an equal probability of 0s and 1s.

    Note: Assuming some form of optimal data encoding, then the bitstream 
    itself will have roughly equal bit probability
    """

    bits = np.array([1 if np.random.rand() < 0.5 else 0 for _ in range(N)])
    if ret == 'array':
        return bits
    elif ret == 'string' or ret == 'str':
        return ''.join([str(i) for i in bits])
    else:
         raise ValueError(f"Invalid return type '{ret}' requested. Valid types are 'array' and 'string'")

def probability_of_symbol_error(sent_symbols: np.array, 
                                received_symbols: np.array):
    assert len(sent_symbols) == len(received_symbols)
    return np.sum(sent_symbols != received_symbols) / len(sent_symbols)

def probability_of_bit_error(sent_bits: np.array, 
                             received_bits: np.array):
    assert len(sent_bits) == len(received_bits)
    return np.sum(sent_bits != received_bits) / len(sent_bits)

def encode_2PAM(bits: np.array):
        return 2*(bits-0.5)

def encode_QPSK(bits: np.array):
    assert len(bits) % 2 == 0
    bits = bits.reshape((-1,2))
    return 2*(bits-0.5) @ np.array([1, 1j])


def decode_2PAM(symbols):
        # ML Detection
        return (symbols > 0).astype(int)

def decode_QPSK(symbols):
    bits = np.apply_along_axis(lambda s: np.array([np.real(s) > 0, np.imag(s) > 0]), 0, symbols).astype(int)
    return bits.T.flatten()

def construct_P(p, N):
    P = toeplitz(np.concatenate((p, np.zeros(N-1))), np.concatenate((p[0:1], np.zeros(N-1))))
    return P

def P_to_Pofdm(P):
    # P = (N+L, N)
    L, N = P.shape
    L -= N 

    arrs = [P[N:, :]]
    for _ in range(N-1):
        arrs.append(np.zeros(arrs[0].shape))

    Pofdm = P[:N,:] + np.row_stack(arrs)

    return Pofdm

def invQfunc(x):
    return np.sqrt(2)*special.erfinv(1-2*x)

def qfunc(x):
    return 0.5-0.5*special.erf(x/np.sqrt(2))

def lin_to_db(x):
    return 10*np.log10(x)

def db_to_lin(x):
    return 10**(x/10)

def pe_to_snr(pe):
    return invQfunc(pe /2) ** 2