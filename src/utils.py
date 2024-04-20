import numpy as np

def generate_bits(N, ret='array'):
    bits = np.array([1 if np.random.rand() < 0.5 else 0])
    if ret == 'array':
        return bits
    elif ret == 'string':
        return ''.join(bits)

def probability_of_symbol_error(sent_symbols: np.array, 
                                received_symbols: np.array):
    assert len(sent_symbols) == len(received_symbols)
    return np.sum(sent_symbols == received_symbols) / len(sent_symbols)

def probability_of_bit_error(sent_bits: str, 
                             received_bits: str):
    assert len(sent_bits) == len(received_bits)
    np.sum(np.array([c for c in sent_bits]) == np.array([c for c in received_bits])) / len(sent_bits)
    pass

def encode_2PAM(bits: str):
        assert len(bits) == 1

        return (int(bits) - 0.5) * 2

def encode_QPSK(bits: str):
    assert len(bits) == 2
    return (int(bits[0]) - 0.5) * 2 + 1j * (int(bits[1]) - 0.5) * 2

def decode_2PAM():
        pass

def decode_QPSK():
    pass
pass