# Folder for a bunch of unit tests for utility functions

# Test length of encodings matches up
# len(encode_pam) == len(bits)
# len(encode_qpsk) = len(bits)/2 

# Test that noiseless encoding / decoding results in no change

import numpy as np

import unittest
from .. import utils, OFDM, waterfilling, encoders

# class TestEncodings(unittest.TestCase):
#     def test_pam(self):
#         bits = utils.generate_bits(10000)
#         pam = utils.encode_2PAM(bits)

#         # Length must be equal
#         self.assertEqual(len(bits), len(pam))

#         self.assertEqual(utils.probability_of_bit_error(bits, 
#                                 utils.decode_2PAM(pam)), 0)

#     def test_qpsk(self):
#         bits = utils.generate_bits(10000)
#         qpsk = utils.encode_QPSK(bits)

#         # Length must be equal
#         self.assertEqual(len(bits), len(qpsk) * 2)

#         self.assertEqual(utils.probability_of_bit_error(bits, 
#                                 utils.decode_QPSK(qpsk)), 0)


class TestOFDM(unittest.TestCase):
    rx = OFDM.OFDM_receiver(64, 8, 'equal', np.array([1, 1]))
    tx = OFDM.OFDM_transmitter(64, 8, 'equal')
    bits = utils.generate_bits(10000)
    
    def test_symbol_encoding_decoding(self):
        # Test the fact that perfect symbols are decoded correctly
        _, tx_cache = self.tx(self.bits)
        b_hat = self.rx.decode_bits(tx_cache['symbols'])
        self.assertEqual(utils.probability_of_bit_error(self.bits, b_hat), 0)

    

if __name__ == '__main__':
    unittest.main()