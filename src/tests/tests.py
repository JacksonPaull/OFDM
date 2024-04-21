# Folder for a bunch of unit tests for utility functions

# Test length of encodings matches up
# len(encode_pam) == len(bits)
# len(encode_qpsk) = len(bits)/2 

# Test that noiseless encoding / decoding results in no change

import unittest
from .. import utils

class TestEncodings(unittest.TestCase):
    def test_pam(self):
        bits = utils.generate_bits(10000)
        pam = utils.encode_2PAM(bits)

        # Length must be equal
        self.assertEqual(len(bits), len(pam))

        self.assertEqual(utils.probability_of_bit_error(bits, utils.decode_2PAM(pam)), 0)

    def test_qpsk(self):
        bits = utils.generate_bits(10000)
        qpsk = utils.encode_QPSK(bits)

        # Length must be equal
        self.assertEqual(len(bits), len(qpsk) * 2)

        self.assertEqual(utils.probability_of_bit_error(bits, utils.decode_QPSK(qpsk)), 0)


if __name__ == '__main__':
    unittest.main()