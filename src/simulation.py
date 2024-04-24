"""
This problem contains an implementation of a simulation of an OFDM channel for homework 9 problem 4

"""

import utils
import OFDM
import argparse
from tqdm import trange

import numpy as np

def main(noise_variance, 
         N, 
         V, 
         H, 
         packets):
    transmitter = OFDM.OFDM_transmitter(V)
    receiver = OFDM.OFDM_receiver(V, H, N)
    Q = np.convolve(H, np.conj(H[::-1]))

    # Create bits
    N_bits = packets * N
    bits = utils.generate_bits(N_bits)
    pe_mean = 0

    # For each block symbol
    for i in (window := trange(packets)):
        # Pass through transmitter
        bits_sent = bits[N*i:N*i+N]
        tx_signal = transmitter(bits_sent)
        
        # Pass through channel
        signal = np.convolve(tx_signal, Q, 'full')[len(H)-1:1-len(H)]

        # Add AWGN noise
        noise = np.sqrt(noise_variance/2)*np.random.randn(*signal.shape)
        noisy_signal = signal + noise

        # Pass through receiver
        bits_received = receiver(noisy_signal)
        pe = utils.probability_of_bit_error(bits_sent, bits_received)
        pe_mean = pe_mean * (i/(i+1)) + pe/(i+1)
        window.set_description(f'Empirical Pe: {pe_mean*100:.2f}%')
    return pe_mean


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='OFDM Simulation',
                                     description='Perform simulated OFDM transmission over an AWGN channel')
    parser.add_argument('-N', default=16, type=int)
    parser.add_argument('-V', default=2, type=int)
    parser.add_argument('-p', '--packets', default=1000, type=int)
    parser.add_argument('-n', '--noise_variance', default=0.2, type=float)
    parser.add_argument('-H', default=np.array([1, 1]), nargs='+', type=float)

    args = vars(parser.parse_args())
    args['H'] = np.array(args['H'], dtype=float)
    pe = main(**args)
    snr = utils.pe_to_snr(pe)
    snr = utils.lin_to_db(snr)
    print(f'Empirical Pe: {pe*100:.2f}% ({snr:.2f} dB)')