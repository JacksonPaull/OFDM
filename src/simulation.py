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
         v, 
         chanel_response, 
         packets):
    
    h = chanel_response # named to get around argparse help option

    receiver = OFDM.OFDM_receiver(N, v, h)
    transmitter = OFDM.OFDM_transmitter(N, v)

    # Create bits
    N_bits = packets * N
    bits = utils.generate_bits(N_bits)

    # For each block symbol
    bits_received = []
    for i in (window := trange(packets)):
        # Pass through transmitter
        bits_sent = bits[N*i:N*(i+1)]
        tx_signal = transmitter(bits_sent)
        
        # Pass through channel
        signal = np.convolve(tx_signal, h)[:len(tx_signal)]

        # Add AWGN noise
        noise = np.sqrt(noise_variance/N) * (np.random.randn(*signal.shape))
        noisy_signal = signal + noise

        # Pass through receiver
        b_hat = receiver(noisy_signal)
        bits_received += list(b_hat)

    pe_mean = utils.probability_of_bit_error(bits, bits_received)
    return pe_mean


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='OFDM Simulation',
                                     description='Perform simulated OFDM transmission over an AWGN channel')
    parser.add_argument('-N', default=16, type=int)
    parser.add_argument('-v', default=2, type=int)
    parser.add_argument('-p', '--packets', default=1000, type=int)
    parser.add_argument('-n', '--noise_variance', default=0.2, type=float)
    parser.add_argument('-c', '--chanel_response', default=np.array([1, 1]), nargs='+', type=float)

    args = vars(parser.parse_args())
    args['chanel_response'] = np.array(args['chanel_response'], dtype=float)
    pe = main(**args)
    snr = utils.pe_to_snr(pe)
    snr = utils.lin_to_db(snr)
    print(f'Empirical Pe: {pe*100:.2f}% ({snr:.2f} dB)')