

import utils
import OFDM
import argparse

import numpy as np

# TODO: Decouple bit generation -> symbol encoding -> transmitter
# works out nicely with N = bits / block in this example, but won't work generally
# with encoding schemes other than 4QAM

def main(noise_variance, 
         N, 
         V, 
         H, 
         packets):
    transmitter = OFDM.OFDM_transmitter(V)
    receiver = OFDM.OFDM_receiver(V, H, N)

    # Create bits
    N_bits = packets * N
    bits = utils.generate_bits(N_bits)
    pe = []

    # For each block symbol
    for i in range(packets):
        # Pass through transmitter
        bits_sent = bits[N*i:N*i+N]
        signal = transmitter(bits_sent)
        assert(np.sum(np.abs(np.imag(signal))) == 0)
        
        # Pass through channel
        #signal = np.convolve(signal, H, 'same')

        # Add AWGN noise
        noise = np.sqrt(noise_variance/2)*(np.random.randn(*signal.shape)
                                           +1j*np.random.randn(*signal.shape))
        signal += noise

        # Pass through receiver
        bits_received = receiver(signal)
        pe.append(utils.probability_of_bit_error(bits_sent, bits_received))
    return np.mean(pe)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='OFDM Simulation',
                                     description='Perform simulated OFDM transmission over an AWGN channel')
    parser.add_argument('-N', default=16, type=int)
    parser.add_argument('-V', default=2, type=int)
    parser.add_argument('-p', '--packets', default=1000, type=int)
    parser.add_argument('-s', '--noise_variance', default=0.2, type=float)
    parser.add_argument('-H', default=np.array([1, 1]), nargs='+', type=float)

    args = vars(parser.parse_args())
    args['H'] = np.array(args['H'], dtype=float)
    pe = main(**args)
    snr = utils.pe_to_snr(pe)
    snr = utils.lin_to_db(snr)
    print(f'Empirical Pe: {pe*100:.2f}% ({snr:.2f} dB)')