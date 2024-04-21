

import utils
import OFDM
import argparse

import numpy as np

def main(noise_variance, 
         N, 
         V, 
         H, 
         packets):
    transmitter = OFDM.OFDM_transmitter(V)
    receiver = OFDM.OFDM_receiver(V, H, N)

    # Create bits
    N_bits = packets * (2*N-2)
    bits = utils.generate_bits(N_bits)
    bits_received = np.array([])

    # For each block symbol
    for i in range(packets):
        # Pass through transmitter
        bits_sent = bits[(2*N-2)*i:(2*N-2)*i+(2*N-2)]
        signal = transmitter(bits_sent)
        
        # Pass through channel
        np.convolve(signal, H, 'same')

        # Add AWGN noise
        signal += np.sqrt(noise_variance)*np.random.randn()

        # Pass through receiver
        rec = receiver(signal)
        bits_received = np.concatenate((bits_received, rec))
    pe = utils.probability_of_bit_error(bits, bits_received)
    return pe


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='OFDM Simulation',
                                     description='Perform simulated OFDM transmission over an AWGN channel')
    parser.add_argument('-N', default=16)
    parser.add_argument('-V', default=2)
    parser.add_argument('-p', '--packets', default=1000)
    parser.add_argument('-s', '--noise_variance', default=0.2)
    parser.add_argument('-H', default=np.array([1, 1]), nargs='+', type=float)

    args = vars(parser.parse_args())
    args['H'] = np.array(args['H'], dtype=float)
    pe = main(**args)
    snr = utils.pe_to_snr(pe)
    snr = utils.lin_to_db(snr)
    print(f'Empirical Pe: {pe*100:.2f}% ({snr:.2f} dB)')