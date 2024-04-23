

import utils
import OFDM
import argparse
from tqdm import trange

import numpy as np

def main(snr, 
         N, 
         V, 
         H, 
         packets):
    transmitter = OFDM.OFDM_transmitter(V)
    receiver = OFDM.OFDM_receiver(V, H, N)

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
        signal = np.convolve(tx_signal, H, 'full')[:len(tx_signal)]

        # Add AWGN noise
        # TODO
        # Should the noise variance be calculated here?
        # I would think that it should be 0.2 based on the MFB, 
        # but that feels too high given signal energy
        signal_power = np.mean(np.abs(signal) ** 2)
        noise_variance = signal_power * 10 ** (-snr/10)

        # TODO
        # Should this noise be complex or real?
        noise = np.sqrt(noise_variance/2)*(np.random.randn(*signal.shape)
                                           +1j*np.random.randn(*signal.shape))
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
    parser.add_argument('-s', '--snr', default=10.0, type=float)
    parser.add_argument('-H', default=np.array([1, 1]), nargs='+', type=float)

    args = vars(parser.parse_args())
    args['H'] = np.array(args['H'], dtype=float)
    pe = main(**args)
    snr = utils.pe_to_snr(pe)
    snr = utils.lin_to_db(snr)
    print(f'Empirical Pe: {pe*100:.2f}% ({snr:.2f} dB)')