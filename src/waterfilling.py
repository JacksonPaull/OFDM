
import numpy as np

def waterfill_alloc(gains: np.array, 
                 P_tot: float, 
                 gap:float=1):
    """
    Description: Determine an optimal power allocation across sub-channels with individual gains given in `gains`. 
    The output gains have array index matched to that of the supplied vector.

    Parameters:
        gains: np.array
            An array containing each individual channel gain, the order does not matter
    
        P_tot: The total energy to be divided among the channels
        gap: Shannon gap related to required probability of error
    """
    idx = np.arange(len(gains))
    a = np.row_stack((idx, gains))
    gains, idx = a[a[1,:].argsort()] # Sort the gains and the index

    n = 0
    g_inv = gap/gains
    a = (np.sum(g_inv) + P_tot) / len(g_inv)
    p_star = a - g_inv
    while p_star[0] < 0:
        n += 1
        g_inv = g_inv[1:]
        a = (np.sum(g_inv) + P_tot) / len(g_inv)
        p_star = a - g_inv

    p_star = np.concatenate([np.zeros(n), p_star])

    return p_star[idx.argsort()] # Re-sort back into original order