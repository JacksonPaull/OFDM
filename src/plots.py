import matplotlib.pyplot as plt

def plot_tx_vs_rx(tx_signal, rx_signal, show = True, number_of_samples=100):
    fig = plt.figure(figsize=(8,2))
    plt.title('Transmitted Signal vs Received Signal')
    plt.plot(tx_signal[:number_of_samples], label='TX signal')
    plt.plot(rx_signal[:number_of_samples], label='RX signal')
    plt.legend(fontsize=10)
    plt.xlabel('Time'); plt.ylabel('$|x(t)|$')
    plt.grid(True)
    if show:
        plt.show()
        return

    return fig

def plot_waterfilling_info(gains, energy, rates, show=True, fsize=(20,5)):
    fig, axs = plt.subplots(1, 3)
    fw, fh = fsize
    fig.set_figwidth(fw), fig.set_figheight(fh)
    fig.suptitle('Waterfilling Gains, Energy, and Rates')

    # Gains
    ax = axs[0]
    ax.set_title('Gain ($\gamma_n$)')
    ax.set_xlabel('Subchannel Index n')
    ax.set_ylabel('Channel Gain')
    ax.stem(gains)
    ax.grid(True, ls='--', lw=0.7)

    # Energy
    ax = axs[1]
    ax.set_title('Allocated Energy ($\overline{e}_n$)')
    ax.set_xlabel('Subchannel Index n')
    ax.set_ylabel('Energy per dimension')
    ax.stem(energy)
    ax.grid(True, ls='--', lw=0.7)


    # Rates
    ax = axs[2]
    ax.set_title('Effective Rate ($\overline{r}_n$)')
    ax.set_xlabel('Subchannel Index n')
    ax.set_ylabel('Spectral Efficiency')
    ax.stem(rates)
    ax.grid(True, ls='--', lw=0.7)

    if show:
        plt.show()
        return
    
    return fig, axs