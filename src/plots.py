import matplotlib.pyplot as plt

def plot_tx_vs_rx(tx_signal, rx_signal, show = True):
    fig = plt.figure(figsize=(8,2))
    plt.plot(tx_signal, label='TX signal')
    plt.plot(rx_signal, label='RX signal')
    plt.legend(fontsize=10)
    plt.xlabel('Time'); plt.ylabel('$|x(t)|$')
    plt.grid(True)
    if show:
        plt.show()

    return fig