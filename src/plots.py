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