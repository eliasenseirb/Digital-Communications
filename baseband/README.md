
This chain is the simplest one, this code will introduce you digital communications chains.

In this chain there are four blocks : 
- **Parameters**
- **Transmitter**
- **Channel**
- **Receiver**

There are several parameters that defines the chain. To quote a few : the numbers of symbols, the SNR, the length of the bit signal, etc.

The transmitter takes a random bit signal. This signal is then modulated with QSPK here before being upsampled and filtered by a shape filter.

The channel is here to add noise that disturbs the signal. Here there isn't any time nor frequency desynchronization.

The receiver filters back the signal before downsampling it and demodulating it. After that the BER is computed and plot.
