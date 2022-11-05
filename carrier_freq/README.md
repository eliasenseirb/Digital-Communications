This chain is very basic, it is really intersting to implement in order to understand how digital communications works and to know what is different with a **carrier frequency** in the signal. 

In this chain there are four blocks : 
- Parameters
- Transmitter
- Channel
- Receiver

There are several parameters that defines the chain. To quote a few : the numbers of symbols, the **Sound to Noise Ratio** (SNR), the length of the bit signal, etc.

The transmitter takes a random bit signal. This signal is then modulated with QSPK here before being upsampled and filtered by a shape filter. The filtered signal is then modulated by a cosine.

The channel is here to add noise that disturbs the signal. Here there isn't any time nor frequency desynchronization.

The receiver splits the complex signal and filters back the signal before downsampling it and demodulating it. After that the BER is computed and plot.
