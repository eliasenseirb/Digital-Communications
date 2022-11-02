This chain is an elaborated one, in Orthognonal Frequency Multiplexing Modulation (OFDM), we use **Multiplexing** (MUX) and **Demultiplexing** (DEMUX). Moreover a **Cyclic Prefix** (CP) is added in front of every OFDM symbol to help out with the time synchronization issue. If you want a OFDM chain without the prefix it's also possible, you have to define the size of the prefix to 0. However, it will be precised in the parameters' section.

In this chain there are four blocks : 
- Parameters
- Transmitter
- Channel
- Receiver

The parameters are the same as the basic ones. Plus everything linked to OFDM: number of OFDM symbols, number of sub carriers, length of the prefix, etc.

The transmitter takes a random bit signal. This signal is then modulated with *QSPK* or *1024 QAM* here. The signal is then demultiplexed, to do this we reshape the signal. After that a **Fast Fourier Transform** (FFT) is applied to each sub carrier. Before entering the channel the signal is multiplexed, reshaped as a long signal.

Since every sub carrier is sent on a different timeline and in order not to lose information on the data, a CP is added. This prefix is made of a portion of the end part of the signal. This prefix is then added in front of each sub carrier.

The channel is here to add noise to disturb the signal. Each sub carrier has different noise values. 

The receiver is doing the opposite transformation of the transmitter, the signal is demodulated and a **Inverse FFT** (IFFT) is applied, the signal is then reshaped. The next step is to estimate the channel, to do this we use the Zero Forcing (ZF) method. This step is here to make a decision before demodulating each symbol and computing the BER.
