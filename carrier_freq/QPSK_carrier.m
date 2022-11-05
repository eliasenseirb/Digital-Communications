%% DELMASURE--DHAENE Elias
clear;
close all;
clc;

%% Parameters %%

fs = 10E3; %sampling frequency
Ds = 1E3; %symbol rate
Fse = fs/Ds; %sampling rate
Ts = 1e-3; %duration of a symbol

M = 4; %number of different symbol
n_b = log2(M); %number of bit per symbol

Ns = 5000; %number of symbol in the data
Nfft = 512; %number of point to compute the FFT for the PSD

%carrier frequency, make sure that fs > 2*f0 to have decent BER curves
f0 = 2.5E3;

%root-raised-cosine filter
alpha = 0.5;
G=rcosfir(alpha,4,Fse,1,'sqrt');

Eg = 0; %energy of the shape filter ->sum of the square of the modules
for i=1:length(G)
    Eg = Eg + G(i)^2;
end

sigA2 = 1; %theoretical variance of the symbols

eb_n0_dB = 0:0.5:10; %Eb/N0 list en dB
eb_n0 = 10.^( eb_n0_dB /10) ; %Eb/N0 list

sigma2 = sigA2 * Eg ./ ( n_b * eb_n0 ) ; %baseband noise's variance

BER = zeros ( size ( eb_n0 ) ); % BER list (results)
Pb = qfunc ( sqrt (2* eb_n0 ) ) ; % theoretical probabilities of error = 0.5*erfc(sqrt(eb_n0))

for j = 1: length(eb_n0)
    bit_error = 0;
    bit_count = 0;
    while bit_error < 100
        %% Transmitter %%

        Sb = randi([0,3],1,Ns); %generate Ns samples between 0 and 3 (00,01,10,11)
        %1 sample = 2 bits

        Ss = pskmod(Sb,M,pi/4,'gray'); %bit->symbol conversion

        Ss2 = upsample(Ss,Fse); %upsampling

        Sl = conv2(G,Ss2); %shaping the data signal

        %generating the carrier signal
        t_carr=(0:length(Sl)-1)/fs;
        carr_cplx = exp(1i*2*pi*f0*t_carr);

        S = real(Sl .*carr_cplx);
        %% Channel %%

        alpha0 = 1;
        nl = sqrt(sigma2(j)/4) * (randn(size(S))) ; %AWGN
        y = alpha0 * S + nl;

        %% RF->Baseband %%

        carr_I = 2*cos(2*pi*(f0)*t_carr);
        carr_Q = 2*cos(2*pi*(f0)*t_carr + pi/2);

        yI = y .*carr_I;
        yQ = y .*carr_Q;

        yl = yI + 1i*yQ;

        %% Receiver %%

        Ga = fliplr(G); %matched filter definition

        Rg = conv2(G, Ga); %Complete filter

        [~,delay] = max(Rg); %Delay created by the signal

        rl = conv2(Ga, yl); %convolving back the signal

        rln = rl(delay:Fse:length(rl)); %downsampling starting at the right spot

        %Decision process
        bn = pskdemod(rln,M,pi/4,'gray'); %Symbol -> bit process

        Sb2 = zeros(1, n_b*length(Sb)); %0,1,2,3 -> 00,01,10,11 to evaluate bit per bit for the input signal
        for i=1:1:length(Sb)
            if (Sb(i) == 0)
                Sb2(2*i-1) = 0; Sb2(2*i) = 0;
            elseif (Sb(i) == 1)
                Sb2(2*i-1) = 0; Sb2(2*i) = 1;
            elseif (Sb(i) == 2)
                Sb2(2*i-1) = 1; Sb2(2*i) = 0;
            else
                Sb2(2*i-1) = 1; Sb2(2*i) = 1;
            end
        end

        bn2 = zeros(1, n_b*length(bn)); %0,1,2,3 -> 00,01,10,11 to evaluate bit per bit for the output signal
        for i=1:1:length(bn)
            if (bn(i) == 0)
                bn2(2*i-1) = 0; bn2(2*i) = 0;
            elseif (bn(i) == 1)
                bn2(2*i-1) = 0; bn2(2*i) = 1;
            elseif (bn(i) == 2)
                bn2(2*i-1) = 1; bn2(2*i) = 0;
            else
                bn2(2*i-1) = 1; bn2(2*i) = 1;
            end
        end

        %BER calculus
        for i =1:Ns*n_b
            if(Sb2(i) ~= bn2(i))
                bit_error = bit_error + 1;
            end
            bit_count = bit_count + 1;
        end
    end
    BER(j) = bit_error/bit_count;
end

%% Plots
Te = 1/fs;

scatterplot(rln);

figure;
semilogy(eb_n0_dB,BER,'b-*');
hold on
semilogy(eb_n0_dB,Pb,'r--+');
xlabel("E_b/N_0 in dB");
ylabel("log(BER)");
title("BER evolution in function of the SNR");
grid on
legend('Simulation','Theory','Location','southwest');



freq = linspace(-fs/2,fs/2, Nfft);

[DSP_exp,f]=pwelch(S,ones(1,Nfft),0,Nfft,fs,'centered');
Tf_S = fft(S,Nfft);
DSP_th = abs((Tf_S)).^2;

figure;
semilogy(freq,fftshift(DSP_exp),'b'); %5000 times lower
hold on;
semilogy(freq,fftshift(real(DSP_th)),'r');
xlabel("Frequency (Hz)");
ylabel("PSD(s(t))");
title("Power Spectral Density of the signal");
grid on
legend('Simulation','Theory','Location','southwest');