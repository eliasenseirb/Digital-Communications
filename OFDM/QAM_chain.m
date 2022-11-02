clear
close all;
clc;

%% Parameters

f0 = 150E9;   %Carrier frequency
B10 = 10E6;   %10MHz band
B4 = 4E9;     %4GHz band

D10 = 100E6;  %Bitrate for the 10MHz band
D4 = 4E9;     %Bitrate for the 4GHz band

SNR_th = 25;  %Theoretical SNR

M=1024;       %Number of symboles
n_b = log2(M);%Number of bits per symbole

%% Q1

%Parameters
NF = 3;
T = 293;
kB = 1.380649E-23;

%Calculus
S10 = 10*log10(kB*T)+10*log10(B10)+NF+SNR_th+30;
S4 = 10*log10(kB*T)+10*log10(B4)+NF+SNR_th+30;

fprintf("\n");
fprintf("----------------Q1----------------\n");
fprintf("The sensativity for 4GHz approximately is %d dBm\n", round(S4));
fprintf("The sensitivity for 10MHz approximately is %d dBm\n", round(S10));

%% Q2

%Parameters
Ptx = 30;
Gtx = 2;
Grx = 30;
Prx4 = S4;
Prx10 = S10;
c = 3E8;
lambda0 = c/f0;

%Calculus
R4 = 1/(4*pi*10^((Prx4 - Ptx - Gtx - Grx)/20)/lambda0);
R10 = 1/(4*pi*10^((Prx10 - Ptx - Gtx - Grx)/20)/lambda0);

diff = abs(R4-R10);

fprintf("\n");
fprintf("----------------Q2----------------\n");
fprintf("The distance for 4GHz is %d m\n", R4);
fprintf("The distance for 10MHz is %d m\n", R10);
fprintf("The difference of distance between the two bandwidth is %d m\n", diff);

%% Q3

%Calculus
Tdmax4 = R4/c;
Tdmax10 = R10/c;

fprintf("\n");
fprintf("----------------Q3----------------\n");
fprintf("The timeslot of the channel for 4GHz is %d s\n", Tdmax4);
fprintf("The timeslot of the channel for 10MHz is %d s\n", Tdmax10);
%% Q4

%Calculus
Bc4 = 1/(10*Tdmax4);
Bc10 = 1/(10*Tdmax10);

fprintf("\n");
fprintf("----------------Q4----------------\n");
fprintf("The coherence bandwidth for 4GHz is %d Hz\n", Bc4);
fprintf("The coherence bandwidth for 10MHz is %d Hz\n", Bc10);
%% Q5

%Calculus
Nb4 = floor(B4/Bc4);
Nb_subcarrier4 = 2^nextpow2(Nb4);
Nb10 = floor(B10/Bc10);
Nb_subcarrier10 = 2^nextpow2(Nb10);

fprintf("\n");
fprintf("----------------Q5----------------\n");
fprintf("The number of sub-carriers for 4GHz is %d\n", Nb_subcarrier4);
fprintf("The number of sub-carriers for 10MHz is %d\n", Nb_subcarrier10);
%% Q6

%Calculus
Ts4 = 1/B4;
Ts10 = 1/B10;

T4 = Nb_subcarrier4*Ts4;
T10 = Nb_subcarrier10*Ts10;

fprintf("\n");
fprintf("----------------Q6----------------\n");
fprintf("The duration for an OFDM symbol for 4GHz is %d s\n", T4);
fprintf("The duration for an OFDM symbol for 10MHz is %d s\n", T10);
%% Q7

%Calculus
T_cp4 = Tdmax4;
T_cp10 = Tdmax10;


fprintf("\n");
fprintf("----------------Q7----------------\n");
fprintf("The duration of the cyclic prefix for 4GHz is %d s\n", T_cp4);
fprintf("The duration of the cyclic prefix for 10MHz is %d s\n", T_cp10);
%% Q8

%Calculus
P4 = Tdmax4/(Tdmax4+T4);
P10 = Tdmax10/(Tdmax10+T10);

Pb4 = P4*D4;
Pb10 = P10*D10;

r4 = Pb4/D4;
r10 = Pb10/D10;

NvD4 = D4-Pb4;
NvD10 = D10 - Pb10;

fprintf("\n");
fprintf("----------------Q8----------------\n");
fprintf("The bit rate loss for 4GHz is %d bits/s which represents %f%%, The new bit rate is %d bits/s\n", Pb4, r4, NvD4);
fprintf("The bit rate loss for 10MHz is %d bits/s which represents %f%%, The new bit rate is %d bits/s\n", Pb10, r10, NvD10);
%% Q9

%Calculus
Peff4 = 100/Nb_subcarrier4;
Peff10 = 10/Nb_subcarrier10;

PDb4 = Peff4*D4;
PDb10 = Peff10*D10;

fprintf("\n");
fprintf("----------------Q9----------------\n");
fprintf("The rate loss for the guard band at 4GHz is %f%%, The bit rate loss is %d bits/s\n", Peff4, PDb4);
fprintf("The rate loss for the guard band at 10MHz is %f%%, The bit rate loss is %d bits/s\n", Peff10, PDb10);
%% Q10

%Calculus
D4f = NvD4 - PDb4;
D10f = NvD10 - PDb10;


fprintf("\n");
fprintf("----------------Q10----------------\n");
fprintf("The new bit rate for 4GHz is %d bits/s\n", D4f);
fprintf("The new bit rate for 10MHz is %d bits/s\n", D10f);
%% Q11

%Parameters
Tc = 1E-3;

%Calculus
Tx4 = floor(Tc/(T4+T_cp4));
Tx10 = floor(Tc/(T10+T_cp10));

Px4 = 1/Tx4*D4f;
Px10 = 1/Tx10*D10f;

fprintf("\n");
fprintf("----------------Q11----------------\n");
fprintf("The number of transmittable OFDM symbols for 4GHz is %d symboles, The loss due to the learning sequence is %d bits/s\n", Tx4, Px4);
fprintf("The number of transmittable OFDM symbols for 10MHz is %d symboles, The loss due to the learning sequence is %d bits/s\n", Tx10, Px10);
%% Q12

%Calculus
Debit4 = D4f - Px4;
Debit10 = D10f - Px10;


fprintf("\n");
fprintf("----------------Q12----------------\n");
fprintf("The real bit rate for 4GHz is %d bits/s\n", Debit4);
fprintf("The real bit rate for 10MHz is %d bits/s\n", Debit10);
%% Q13

%Parameters
r=1/2;

%Calculus
DD4 = r*Debit4;
DD10 = r*Debit10;

fprintf("\n");
fprintf("----------------Q13----------------\n");
fprintf("The real bit rate for a channel coding (r=0.5) for 4GHz is  %d bits/s\n", DD4);
fprintf("The real bit rate for a channel coding (r=0.5) for 10MHz is %d bits/s\n", DD10);

%% Q14

%Parameters
NbOFDM = Tx4;                   %Number of OFDM symbols
NbBits = n_b*NbOFDM*Nb_subcarrier4; %Total number of bits for the sequency
D = floor(Tdmax4/Ts4);          %Size of the prefix
L=floor(D*rand);                %Size of the noise

guard = 50;                     %Guard band's width
SNR_exp = 40;                   %SNR used
%% Transmitter %%

signal = randi([0 M-1],1,NbBits/n_b); %Generating samples between 0 and 1023
ss = qammod(signal,M,'gray'); %QAM Modulation

ssMAT = reshape(ss,Nb_subcarrier4,NbOFDM); %Shaping the symbols according to the IFFT of Matlab
ssMAT(Nb_subcarrier4/2-guard:Nb_subcarrier4/2+guard,:) = 0; %Guard band of the signal

sMAT = ifft(ssMAT, Nb_subcarrier4);  %Matrix that contains the symbols sorted in columns

sMAT2 = cat(1,sMAT(end-D+1:end,:),sMAT); %Adding the cyclic prefix (CP)

s = reshape(sMAT2,1,(Nb_subcarrier4+D)*NbOFDM); %Complex envelope of the signal
%% Channel %%

noise = randn(1,L)+1i*randn(1,L);
h = sqrt(1/(2*L))*noise; %Channel
H = fft(h,Nb_subcarrier4);   %Frequency response on the sub-carriers

y = filter(h,1,s);  %Received signal
%% Receiver %%

Py = mean(abs(y).^2);   %Power of y
Pb = Py/(10^(SNR_exp/10));
b = sqrt(Pb/2)*(randn(size(y))+1i*randn(size(y))); %Noise

x = y + b;  %Noised signal

xMat = reshape(x,Nb_subcarrier4+D,NbOFDM);
ss_detect_MAT = fft(xMat(D+1:end,:),Nb_subcarrier4); %Not taking the prefix


H_zf = ss_detect_MAT(:,1)./ssMAT(:,1); %Channel estimation with the learning samples

ss_ZF = ss_detect_MAT./repmat(H_zf,1,NbOFDM); %Zero forcing (ZF) equalizer

ss_detect = reshape(ss_ZF,1,Nb_subcarrier4*NbOFDM);

s_demod = qamdemod(ss_detect,  M, 'gray'); %Demodulation


%BER Calculus
sigbit = zeros(1,NbBits);
demodbit = sigbit;
k=1;
for i=1:n_b:length(signal) %Bit conversion of the signals
    sigbit(i:i+n_b-1) = de2bi(signal(k),n_b);
    demodbit(i:i+n_b-1) = de2bi(s_demod(k),n_b);
    k=k+1;
end

BER = sum(sigbit ~= demodbit)/NbBits;


fprintf("\n");
fprintf("----------------Code----------------\n");
fprintf("BER value is: %f\n", BER);
fprintf("Channel coding helps reducing the BER. The reduction depends on the complexity of the code.\n");
fprintf("The code (133,171) will be more efficient than the code (5,7) but it will take more time to compute.\n");

%% Figures

%Plotting only a small portion of the signal, it gains time and it's enough
%to draw conclusions on whether the chains works or not
scatterplot(ss(1:50000));
title("Scatterplot of the transmitted signal");

scatterplot(ss_detect(1:50000));
title("Scatterplot of the received signal");