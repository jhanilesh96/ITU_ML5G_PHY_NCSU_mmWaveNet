
clear
Dataset_pilots = 40;
Dataset_snr = 1;

load('prec_comb_sig_'+string(Dataset_pilots)+'_pilots_'+string(Dataset_snr)+'_data_set.mat');


%% System parameters
Nt = 16; % Number of TX antennas
Nr = 64; % Number of RX antennas
Nbits= 4; % Number of bits available to represent a phase shift in the analog precoder/combiner.
Lt = 2;  % Number of TX RF chains 
Lr = 4;  % Number of RX RF chains 
Ns = 2;  % Number of data streams to be transmitted
Nfft=256; % Number of subcarriers in the MIMO-OFDM system
Pt=1; % Transmit power(mw)
Nfilter = 20;
Mfilter = 1; %no oversampling
rolloff = 0.8;
MHz = 1e6; 
fs = 1760*MHz; %Sampling frequency
Ts = 1/fs;

% Training parameters
Nc = 100; % Number of channels available in the channel data set up to 10000
Ntrain=Dataset_pilots; % Number of training symbols to be received for each one of the available channels

Nres = 2^Nbits; %resolution ofr the phase shifters
Phi=zeros(Ntrain*Lr,Nt*Nr);%Initialize measurement matrix Phi in [1,(10)] of size LrxNtNr.
                           % We have a different measurement matrix for every training symbol.
                           % Here we are initilizaing the measurement
                           % matrices for all the training symbols, which
                           % correspond to matrix Phi in (13)
rng(1);
Ftr = Ftr_save;
Wtr = Wtr_save;
signal = signal_save;

Cw=zeros(Ntrain*Lr,Ntrain*Lr);
for i=1:Ntrain
    Cw((i-1)*Lr+1:i*Lr,(i-1)*Lr+1:i*Lr)= Wtr(:,(i-1)*Lr+1:i*Lr)'*Wtr(:,(i-1)*Lr+1:i*Lr);
end
for i=1:length(Cw)
    Cw(i,i)=real(Cw(i,i));
end
Dw=chol(Cw);

for i=1:Ntrain
   Phi((i-1)*Lr+(1:Lr),:)=kron(signal(:,i).'*Ftr(:,(i-1)*Lt+(1:Lt)).',Wtr(:,(i-1)*Lr+(1:Lr))');% Generate Phi in (13)
end

Gr=2*Nr*2;
Gt=2*Nt*2;
zt = (0:Nt-1)';
zr = (0:Nr-1)';   
At=zeros(Nt,Gt);%Nt-by-Gt
Ar=zeros(Nr,Gr);%Nr-by-Gr
for ite=1:1:Gt
    At(:,ite)=(sqrt(1/Nt)*exp(-1j*2*pi*(-0.5+(ite-1)/Gt)*[0:Nt-1])).';
end
for ite=1:1:Gr
    Ar(:,ite)=(sqrt(1/Nr)*exp(-1j*2*pi*(-0.5+(ite-1)/Gr)*[0:Nr-1])).';
end

Psi=kron(conj(At),Ar);

Upsilon_w=Dw'\(Phi*Psi);

y_real = h5read('test_dataset_v3_'+string(Dataset_pilots)+'_pilots_'+string(Dataset_snr)+'_data_set.hdf5', '/training_data_real');
y_imag = h5read('test_dataset_v3_'+string(Dataset_pilots)+'_pilots_'+string(Dataset_snr)+'_data_set.hdf5', '/training_data_imag');
y = y_real+ 1i*y_imag;

clear y_real y_imag



H = zeros(length(y(:,1,1)), Nr, Nt, Nfft);

tic
for i = 1:length(y(:,1,1))
    y_i = squeeze(y(i, :, :));
    y_i=Dw'\(y_i);

    if(Dataset_snr==1)
        spar=0;
    elseif (Dataset_snr==2)
        spar=1;
    else
        spar=3;
    end

    [H_est_spar,index_set2]= SOMP_sparse(y_i,Upsilon_w,Psi,Nfft,Ntrain,Lr,Nr,Nt,spar);
    H(i,:,:,:) = H_est_spar;
    if mod(i,50) == 0
        [i,length(y(:,1,1))]
    end
end
toc

clearvars -except H Dataset_pilots Dataset_snr
tic
save('test_result_v3_'+string(Dataset_pilots)+'_pilots_'+string(Dataset_snr)+'_data_set.mat', 'H', '-v7.3')
toc



