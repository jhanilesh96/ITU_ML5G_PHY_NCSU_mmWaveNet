%% Script to generate M received training symbols for every MIMO channel in the matrix obtained from NYUSIM channel simulator
% - A fully connected hybrid MIMO architecture is assumed at the TX and RX. 
% - The received training pilot is obtained at the output of the ADCs for the different RF chains. 
% - Only analog precoding/combining during training is used. 
% - ULAs are assumed
% References:
% [1] J. Rodríguez-Fernández, N. González-Prelcic, K. Venugopal and R. W. Heath, "Frequency-Domain Compressive Channel Estimation for Frequency-Selective Hybrid Millimeter Wave MIMO Systems," IEEE Transactions on Wireless Communications, vol. 17, no. 5, pp. 2946-2960, May 2018.

clc, clear all;
%delete(gcp('nocreate')); %Delete previously parallel pools
% parpool(4); %declare number of workers (max=cores) 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Initilization
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

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
Nc = 50; % Number of channels available in the channel data set up to 10000
Ntrain=80; % Number of training symbols to be received for each one of the available channels

% SNR=-20:5:0;
SNR=[-15 -10];
snr = 10.^(SNR/10);

Nres = 2^Nbits; %resolution ofr the phase shifters
Phi=zeros(Ntrain*Lr,Nt*Nr);%Initialize measurement matrix Phi in [1,(10)] of size LrxNtNr.
                           % We have a different measurement matrix for every training symbol.
                           % Here we are initilizaing the measurement
                           % matrices for all the training symbols, which
                           % correspond to matrix Phi in (13)
rng(1);
tt=randi(Nres,[Nt Ntrain*Lt]);
for i = 1:Nres
   tt(tt==i) = exp(1i*2*pi*(i-1)/Nres);
end
Ftr=tt;  
        
tt=randi(Nres,[Nr Ntrain*Lr]);
for i = 1:Nres
   tt(tt==i) = exp(1i*2*pi*(i-1)/Nres);
end
Wtr=tt;    

Ftr = Ftr/sqrt(Nt);% pseudorandom training precoders (generating Ntrain, precoders for all the training symbols)
Wtr = Wtr/sqrt(Nr);% pseudoranmdom training combiners

Cw=zeros(Ntrain*Lr,Ntrain*Lr);
for i=1:Ntrain
    Cw((i-1)*Lr+1:i*Lr,(i-1)*Lr+1:i*Lr)= Wtr(:,(i-1)*Lr+1:i*Lr)'*Wtr(:,(i-1)*Lr+1:i*Lr);
end
for i=1:length(Cw)
    Cw(i,i)=real(Cw(i,i));
end
Dw=chol(Cw);

for i=1:Ntrain,
   signal = sqrt(1/2/Lt)*(sign(randn(Lt,1))+1i*sign(randn(Lt,1))); %training signal q (frequency flat)
   Phi((i-1)*Lr+(1:Lr),:)=kron(signal.'*Ftr(:,(i-1)*Lt+(1:Lt)).',Wtr(:,(i-1)*Lr+(1:Lr))');% Generate Phi in (13)
end

Gr=2*Nr*2;
Gt=2*Nt*2;
zt = (0:Nt-1)';
zr = (0:Nr-1)';        
% At = zeros(Nt,Gt);
% Ar = zeros(Nr,Gr);
% res_r=linspace(0,pi,Gr);
% res_t=linspace(0,pi,Gt);
% for i=1:Gt
%     At(:,i) = exp(1i*pi*cos(res_t(i))*zt)/sqrt(Nt);
% %     Ar(:,i) = exp(1i*pi*cos(res(i))*zr)/sqrt(Nr);
% end
% 
% for i=1:Gr
% %     At(:,i) = exp(1i*pi*cos(res(i))*zt)/sqrt(Nt);
%     Ar(:,i) = exp(1i*pi*cos(res_r(i))*zr)/sqrt(Nr);
% end

At=zeros(Nt,Gt);%Nt-by-Gt
Ar=zeros(Nr,Gr);%Nr-by-Gr
for ite=1:1:Gt
    At(:,ite)=(sqrt(1/Nt)*exp(-1j*2*pi*(-0.5+(ite-1)/Gt)*[0:Nt-1])).';
end
for ite=1:1:Gr
    Ar(:,ite)=(sqrt(1/Nr)*exp(-1j*2*pi*(-0.5+(ite-1)/Gr)*[0:Nr-1])).';
end

Psi=kron(conj(At),Ar);
% A=Phi*Psi;

NMSE_SOMP=ones(length(SNR),1);
NMSE_MSOMP=ones(length(SNR),1);
% NMSE_genie_LS2=ones(length(SNR),1);
% err1=zeros(Nc,1);
Upsilon_w=Dw'\(Phi*Psi);
% A = {};
% sw=size(Upsilon_w);
% for t=1:Nfft
%    A{t}= Upsilon_w;
% end
% spar=10;
tic

for s=1:length(SNR)
    err1=0;
    err2=0;
%     err3=0;
    
for j=1:Nc %Nc is number of channels 
%     Hk = gen_channel_ray_tracing(j,Nr,Nt,Nfft,Ts,rolloff,Mfilter); %NrxNt
    Hk = gen_channel_ray_tracing_rev(j+4687,Nr,Nt,Nfft,Ts,rolloff,Mfilter); %NrxNt

%     rank(Hk(:,:,1))
    % Load channel parameters for channel j and build channel matrix including filtering
    % effects
   
    var_n = Pt/snr(s);
    Noise = sqrt(var_n/2)*(randn(Nr,Ntrain,Nfft)+1i*randn(Nr,Ntrain,Nfft));
    SNRaux = zeros(Nfft,1);
    nn=zeros(Ntrain*Lr,Nfft);
    r=zeros(Ntrain*Lr,Nfft);
    for k = 1:Nfft % Generate RX pilots for every subcarrier
            for t=1:Ntrain % Ntrain is M
                Wrf_t = Wtr(:,(t-1)*Lr+(1:Lr));
                nn((1:Lr)+Lr*(t-1),k) = Wrf_t'*Noise(:,t,k);
            end
            signal_k = Phi*reshape(Hk(:,:,k),[],1);
            noise_k = nn(:,k);
            r(:,k) = Phi*reshape(Hk(:,:,k),[],1) + nn(:,k);
            
    end
   
    yw=Dw'\(r);
%     [SNR(s) var_n]
    if(SNR(s)<-11)
        spar=9;
        f_actor = 0.999;
    elseif (SNR(s)<-6)
        spar=9;
        f_actor = 0.999;
    else
        spar=3;
        f_actor = 0.995;
        end
    f_actor = 0.95;
    [H_est1, idx_set]=SOMP_v2(yw,Upsilon_w,Psi,Nfft,Ntrain,Lr,Nr,Nt,f_actor*var_n);
    err1= err1+sum(sum(sum(abs(Hk-H_est1).^2)))/ sum(sum(sum(abs(Hk).^2)));
    

    [H_est_spar,index_set2]= SOMP_sparse(yw,Upsilon_w,Psi,Nfft,Ntrain,Lr,Nr,Nt,spar);
    err2= err2+sum(sum(sum(abs(Hk-H_est_spar).^2)))/ sum(sum(sum(abs(Hk).^2)));
    
    fprintf('%dth Nc is done...................\n',j)
end

NMSE_SOMP(s)=err1/Nc;
NMSE_MSOMP(s)=err2/Nc;
% NMSE_genie_LS2(s)=err3/Nc;
fprintf('%dth SNR is done...................\n',s)

end
toc

% leg1 = legend('$\bar{x}$','$\tilde{x}$','$\hat{x}$');
% set(leg1,'Interpreter','latex');

figure(1)
plot(SNR,10*log10(NMSE_SOMP),'r-o')
hold on
plot(SNR,10*log10(NMSE_MSOMP),'k-*')
leg1 = legend('SWOMP [3]','M-SWOMP');
set(leg1,'Interpreter','latex');
% semilogy(SNR,NMSE,'r-o')
xlabel('SNR (dB)','Interpreter','latex')
ylabel('NMSE (dB)','Interpreter','latex')
title("$ M=\, $"+Ntrain+" Pilots ",'Interpreter','latex');

% title("Nr= "+Nr+" Nt="+Nt+" N_c= "+Nfft+" M= "+Ntrain+" Gr,Gt= "+Gr+", "+Gt);



% kk=3;
% H_sparse1=Ar'*Hk(:,:,kk)*At;
% figure(2)
% imagesc(abs(H_sparse1));
% 
% Hs2=reshape(H_sparse,[Gr,Gt,Nfft]);
% figure(3)
% imagesc(abs(Hs2(:,:,kk)));
% 
% Hes1=Ar*H_sparse1*At';
% Hes2=Ar*Hs2(:,:,kk)*At';
% 
% mse1=norm(Hk(:,:,kk)-Hes1,'fro')^2
% mse2=norm(Hk(:,:,kk)-Hes2,'fro')^2
% mse3=norm(Hk(:,:,kk)-H_est_gen_LS(:,:,kk),'fro')^2
% mse4=norm(Hk(:,:,kk)-H_est1(:,:,kk),'fro')^2

% h_s=Psi'*reshape(Hk(:,:,1),[],1);
% Hs1=reshape(h_s,[Gr,Gt]);
% 
% figure(3)
% imagesc(abs(Hs1));
% ik=1;
% [H_est_gen_LS1,H_sparse12,index_set3]= Genie_LS_SMV(reshape(Hk(:,:,ik),[],1),yw,Upsilon_w,Psi,Nfft,Ntrain,Lr,Nr,Nt,epsilon);


