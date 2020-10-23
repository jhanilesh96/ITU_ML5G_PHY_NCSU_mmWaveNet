function [H_freq, H_time] = gen_channel_ray_tracing_rev(nc,Nr,Nt,Nfft,Ts,rolloff,Mfilter,chan_save_file)
%function [H_freq, H_time] = gen_channel_ray_tracing(nc,Nr,Nt,Nfft,Ts,rolloff,Mfilter,chan_save_file)
%
% INPUTS
% nc - index of channel from 1 to 10,000
% Nr - number of receive antennas
% Nt - number of transmit antennas
% Nfft - size of the FFT
% BW - bandwidth
% chan_save_file (optional file name where channels are stored)
%
% OUTPUTS
% H_freq is Nr x Nt x Nfft
% H_time is Nr x Nt x L where L varies based on the delay spread
%
% Created June 9, 2020

if nargin<8
    chan_save_file = 'target_channels.hdf5';
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% load the file
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
channel_data = h5read(chan_save_file,'/channel_challenge');
% channel_data has dimensions 8 x 100 x 10000 

% Initialize parameters
[N_ray_param, N_ray_max,N_chan_target] = size(channel_data);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Generate the channel based on the given parameters
%% Filtering effects are considered
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


gain_db         = channel_data(1,:,nc);
gain_lin        = 10.^(gain_db/10);
gain_phase      = channel_data(8,:,nc)*pi/180;
gain_comp       = gain_lin.*exp(1i*gain_phase);
toa_raw         = channel_data(2,:,nc);
toa             = toa_raw - min(toa_raw);
%AoD_el          = channel_data(3,:,nc); % IGNORED
AoD_az          = channel_data(4,:,nc);
%AoA_el          = channel_data(5,:,nc); % IGNORED
AoA_az          = channel_data(6,:,nc);
is_LOS          = channel_data(7,:,nc);
is_channel_LOS  = sum(is_LOS);

% establish channel length, supposing we start a few samples before the
% first ray
early_samples = 3; % just to capture the beginning part of the sinc

% compute rms delay spread
mean_delay = toa*gain_lin' / norm(gain_lin);
rms_delay_spread = sqrt( sum( (toa-mean_delay).^2 .* gain_lin) / norm(gain_lin));

L = min(ceil(rms_delay_spread/Ts),ceil(Nfft/3)) + early_samples;

% Generate At and Ar
zt = (0:Nt-1)';
zr = (0:Nr-1)';        
At = zeros(Nt,N_ray_max);
Ar = zeros(Nr,N_ray_max);
for i=1:N_ray_max
    At(:,i) = exp(1i*pi*cos(AoD_az(i))*zt)/sqrt(Nt);
    Ar(:,i) = exp(1i*pi*cos(AoA_az(i))*zr)/sqrt(Nr);
end

% Generate time domain channel
H_time = zeros(Nr,Nt,L);
for ell=1:L        
    for paths = 1:N_ray_max
        H_time(:,:,ell)= H_time(:,:,ell) + gain_comp(paths) ...
        * sinc((ell*Ts-toa(paths) + early_samples*Ts)/Mfilter/Ts)*cos(pi*rolloff*(ell*Ts-toa(paths) + early_samples*Ts)/Mfilter/Ts)/(1-(2*rolloff*(ell*Ts-toa(paths) + early_samples*Ts)/Mfilter/Ts)^2) ...
        * Ar(:,paths) * At(:,paths)';
    end %paths
end %ell

H_freq = zeros(Nr,Nt,Nfft);

% Generate frequency domain channel
for nt = 1:Nt
    for nr = 1:Nr
        H_freq(nr,nt,:) = fft(H_time(nr,nt,:),Nfft);
    end %nr
end %nt    

% Normalize the channel
rho = Nt*Nr*Nfft/norm(H_freq(:),'fro')^2;
H_freq = sqrt(rho)*H_freq;
H_time = sqrt(rho)*H_time;

end
