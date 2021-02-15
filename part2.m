%% Code for part 2
clear all; close all; clc

%%
[y, Fs] = audioread('Floyd.m4a');
% p8 = audioplayer(y,Fs); playblocking(p8); 

y = y(1:end-1);
yt = fft(y);
n = length(y);
L = n/Fs; % record time in seconds

k=(1/(L))*[0:n/2-1 -n/2:-1]; 
ks=fftshift(k);


%%

filter=zeros(size(yt));

for i = 1:length(ks)
     if(abs(ks(i))> 1)
         filter(i) =0;
     else
         filter(i) = 1;
     end
end

fftg = fftshift(filter);

ys = yt.*fftg;
s = (ifft(ys));

%% Plot

figure(1)
subplot(2,1,1)
plot((1:n)/Fs, y);
xlabel('Time [sec]'); 
ylabel('Amplitude');
title('Comfortably Numb')

subplot(2,1,2)
plot(ks, abs(fftshift(yt)));
xlabel('Frequency'); 
title('FFT of y Comfortably Numb')

figure(2)
plot((1:n)/Fs,(s));
xlabel('Time [sec]'); 

%%
%p8 = audioplayer((s),Fs); playblocking(p8)

