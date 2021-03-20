clear all; close all; clc

%% P1 for GNR -- Define space and time domain

[y, Fs] = audioread('GNR.m4a');
% p8 = audioplayer(y,Fs); playblocking(p8); 

v = y.'/10;
n = length(v);
L = n/Fs; % record time in seconds
a = 500;
k=(1/(L))*[0:n/2-1 -n/2:-1]; 
ks=fftshift(k);
 
tfinal =n/Fs;
t = (1:n)/Fs;
tslide = 0:.1:tfinal;

%%
Sgt_spec = zeros(length(tslide), length(t)); 
%Max=  zeros(length(tslide), length(t));  % to 

for n = 1:length(tslide)
     g = exp(-a*(t-tslide(n)).^2);
     Sg = g.*v;
     Sgt = fft(Sg);
     Sgt_spec(n,:) =  fftshift(abs(Sgt)); 
     
     %[fmax, ind] = max(Sgt);
     %Max(n,:)= k(ind);
end

%%
%figure(1)
%plot(tslide, abs(Max),'o')
%xlabel('Time(sec)'); ylabel('Frequency(Hz)');
%set(gca,'Ylim',[0 800],'Fontsize',16)

%%

figure(2)
pcolor(tslide,ks,log(Sgt_spec.'+1))
shading interp 
colormap('hot')
set(gca,'Ylim',[0 800],'Fontsize',16)
xlabel('Time(sec)'); ylabel('Frequency(Hz)');

