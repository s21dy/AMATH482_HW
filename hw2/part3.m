clear all; close all; clc

%%

[y, Fs] = audioread('Floyd.m4a');
% p8 = audioplayer(y,Fs); playblocking(p8); 

y = y(1:end-1);

v = y.'/10;
n = length(v);
L = n/Fs/6;% record time in seconds
a = 20;
k=(1/(L))*[0:n/2-1 -n/2:-1]; 
ks=fftshift(k);
index = find(ks > 0 & ks < 800);

tfinal1 =n/Fs;
t = (1:n)/Fs;
tslide = 0:0.1:tfinal1;

%% Gabor 

Sgt_spec = zeros(length(tslide), length(index)); 

for n = 1:length(tslide)
    %Gaussian window
    g = exp(-a*(t-tslide(n)).^2); 
     
    Sg = g.*v;
    Sgt = fft(Sg);
    fftshtsgt = abs(fftshift(Sgt));
    Sgt_spec(n,:) = fftshtsgt(index); 
end
Sgt_spec = Sgt_spec(:, index); % trim the matrix so ploting can be faster

%%
figure()

pcolor(tslide,ks(index),log(Sgt_spec.'+1))
shading interp 
colormap('hot')
set(gca,'Ylim',[20 200],'Fontsize',16)
xlabel('Time(sec)'); ylabel('Frequency(Hz)');

