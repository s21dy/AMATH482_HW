clear all;
close all;
clc;

%% Video 1: monte carlo
video = [];
v = VideoReader('monte_carlo_low.mp4');
while hasFrame(v)
    frame = readFrame(v);
    frame = rgb2gray(frame);
    frame = reshape (frame, [], 1);
    video = [video, frame];
end
%%
n = v.numFrames;
h = v.height;
w = v.width;
video = reshape(video, [h*w,n]);
video = double(video);

v1 = video(:,1:end-1);
v2 = video(:,2:end);
[U, Sigma, V] = svd(v1, 'econ');
%%
energy = 0;
total = sum(diag(Sigma));
% how much energy we want our modes to capture.
% 75% does alright; 90% does very good.
threshold = 0.9; 
r = 0;
while energy < threshold
    r = r + 1;
    energy = energy + Sigma(r,r)/total;
end
%% DMD
r=1;
Sr = Sigma(1:r, 1:r);
Ur = U(:, 1:r);
Vr = V(:, 1:r);
Stilde = Ur'*v2*Vr*diag(1./diag(Sr));
[eV, D] = eig(Stilde);
mu = diag(D);
omega = log(mu);
Phi = v2*Vr/Sr*eV;
y0 = Phi\video(:,1);
v_modes = zeros(r,length(v1(1,:)));
for i = 1:length(v1(1,:))
    v_modes(:,i) = (y0.*exp(omega*i));
end

%%
v_dmd = Phi*v_modes;
v_dmd = abs(v_dmd);
v_sparse = v1 - v_dmd;

%%
figure(1)

for i = 1:12
    subplot(3,4,i)
    vidtype = floor((i-1)/4);
    timeframe = mod(i,4);
    if timeframe == 0
        timeframe = 4;
    end
    if vidtype == 0
        temp = video(:,timeframe*80);
    elseif vidtype == 1
        temp = v_dmd(:,timeframe*80); %background
    elseif vidtype == 2
        temp = v_sparse(:,timeframe*80); %forground
    end
    temp = reshape(temp, h, w);
    imagesc(temp);
    colormap(gray);
    axis off;
end
%% play vids

for i = 1:n-1
    subplot(3,1,1)
    og = reshape(video(:,i), h, w);
    imagesc(og);
    
    subplot(3,1,2)
    bg = reshape(v_dmd(:,i), h, w);
    imagesc(bg);
    
    subplot(3,1,3)
    fg = reshape(v_sparse(:,i), h, w); %forground
    imagesc(fg);

    colormap(gray);
    axis off;
    pause(.0000001)
end
