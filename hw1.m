% Clean workspace
clear all; close all; clc

load subdata/subdata.mat % Imports the data as the 262144x49 (space by time) matrix called subdata

L = 10; % spatial domain
n = 64; % Fourier modes

x2 = linspace(-L,L,n+1); x = x2(1:n); y =x; z = x;
k = (2*pi/(2*L))*[0:(n/2 - 1) -n/2:-1]; 
ks = fftshift(k);

[X,Y,Z]=meshgrid(x,y,z);
[Kx,Ky,Kz]=meshgrid(ks,ks,ks);

Uave = zeros(n,n,n);
for j=1:49
    %M = max(abs(Un),[],'all');
    % close all, isosurface(X,Y,Z,abs(Un)/M,0.7)
    % axis([-20 20 -20 20 -20 20]), grid on, drawnow
    % pause(1)
    Utn(:,:,:) = fftn(reshape(subdata(:,j),n,n,n));
    Uave = Uave + Utn;
end

Uave = fftshift(Uave)./49;
[val,idx] = max(Uave(:));
[a,b,c] = ind2sub(size(Uave),idx);

x0 = Kx(a,b,c);
y0 = Ky(a,b,c);
z0 = Kz(a,b,c);

%isosurface(Kx,Ky,Kz,abs(Uave)./max(abs(Uave(:))),0.7)

%axis([-20 20 -20 20 -20 20])
%xlabel('Kx')
%ylabel('Ky')
%zlabel('Kz')
%title('frequency')


%%

A = [];
B = [];
C = [];

tau = -.5;
filter=exp(tau*(((Kx-x0).^2)+((Ky-y0).^2)+((Kz-z0).^2)));

for j = 1:49
    Un = fftn(reshape(subdata(:,j),n,n,n));
    Unt = fftshift(Un);
    Unft = filter.*Unt;
    Unf = ifftn(Unft);
    
    [val,idx] = max(Unf(:));
    [b,a,c] = ind2sub(size(Unf),idx);
    A(j) = a;
    B(j) = b;
    C(j) = c;
end

figure(2)
plot3(x(A),y(B),z(C), 'Linewidth', 2)

xlabel('x')
ylabel('y')
zlabel('z')
title('Submarine Movement Trajectory')
x49 = x(A(end));
y49 = y(B(end));
z49 = z(C(end));


hold on
plot3(x49, y49, z49, 'r*')

sprintf('x: %s, y: %f, z: %d', x49,y49,z49)

%%

result= [x(A); y(B);]';



