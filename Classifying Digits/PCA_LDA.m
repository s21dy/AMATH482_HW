clear all; close all; clc;
load fisheriris

[train_image, train_label] = mnist_parse('train-images-idx3-ubyte', 'train-labels-idx1-ubyte');
train_image = im2double(reshape(train_image, size(train_image,1)*size(train_image,2), []).');
train_label = im2double(train_label);
train_image = train_image'; %784* 60000

[test_image,  test_label] = mnist_parse('t10k-images-idx3-ubyte', 't10k-labels-idx1-ubyte');
test_image = im2double(reshape(test_image, size(test_image,1)*size(test_image,2), []).');
test_label = im2double(test_label);
test_image = test_image';


% substract row-wise mean
mn = mean(train_image,2); % compute mean for each row
train_image = double(train_image)-repmat(mn,1,length(train_image)); % subtract mean

% Do Singular Value Decomposition(SVD) of matrix 
% piled with 60,000 vectorized images and get Principal images
[U, S, V ] = svd(train_image, 'econ');

energy = 0;
total = sum(diag(S));
% how much energy we want our modes to capture.
% 75% does alright; 90% does very good.
threshold = 0.9; 
r = 0;
while energy < threshold
    r = r + 1;
    energy = energy + S(r,r)/total;
end

rank = r;

train_image = (U(:, 1:rank))'*train_image; %project on to PCA compenent
test_image = (U(:, 1:rank))'*test_image;

lamda = diag(S).^2;

%% Plot in PCA space
for i = [0,7]
    Projection_idx = train_image(:, find(train_label == i));
    plot3(Projection_idx(1,:), Projection_idx(2,:), Projection_idx(3,:),...
          'o', 'DisplayName', num2str(i)); 
    hold on
end
legend show
%legend('0', '1', '2', '3','4', '5', '6','7', '8', '9')
%xlabel('Mode 1')
%ylabel('Mode 2') 
%zlabel('Mode 3') 

%% Variable Initialization

scrsz = get(groot, 'ScreenSize'); % Get screen width and height

% LDA Step 1. Prepare data matrix
X = train_image;
T = test_image;

% Retrieve data
dimension = size(train_image,1);
Sw = zeros(dimension);
Sb = zeros(dimension);   % Could consider sparse here
N = size(train_image, 2); % Train data size
Nt = size(test_image, 2); % Test data size
Mu = mean(train_image, 2);  % Get mean vector of train data

for i = [0,7] % construct matrix 
    
    % LDA Step 2. Construct Si matrix of each category
    mask = (train_label ==  i);
    x = X(:, mask);
    ni = size(x, 2);
    pi = ni / N;
    mu_i = mean(x, 2);

    Si = (x - repmat(mu_i, [1,ni]))*(x - repmat(mu_i, [1,ni]))';

    % LDA Step 3. Construct Sw within class covariance
    Sw = Sw + Si ;

    % LDA Step 4. Construct Sb between class covariance
    Sb = Sb + (mu_i - Mu) * (mu_i - Mu)';
end

% LDA Step 5. Singular Value Decomposition of Sw\Sb
M = pinv(Sw) * Sb;  % Sw maybe singular, use pseudo-inverse
[U, D, V] = svd(M);

% ===========================
%% Plot out the classifier
%
disp('Task 1: Visualize projected data to 2D and 3D plots');

G2 = U(:,1:rank);
Y2 = G2' * X;

% Plot 2d figure
data2d_fig = figure('Name', '2-D Plot');
set(data2d_fig, 'Position', [60 60 scrsz(3)-120 scrsz(4) - 140]);

for number = [0,7]
    
    mask = (train_label ==  number);
    a = Y2(1,mask);
    b = Y2(2,mask);
    
    d = [a'; b'];
    % Draw 2D visualization in separate view
    plot(d, 1*number*ones(size(d)),'o',...
        'DisplayName', num2str(number)); hold on 
    title(['LDA classifier']);
    
end
legend show

%xlim([-0.02 0.02])    
 ylim([-1 number+1])       

 %% Accuracy Test for LDA
na = 0;
nb = 7;

Y = G2' * X;
Y_t = G2'* T;

train_n= Y(:, find(train_label == na|train_label ==nb));
test_n = Y_t(:, find(test_label == na |test_label ==nb)); 

accuracy = classifyNN(test_n, train_n,...
    test_label(find(test_label == na |test_label ==nb)), ...
    train_label(find(train_label == na |train_label ==nb)));
  
%% Defining function 

function [accuracy] = classifyNN(test_data, train_data, test_label, train_label)
%
% Description:  
% Classify test data using Nearest Neighbor method withEuclidean distance
% criteria. 
% 
% Usage:
% [accuracy] = classifyNN(test_data, train_data, test_label, train_label)
%
% Parameters:
% test_data = test images projected in reduced dimension  dxtn
% train_data = train images projected in reduced dimension dxN
% test_label = test labels for each data tn x 1
% train_label = train labels for each train data Nx1
%
% Returns:
% accuracy: a scalar number of the classification accuracy

train_size = size(train_data, 2);
test_size = size(test_data, 2);
counter = zeros(test_size, 1);

parfor test_digit = 1:1:test_size

    test_mat = repmat(test_data(:, test_digit), [1,train_size]);
    distance = sum(abs(test_mat - train_data).^2);
    [M,I] = min(distance);
    if train_label(I) == test_label(test_digit)
        counter(test_digit) = counter(test_digit) + 1;
    end
end

accuracy = double(sum(counter)) / test_size;
end
