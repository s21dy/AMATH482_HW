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

train_image = (U(:, 1:rank))'*train_image; %project onto PCA compenent
test_image = (U(:, 1:rank))'*test_image;
%% SVM Modeling

%t1 = templateSVM('KernelFunction','gaussian'); 
svm_model = fitcecoc(train_image',train_label);
%svm_model1 = fitcecoc(train_image',train_label,'Coding','onevsone','Learners',t);
%svm_model2 = fitcecoc(train_image',train_label,'Coding','onevsall','Learners',t);

result = predict(svm_model,test_image');
%result1 = predict(svm_model1,test_image');
%result2 = predict(svm_model2,test_image');
%svmL1 = loss(svm_model1, test_image.', test_label);
%svmL2 = loss(svm_model2, test_image.', test_label);

svmL = loss(svm_model2, test_image.', test_label);
figure()
bar(1-[svmL1 svmL2])
xticklabels( {'one vs one', 'one vs all'});
title("SVM accuracy without Cross Validation");

%% Decision Tree Modeling

rfMdl = fitctree(train_image.', train_label,'CrossVal','on');
rfL = kfoldLoss(rfMdl, 'LossFun','ClassifErr');
%view(rfMdl.Trained{1},'Mode','graph');
figure()
bar(1-rfL)
xticklabels([ "Accuracy of Decision Tree"]);
title("Decision Tree accuracy with Cross Validation");

%% Decided pair of digit
na = 1;
nb = 4;

train_2 = train_image(:, find(train_label == na|train_label == nb));
label_2 = train_label(find(train_label == na|train_label == nb));

test_2 = test_image(:, find(test_label == na|test_label == nb));
tlabel_2 = test_label(find(test_label == na|test_label == nb));

%% modeling SVM
svm_best = fitcecoc(train_2',label_2);
svml_best = loss(svm_best, test_2.', tlabel_2);

%% modeling Tree
rfMdl = fitctree(train_2.', label_2,'CrossVal','on');
             
rfL = kfoldLoss(rfMdl, 'LossFun','ClassifErr');
view(rfMdl.Trained{1},'Mode','graph');