%% Thanks to http://www.ilovematlab.cn/thread-48175-1-1.html
%% Thanks to Faruto enhancement toolkit for libsvm

clear;
clc;

%% Read in the raw data
test_data_original = load('.\testing.data');
train_data_original = load('.\training.data');

%% Data preprocessing, the training set is divided into training data and test data, and normalized to the interval of [0,1]
[mtrain,ntrain] = size(train_data_original);

% Clear extreme data
delete = [];
for i=1:mtrain
    for j=1:ntrain-1
        if train_data_original(i,j)>12.5
           delete = [delete,i] ;
           break
        end
    end
end

train_data_original(delete,:) = [];
[mdelete,ndelete] = size(delete);

%% Extract the data
% labels = train_data_original(:,ntrain);
% data = train_data_original(:,1:ntrain-1);
% [mtest,ntest] = size(test_data_original);

% Random extraction
abstraction = train_data_original( randperm(mtrain-ndelete,10000),: );
test_labels = abstraction(1:5000,ntrain);
test_data = abstraction(1:5000,1:ntrain-1);
train_labels = abstraction(5001:10000,ntrain);
train_data = abstraction(5001:10000,1:ntrain-1);

%% Equal uniform extraction
% test_labels = [labels(mtrain/6+1:mtrain/3);labels(mtrain/2+1:2*mtrain/3);...
%     labels(5*mtrain/6+1:mtrain)];
% test_data = [data(mtrain/6+1:mtrain/3,1:ntrain-1);...
%     data(mtrain/2+1:2*mtrain/3,1:ntrain-1);data(5*mtrain/6+1:mtrain,1:ntrain-1)];
% train_labels = [labels(1:mtrain/6);labels(mtrain/3+1:mtrain/2);...
%     labels(2*mtrain/3+1:5*mtrain/6)];
% train_data = [data(1:mtrain/6,1:ntrain-1);...
%     data(mtrain/3+1:mtrain/2,1:ntrain-1);data(2*mtrain/3+1:5*mtrain/6,1:ntrain-1)];

%dataset = [train_data;test_data];
[mtrain,ntrain] = size(train_data);
[mtest,ntest] = size(test_data);

%% Raw data visualization
figure;
boxplot(train_data,'orientation',0);
grid on;
title('Visualization for original data');
figure;
for i = 1:length(train_data(:,1))
    plot(train_data(i,1),train_data(i,2),'r*');
    hold on;
end
grid on;
title('Visualization for 1st dimension & 2nd dimension of original data');

%% ScaleForSVM comes with the Faruto enhancement toolkit
% [train_scale,test_scale] = scaleForSVM(train_data,test_data,0,1);

%% mapminmax is MATLAB's built-in normalization function :[0,1] interval normalization: y=(x-xmin)/(xmax-xmin)
%% normalizes the column vectors, so take transpose
% train_scale = mapminmax(train_data',0,1);
% train_scale = train_scale';
% test_scale = mapminmax(test_data',0,1);
% test_scale = test_scale';

%% Normal normalization
% train_scale = zscore(train_data);
% test_scale = zscore(test_data);
mean = mean(train_data);
std = std(train_data);
for i = 1:mtrain
    for j = 1:ntrain
        train_scale(i,j) = (train_data(i,j)-mean(j))/std(j);
    end
end
for i = 1:mtest
    for j = 1:ntest
        test_scale(i,j) = (test_data(i,j)-mean(j))/std(j);
    end
end


%% Visualization after Normalization
figure;
for i = 1:length(train_scale(:,1))
    plot(train_scale(i,1),train_scale(i,2),'r*');
    hold on;
end
grid on;
title('Visualization for 1st dimension & 2nd dimension of scale data');

%% Dimension reduction preprocessing: use the Faruto enhancement toolkit with its own function pcaForSVM
% [train_pca,test_pca] = pcaForSVM(train_scale,test_scale,90);

% The parameters c and g are optimized
  [bestacc,bestc,bestg] = SVMcgForClass(train_labels,train_scale,-5,5,-5,5,3,0.5,0.5,0.9);
  cmd = ['-c ',num2str(bestc),' -g ',num2str(bestg)];

% Classification and prediction based on SVM
% SVM training uses RBF function as the kernel function
model = svmtrain(train_labels, train_scale, cmd);

%% SVM prediction
[predict_labels, accuracy, dec_values] = svmpredict(test_labels, test_scale, model);
 
%% Results analysis

%ROC
 figure;
 auc = plot_roc(predict_labels,test_labels);
 
%% Actual classification and predictive classification graph of test set
figure;
hold on;
plot(test_labels(1:100),'o');
plot(predict_labels(1:100),'r*');
xlabel('Test set sample','FontSize',12);
ylabel('Category label','FontSize',12);
ylim([-5,5]);
legend('Actual test set classification','Forecast test set classification');
title('The actual classification and prediction classification diagram of the test set','FontSize',12);
grid on;
