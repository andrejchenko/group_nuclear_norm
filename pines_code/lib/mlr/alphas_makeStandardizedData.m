clear;
load('A:\Projects\Matlab\data\alphas as train and test data\a_testLabels.mat')
load('A:\Projects\Matlab\data\alphas as train and test data\a_trainLabels.mat')
load('A:\Projects\Matlab\data\alphas as train and test data\testAlphas.mat')
load('A:\Projects\Matlab\data\alphas as train and test data\trainAlphas.mat')
testAlphas = testAlphas';

lambda = 0.505971;
alpha = 0.3;
verbose = 1;

% Xmu=mean(trainAlphas);
% Xsd=std(trainAlphas);
% trainStd=trainAlphas- repmat(Xmu,size(trainAlphas,1),1);
% trainStd = trainStd ./ repmat(Xsd,size(trainStd,1),1);
% trainStd=horzcat(ones(size(trainStd,1),1),trainStd);
labelsTrainInt=int32(a_trainLabels);

% testStd=testAlphas - repmat(Xmu,size(testAlphas,1),1);
% testStd = testStd ./ repmat(Xsd,size(testStd,1),1);
% testStd=horzcat(ones(size(testStd,1),1),testStd);


%beta = mlr_ADMM( trainAlphas, a_trainLabels, testAlphas, a_testLabels, lambda, alpha, 1.0 , labelsTrainInt)
