clear;
load('A:\Projects\Matlab\data\same_as_przemek\trainLabels.mat')
load('A:\Projects\Matlab\data\same_as_przemek\trainData.mat')
load('A:\Projects\Matlab\data\same_as_przemek\testLabels.mat')
load('A:\Projects\Matlab\data\same_as_przemek\testData.mat')

%load('A:\Projects\Matlab\data\trainLabels.mat')
%load('A:\Projects\Matlab\data\trainData.mat')
%load('A:\Projects\Matlab\data\testLabels.mat')
%load('A:\Projects\Matlab\data\testData.mat')
%beta = mlr_ADMM( trainAlphas, a_trainLabels, testAlphas, a_testLabels, lambda, alpha, 1.0 , labelsTrainInt)
%testAlphas = testAlphas';

lambda = 0.505971;
alpha = 0.3;
verbose = 1;

Xmu=mean(trainData);
Xsd=std(trainData);
trainStd=trainData- repmat(Xmu,size(trainData,1),1);
trainStd = trainStd ./ repmat(Xsd,size(trainStd,1),1);
trainStd=horzcat(ones(size(trainStd,1),1),trainStd);
labelsTrainInt=int32(trainLabels);

testStd=testData - repmat(Xmu,size(testData,1),1);
testStd = testStd ./ repmat(Xsd,size(testStd,1),1);
testStd=horzcat(ones(size(testStd,1),1),testStd);

[ beta ] = mlr_ADMM( trainStd, trainLabels, testStd, testLabels, lambda, alpha, 1.0 , labelsTrainInt)

