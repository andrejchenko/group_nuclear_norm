function test_fun_grad_and_subrad_descent()
clc;clear;
c = 0; 
[D,trainData2,trainLabels,testData1,testLabels,T1] = load_DataSet();

MAX_ITER = 5000;
numPix = 5;
 
lambda = [0.1];
acc_l_0_001 = 0;
numClasses = 16; % for the indian pines imagenumPix = 5;
    for j = 1: length(lambda)
        %lambda = 0.01;
        alphasP_train = [];
        alphasP_test = [];
        c = c+1;
        for i = 1: size(trainData2,3) % for each point in the training data, apply low group rank sparse coding
            AhatOne =      fun_grad_and_subrad_descent_threshold(D,trainData2(:,:,i),lambda(j),MAX_ITER,numClasses,numPix);
            %Ahat =        least_squares_with_gradient_descent(D,trainData2(:,:,i),lambda,MAX_ITER,numClasses);
            abunFromCenPix = AhatOne(:,1);
            alphasP_train = [alphasP_train abunFromCenPix];
            i
        end
        for i = 1: size(T1,3) % for each point in the training data, apply low group rank sparse coding

            Ahat_testOne = fun_grad_and_subrad_descent_threshold(D,T1(:,:,i),lambda(j),MAX_ITER,numClasses,numPix); 
            abunFromCenPix = Ahat_testOne(:,1);
            alphasP_test = [alphasP_test abunFromCenPix];
            i
        end

        lambda_EN = 0;
        alpha_EN = 0;
        theBeta=multinomialLogisticRegressionL1(alphasP_train',trainLabels,lambda_EN,alpha_EN,0); 
        acc = evaluateLogisticRegressionModel(alphasP_test',testLabels,theBeta);
        acc_l_0_001(j) = acc*100; 
        save acc_l_0_001 acc_l_0_001
        
        %[predict_label,acc1] = svmClassification(alphasP_train',trainLabels, alphasP_test', testLabels,1);
    end
end


function [D,trainData2,trainLabels,testData1,testLabels,T1] = load_DataSet()
    %[D,trainData2,X,trainLabels,testData,T1,testLabels] = load_Data();
    load('D:\Projects\Matlab\tddl_group_nuclear\data\pines\data_with_indices\Dict.mat');
    %load('D:\Projects\Matlab\tddl_group_nuclear\data\pines\testData.mat')
    load('D:\Projects\Matlab\tddl_group_nuclear\data\pines\data_with_indices\testData1.mat')
    load('D:\Projects\Matlab\tddl_group_nuclear\data\pines\data_with_indices\testLabels.mat')
    load('D:\Projects\Matlab\tddl_group_nuclear\data\pines\data_with_indices\trainLabels.mat')
    load('D:\Projects\Matlab\tddl_group_nuclear\data\pines\data_with_indices\X.mat')
    
    load('D:\Projects\Matlab\tddl_group_nuclear\data\pines\data_with_indices\T1.mat')    

    trainData2 = X{1}; %testData1 is already loaded     
    trainLabels = trainLabels{1};
    testLabels = testLabels{1};
    D = Dict{1}';
end

function [D,trainData2,X,trainLabels,testData,T1,testLabels] = load_Data()
    load('D:\Projects\Matlab\tddl_group_nuclear\data\pines\data_with_indices\Dict.mat');
    %load('D:\Projects\Matlab\tddl_group_nuclear\data\pines\testData.mat')
    load('D:\Projects\Matlab\tddl_group_nuclear\data\pines\data_with_indices\testData1.mat')
    load('D:\Projects\Matlab\tddl_group_nuclear\data\pines\data_with_indices\testData.mat')
    load('D:\Projects\Matlab\tddl_group_nuclear\data\pines\data_with_indices\testLabels.mat')
    load('D:\Projects\Matlab\tddl_group_nuclear\data\pines\data_with_indices\trainLabels.mat')
    load('D:\Projects\Matlab\tddl_group_nuclear\data\pines\data_with_indices\X.mat')
    load('D:\Projects\Matlab\tddl_group_nuclear\data\pines\data_with_indices\T1.mat')

    trainData2 = X{1}; %testData1 is already loaded 
    trainLabels = trainLabels{1};
    testLabels = testLabels{1};
    D = Dict{1}';


%     load('D:\Projects\Matlab\tddl_group_nuclear\data\pines\Dict.mat')
%     %load('D:\Projects\Matlab\tddl_group_nuclear\data\pines\T.mat')
%     load('D:\Projects\Matlab\tddl_group_nuclear\data\pines\T1.mat')
%     load('D:\Projects\Matlab\tddl_group_nuclear\data\pines\testData.mat')
%     load('D:\Projects\Matlab\tddl_group_nuclear\data\pines\testLabels.mat')
%     %load('D:\Projects\Matlab\tddl_group_nuclear\data\pines\testNeighbours.mat')
%     load('D:\Projects\Matlab\tddl_group_nuclear\data\pines\trainData.mat')
%     load('D:\Projects\Matlab\tddl_group_nuclear\data\pines\trainLabels.mat')
%     %load('D:\Projects\Matlab\tddl_group_nuclear\data\pines\trainNeighbours.mat')
%     load('D:\Projects\Matlab\tddl_group_nuclear\data\pines\X.mat')
%     
%     trainData2 = X{1}(:,1,:);
%     trainData2 = squeeze(trainData2);
%     trainLabels = trainLabels{1};
%     
%     testData = T1(:,1,:);
%     %testData = T1(:,1,:);
%     %testData = testData(:,1,:);
%     testData = squeeze(testData);
%     testLabels = testLabels{1};
%     D = Dict{1}';
end
