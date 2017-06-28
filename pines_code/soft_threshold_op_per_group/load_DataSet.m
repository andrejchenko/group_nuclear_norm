function [D,trainData2,trainLabels,testData1,testLabels,T1] = load_DataSet()
    %[D,trainData2,X,trainLabels,testData,T1,testLabels] = load_Data();
    load('/home/vera/Documents/MATLAB/Projects/tddl_group_nuclear/data/pines/data_with_indices/Dict.mat');
    %load('D:\Projects\Matlab\tddl_group_nuclear\data\pines\testData.mat')
    load('/home/vera/Documents/MATLAB/Projects/tddl_group_nuclear/data/pines/data_with_indices/testData1.mat')
    load('/home/vera/Documents/MATLAB/Projects/tddl_group_nuclear/data/pines/data_with_indices/testLabels.mat')
    load('/home/vera/Documents/MATLAB/Projects/tddl_group_nuclear/data/pines/data_with_indices/trainLabels.mat')
    load('/home/vera/Documents/MATLAB/Projects/tddl_group_nuclear/data/pines/data_with_indices/X.mat')
    
    load('/home/vera/Documents/MATLAB/Projects/tddl_group_nuclear/data/pines/data_with_indices/T1.mat')    


    trainData2 = X{1}; %testData1 is already loaded     
    trainLabels = trainLabels{1};
    testLabels = testLabels{1};
    D = Dict{1}';
end