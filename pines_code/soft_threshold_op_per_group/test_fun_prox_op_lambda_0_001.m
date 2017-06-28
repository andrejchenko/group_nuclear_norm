function test_fun_prox_op_lambda_0_001()
clc;clear;
c = 0; 
[D,trainData2,trainLabels,testData1,testLabels,T1] = load_DataSet();

MAX_ITER = 5000;
numPix = 5;
%stepLen = 0.001;
 
lambda = [0.001];
acc_l_0_001 = 0;
numClasses = 16; % for the indian pines imagenumPix = 5;
    for j = 1: length(lambda)
        %lambda = 0.01;
        alphasP_train = [];
        alphasP_test = [];
        c = c+1;
        for i = 1: size(trainData2,3) % for each point in the training data, apply low group rank sparse coding
            AhatOne =      fun_prox_op(D,trainData2(:,:,i),lambda(j),MAX_ITER,numClasses,numPix);
            %Ahat =        least_squares_with_gradient_descent(D,trainData2(:,:,i),lambda,MAX_ITER,numClasses);
            abunFromCenPix = AhatOne(:,1);
            alphasP_train = [alphasP_train abunFromCenPix];
            i
        end
        for i = 1: size(T1,3) % for each point in the training data, apply low group rank sparse coding

            Ahat_testOne = fun_prox_op(D,T1(:,:,i),lambda(j),MAX_ITER,numClasses,numPix); 
            abunFromCenPix = Ahat_testOne(:,1);
            alphasP_test = [alphasP_test abunFromCenPix];
            i
        end

        lambda_EN = 0;
        alpha_EN = 0;
        theBeta=multinomialLogisticRegressionL1(alphasP_train,trainLabels,lambda_EN,alpha_EN,0); 
        acc = evaluateLogisticRegressionModel(alphasP_test',testLabels,theBeta);
        acc_l_0_001(j) = acc*100; 
        save acc_l_0_001 acc_l_0_001
    end
end

