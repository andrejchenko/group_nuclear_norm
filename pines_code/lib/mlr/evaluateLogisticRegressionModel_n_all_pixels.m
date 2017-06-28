function [ovAcc,logLik,prob_values,predict_label] = evaluateLogisticRegressionModel_n_all_pixels( testData, testLabels, beta ,r,c)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
theProbs = calculateLogisticRegressionProbs(testData, beta);

[val, predict_label] = max(theProbs);
%theProbs = calculateLogisticRegressionProbsFull(testData, beta);
n = size(testData,1);

nclass = size(beta,1) + 1;
confMatrix = zeros(nclass,nclass);

logLik = 0;

% for i=1:n
%     [~,ind] = max(theProbs(:,i));
%     confMatrix(ind,testLabels(i)) = confMatrix(ind,testLabels(i)) + 1;
%     classProb = theProbs(testLabels(i), i);
%     classProb = min(classProb, 1.0);
%     classProb = max(classProb, 1e-10);
%     logLik = logLik + log(classProb);
% end
% 
% ovAcc = sum(diag(confMatrix)) / n;
ovAcc = 0;

%kappa(confMatrix,1);

%fprintf('Overall accuracy on test data: %f, confusion matrix: \n', ovAcc);

%disp(confMatrix);

[classes,~] = size(theProbs);           
prob_values = reshape(theProbs,r,c,classes);
predict_label = reshape(predict_label,r,c);

end
