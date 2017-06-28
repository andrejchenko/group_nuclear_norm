function [ beta ] = mlr_ADMM( Xtrain, Ytrain, Xtest, Ytest, lambda, alpha, r , YtrainInt)
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here

nclass = max(Ytrain(:));
nfeat = size(Xtrain, 2);
nParam = (nclass-1)*nfeat;

L = zeros(nParam, 1);
psi = randn(nParam, 1);

%nIter = 1000;
nIter = 140;

verbose = 1;

for i=1:nIter
    %betaMex = mlrADMMcontinuous(Xtrain, YtrainInt, lambda, alpha, r, psi, L);
    
    %betaMex = mlrAdmmSub1(Xtrain, YtrainInt, lambda, alpha, r, psi, L);
    
    beta = mlr_ADMM_problem1(Xtrain, Ytrain, lambda, alpha, r, psi, L, 1);
    
    %beta = betaMex;
    
    beta = reshape(beta',[],1);
    
    psi = mlr_ADMM_problem2(beta, lambda, alpha, L, r);
    
    L = L + r*(beta - psi);
    
    if verbose == 1
        wtAvgV = 0.5 * (beta + psi);
        wtAvg = reshape(wtAvgV, [nfeat,nclass-1])';
        [trainAcc, logLikTrain] = evaluateLogisticRegressionModel(Xtrain, Ytrain, wtAvg);
        [testAcc,~] = evaluateLogisticRegressionModel(Xtest, Ytest, wtAvg);
        
        penaltyTerm = lambda*(alpha*sum(abs(wtAvgV)) + 0.5*(1-alpha)*(wtAvgV'*wtAvgV));
        funValue = -logLikTrain + penaltyTerm;
        varDivergence = sum(abs(beta - psi));
        
        %fprintf('[ADMM] Iteration %i, ov. acc.: train %f, test %f, fun. value %f, var. divergence %f\n', i, trainAcc, testAcc, funValue, varDivergence);
        fprintf('[ADMM] Iteration %i, ov. acc.: test %f, fun. value %f \n', i, testAcc, funValue);
    end
end

beta = 0.5 * (beta + psi);

end
