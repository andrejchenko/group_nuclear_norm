function [ betas ] = mlr_ADMM_problem1( trainData, trainLabels, lambda, alpha, r, psi, u, verbose, betaInit)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
if nargin < 8
    verbose = 0;
end

nclass = max(trainLabels(:));
nfeat = size(trainData, 2);
nParam = (nclass-1)*nfeat;
n = size(trainData, 1);

X = trainData;
Y = trainLabels;

betaCurr = zeros(nclass-1,nfeat);
if nargin >= 9
    if length(betaInit) ~= length(betaCurr(:))
        error('mismatched beta dimensions !');
    end
    betaCurr = reshape(betaInit, [size(betaCurr,2),size(betaCurr,1)])';
end

funLast = 1e+9;
resid = funLast;

iterNum = 1;
maxIter = 3000;
%maxIter = 300;
residConvThr = 1e-9;
%stepLen = 1e-1;
%stepLen = 0.4;
stepLen = 0.001;
probs = zeros(nclass,n);

Ind = zeros(nclass-1,n);

for i=1:nclass-1
    Ind(i,:) = (Y==i);
end

rho = lambda*(1-alpha);

funValBest = 1e+20;
betaBest = betaCurr;

logLikAtBest = 0.0;
penaltyAtBest = 0.0;
betaPsiDiffAtBest = 0.0;

while iterNum < maxIter && resid > residConvThr
    
    probs = calculateLogisticRegressionProbs(X, betaCurr);

    Q0 = 0.0;
    for j=1:n
        Q0 = Q0 - log(probs(Y(j),j));
    end
    
    %betaFlat = reshape(betaCurr, [1,nParam]);
    betaFlat = reshape(betaCurr',[],1)';
    regTerm = (rho/2)*(betaFlat*betaFlat');
    betaPsiDiff = (betaFlat - psi');
    admmCnstrTerm = (r/2)*(betaPsiDiff*betaPsiDiff');
    admmDualTerm = u'*betaPsiDiff';
    gBeta = Q0 + regTerm + admmCnstrTerm + admmDualTerm;
    
    if verbose && iterNum == 1
        fprintf('Initial function value: %f, neg.log.lik. %f \n', gBeta, Q0);
    end
        
    if gBeta < funValBest
        funValBest = gBeta;
        betaBest = betaCurr;
        logLikAtBest = Q0;
        penaltyAtBest = regTerm;
        betaPsiDiffAtBest = (betaPsiDiff*betaPsiDiff');
    end
    
    %labelsTrainInt=int32(trainLabels);
    %fvalFromMex = mlrAdmmSub1(trainData, labelsTrainInt, lambda, alpha, r, psi, u, betaFlat);
    %fprintf('Iteration %d, obj. func. %f, value from mex: %f, diff: %f\n', iterNum, gBeta, fvalFromMex, (gBeta-fvalFromMex));
        
    if verbose > 0 && mod(iterNum,100) == 0
        %fprintf('Iteration %d, obj. func. %f, neg.log.lik. %f \n', iterNum, gBeta, Q0);
    end
    
    gradReg = rho*betaCurr;
    gradQ0 = (probs(1:nclass-1,:)-Ind)*X;
    gradAdmmCnstr = r*betaPsiDiff;
    gradAdmmCnstrMat = reshape(gradAdmmCnstr, [size(gradQ0,2),size(gradQ0,1)])';
    gradAdmmDual = reshape(u, [size(gradQ0,2),size(gradQ0,1)])';
    grad = gradQ0 + gradReg + gradAdmmCnstrMat + gradAdmmDual;
    
    %labelsTrainInt=int32(trainLabels);
    %gradFromMexV = mlrAdmmSub1(trainData, labelsTrainInt, lambda, alpha, r, psi, u, betaFlat);
    %gradFromMex = reshape(gradFromMexV, [size(grad,2),size(grad,1)])';
    %fprintf('Iteration %d, gradient diff norm: %f\n', iterNum, norm(grad - gradFromMex));
        
    betaCurr = betaCurr - stepLen * grad;
    
    iterNum = iterNum + 1;
    resid = abs(funLast - gBeta);
    funLast = gBeta;
end

betas = betaBest;
if verbose > 0
    fprintf('Best function value: %f, log. lik. %f, penalty %f, variable divergence norm %f\n', ...
        funValBest,logLikAtBest, penaltyAtBest,betaPsiDiffAtBest);
end
