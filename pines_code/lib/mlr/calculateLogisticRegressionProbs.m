function [ probs ] = calculateLogisticRegressionProbs( X, beta, verbose)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
    n = size(X,1);
    nclass = size(beta,1)+1;
    probs = zeros(nclass,n);
    probsCmp = zeros(nclass,n);
    
    if nargin < 3
        verbose = 0;
    end
    
    for j=1:n
        if verbose
            fprintf('Doing example #%d', j);
            fprintf('%f ', X(j,:));
            fprintf('\n');
        end
        
        ps = zeros(nclass-1,1);
        ps0 = zeros(nclass-1,1);
        for i=1:nclass-1
            ps0(i) = exp(beta(i,:)*X(j,:)');
            ps(i) = beta(i,:)*X(j,:)';
            if isnan(ps(i))
                fprintf('nan detected in probability calculation !');
            end
            if verbose
                fprintf('\texp for class %d: %f\n', i, ps0(i));
            end
        end
        
        maxExp = max(ps(:));
        if maxExp < 0
           maxExp = 0;
        end
        
        for i=1:nclass-1
            ps(i) = exp(ps(i) - maxExp);
            if isnan(ps(i))
                fprintf('nan detected in probability calculation !');
            end
        end
        
        sumP0 = sum(ps0(:)) + 1;
       
        sumP = sum(ps(:)) + exp(-maxExp);
        if isnan(sumP)
            fprintf('nan detected in probability calculation !');
        end
        if sumP > exp(50)
            fprintf('what happens ?\n');
        end
        
        for i=1:nclass-1
            probs(i,j) = ps0(i) / sumP0;
            probsCmp(i,j) = ps(i) / sumP;
            
            probs(i,j) = max(probs(i,j), 1e-9);
            probsCmp(i,j) = max(probsCmp(i,j), 1e-9);
            
%             if isnan(probs(i,j))
%                 fprintf('nan detected in probability calculation !');
%             end

            if verbose
                fprintf('\t\tprob for class %d: %f\n', i, probs(i,j));
            end
        end
        
         probs(nclass,j) = 1-sum(probs(1:nclass-1,j));
         probsCmp(nclass,j) = 1-sum(probsCmp(1:nclass-1,j));
         
         probs(nclass,j) = max(probs(nclass,j), 1e-9);
         probsCmp(nclass,j) = max(probsCmp(nclass,j), 1e-9);
         
         if isnan(probs(nclass,j))
            fprintf('nan detected in probability calculation !');
         end
        
         if verbose
            fprintf('\t\tprob for class %d: %f\n', nclass, probs(nclass,j));
         end
    end

    %probs = probsCmp;
end
