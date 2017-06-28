function [ probs ] = calculateLogisticRegressionProbsFull( X, beta )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
 n = size(X,1);
    nclass = size(beta,1);
    probs = zeros(nclass,n);
    probsCmp = zeros(nclass,n);
    
    for j=1:n
        ps = zeros(nclass,1);
        ps0 = zeros(nclass,1);
        for i=1:nclass
            ps0(i) = exp(beta(i,:)*X(j,:)');
            ps(i) = beta(i,:)*X(j,:)';
            if isnan(ps(i))
                fprintf('nan detected in probability calculation !');
            end
        end
        
        maxExp = max(ps(:));
        if maxExp < 0
           maxExp = 0;
        end
        
        for i=1:nclass
            ps(i) = exp(ps(i) - maxExp);
            if isnan(ps(i))
                fprintf('nan detected in probability calculation !');
            end
        end
        
        sumP0 = sum(ps0(:));
       
        sumP = sum(ps(:));
        if isnan(sumP)
            fprintf('nan detected in probability calculation !');
        end
        if sumP > exp(50)
            fprintf('what happens ?\n');
        end
        
        for i=1:nclass
            probs(i,j) = ps0(i) / sumP0;
            probsCmp(i,j) = ps(i) / sumP;
            
            probs(i,j) = max(probs(i,j), 1e-9);
            probsCmp(i,j) = max(probsCmp(i,j), 1e-9);
            
%             if isnan(probs(i,j))
%                 fprintf('nan detected in probability calculation !');
%             end
        end
         
         if isnan(probs(nclass,j))
            fprintf('nan detected in probability calculation !');
        end
    end

    probs = probsCmp;

end

