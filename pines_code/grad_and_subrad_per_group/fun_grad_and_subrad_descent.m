function B = fun_grad_and_subrad_descent(D,X_s,lambda,MAX_ITER,numClasses,numPix)

X = D;
Y = X_s;

stepLen = 0.001;
B0 = zeros(size(D,2),9); B = B0;
funLast = 1e+9;
resid = funLast;
iterNum = 1;
residConvThr = 1e-4;
%residConvThr = 1e-9;
funValBest = 1e+20;
fValues = zeros(iterNum,1);
while iterNum < MAX_ITER && resid > residConvThr
    g = 0.5 * norm((X*B -Y),2)^2; %smooth term

    br = 1;
    sum_singValPerMat = zeros(numClasses,1);
     for j = 1: numClasses
        sel_group = br:j*numPix;
        [U_b,S_b,V_b] = svd(B(sel_group,:));           % svd of the parameter B, for the function value, right?  
        [r,c] = size(S_b);
        minRank = 0;
        if(r<c)
            minRank = r;
            else minRank = c;
        end
        S_b = S_b(1:minRank,1:minRank);
        sum_singValPerMat(j) = sum(diag(S_b));
     end
    sum_Sing_values = sum(sum_singValPerMat);
    h = lambda * sum_Sing_values;
    f = g + h;
    del_g = - X'*( Y - X*B);      %gradient of the smoth term
    
    %Subgradient of the non-smooth term:
    br = 1;
    for j = 1: numClasses
        sel_group = br:j*numPix;
        del_g_group{j} = del_g(sel_group,:);
        br = br + numPix;
        [U,S,V] = svd(del_g_group{j});                 % svd of grad_group_j as the subradient, right?
            
        [r,c] = size(S);
        minRank = 0;
        if(r<c)
            minRank = r;
        else minRank = c;
        end
        V = V(:,1:minRank);                             %V ->?? 
        grad{j} = del_g_group{j} + lambda *U*V';        % sum the gradient and subradient for group j
        B_group{j} = B(sel_group,:) - stepLen * grad{j};% update the j group of the B parameter 
        
        B(sel_group,:) = B_group{j};
        del_g = - X'*( Y - X*B); 
        
    end

    if f < funValBest
        funValBest = f;
        B_Best = B; 
    end
    %fprintf('Iteration %d, objective function %f \n', iterNum, f);
    %fprintf('Iteration %d, smooth term value:, %f \n', iterNum, g);

    fValues(iterNum) = f;
    iterNum = iterNum + 1;
    resid = abs(funLast - f);     %resid = abs(funLast - gBeta);
    %resid = abs(funLast - f);
    funLast = f;
    
end

B = B_Best;
fprintf('Best function value: %f \n', funValBest);

%x = 1:1:(iterNum-1);
%plot(x,fValues);
end