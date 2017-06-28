function B = fun_grad_and_subrad_descent(D,X_s,lambda,MAX_ITER,numClasses,numPix)

X = D;
Y = X_s;

stepLen = 0.001;
%A0 = zeros(size(D,2),size(X{1},3));%A0 = zeros(size(D,2),9);  % A = p x 9 = atoms x number of neighbour points%A0 = randn(size(D,2),9);  % A = p x 9 = atoms x number of neighbour points
B0 = zeros(size(D,2),9);
B = B0;
funLast = 1e+9;
resid = funLast;
iterNum = 1;
residConvThr = 1e-9;
funValBest = 1e+20;

while iterNum < MAX_ITER && resid > residConvThr
    % f = g + h = 0.5 * |XB-Y|_F + lambda*sum_{j}(|B_j|_*)
    %calculate g = 0.5 * |XB-Y|_F -> forbenious norm of some matrix  g = 0.5 * norm((XB-Y),'fro')^2; or
    %calculate g = 0.5 * |XB-Y|_2 -> L2 norm of some matrix
    g = 0.5 * norm((X*B -Y),2)^2;
    
    %calculate h = lambda*sum_{j}(|B_j|_*)
    % in total for indian pines we have 16 groups so B_1 = 5 x 9, B_2 =5 x 9,...B_16 = 5 x 9;
    %h = lambda * sum(sumSingularValues(B_j))
    br = 1;
    sum_singValPerMat = zeros(numClasses,1);
    for j = 1: numClasses
        B_g{j} = B(br:j*5,:);
        br = br + 5;
        sinValues = svd(B_g{j});
        sum_singValPerMat(j) = sum(sinValues);
    end
    sum_Sing_values = sum(sum_singValPerMat);
    h = lambda * sum_Sing_values;
    f = g + h;
    
    if f < funValBest
        funValBest = f;
        B_Best = B; %betaBest = betaCurr;
    end
    %fprintf('Iteration %d, objective function %f \n', iterNum, f);
    %fprintf('Iteration %d, smooth term value:, %f \n', iterNum, g);
    
   % Gradient of the smooth g term:
    % (Positive) gradient
    % del_g = D'*(D*A - X_s);
    % (Negative) gradient
    del_g = X'*( Y - X*B); % D = m x p, D' = p x m, A = p x n: -> the gradient for all groups
    % Proximal map -> Matrix Soft thresholding of the singular values of the gradient of the smooth part for each group
    
    % With other words we do: Proximal Gradient Descent, and we have to additionally include
    % See: Optimization course from Tibshiriani, Proximal gradietn descent, acceleration folder (08-prox-grad, page 18)
    %We follow the proximal descent direction, as in the unconstrained case. 
    % If soft thresholding the whole matrix:
    %A = matrix_SoftThreshold(A - stepLen * del_g,stepLen*lambda,numClasses); 
    %A = matrix_SoftThreshold_whole(A + stepLen * del_g,stepLen*lambda); %% I use already the negative gradient, so here I will have a + sign
    B = matrix_SoftThreshold(B + stepLen * del_g,stepLen*lambda,numClasses); 
    iterNum = iterNum + 1;
    resid = abs(funLast - g);     %resid = abs(funLast - gBeta);
    %resid = abs(funLast - f);
    funLast = g;
end

B = B_Best;
fprintf('Best function value: %f \n', funValBest);

end