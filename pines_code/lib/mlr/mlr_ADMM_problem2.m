function [ psi ] = mlr_ADMM_problem2( beta, lambda, alpha, u, r )
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

vecForThresholding = beta + u / r;

thr = lambda*alpha;

psi = softThreshold(vecForThresholding, thr);

%fast_sthresh = @(x,th) sign(x).*max(abs(x) - th,0);
%psi2 = fast_sthresh(beta, thr);

end
