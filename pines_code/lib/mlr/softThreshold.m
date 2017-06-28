function [ xthr ] = softThreshold( x,thr )
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
xthr=x;

for i=1:length(x)
    if x(i) > thr
        xthr(i) = x(i) - thr;
    elseif x(i) >= -thr && x(i) <= thr
        xthr(i) = 0;
    elseif x(i) < -thr
        xthr(i) = x(i) + thr;
    else
        disp('logic does not work anymore.');
    end
end

end
