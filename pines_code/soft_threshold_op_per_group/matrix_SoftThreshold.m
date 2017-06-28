function B = matrix_SoftThreshold_whole(B,thr)

    [U,S,V]  = svd(B);
    %Threshold the singular values of the whole matrix B:
    minimum = min(size(S,1),size(S,2));
    for j = 1: minimum
        if(S(j,j) > thr)
           S(j,j) =  S(j,j) - thr;
        elseif S(j,j) >= -thr && S(j,j) <= thr
            S(j,j) = 0;
        elseif S(j,j) < - thr
            S(j,j) = S(j,j) + thr;
        end
    end
    B =  U* S*V'; % reconstruct B
end
