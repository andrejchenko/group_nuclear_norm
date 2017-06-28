function [bestG,bestC] = selectParams(trainData,trainLabels)
            % input: trainData,trainLabels
            bestcv = 0;
            log2c = -1:1:20;
            log2g = -5:1:5;
            
            cGridLength = length(log2c);
            gGridLength = length(log2g);
            
            for indexc = 1:cGridLength,
                clc;
                %fprintf('Iteration %i of %i...',indexc,cGridLength);
                for indexg = 1:gGridLength,
                    cmd = ['-q -v 5 -c ', num2str(2^log2c(indexc),2), ' -g ', num2str(2^log2g(indexg))];
                    cv = libsvmtrain(trainLabels, trainData, cmd);
                    if (cv >= bestcv),
                        bestcv = cv; bestC = 2^log2c(indexc); bestG = 2^log2g(indexg);
                    end
                end
            end
end