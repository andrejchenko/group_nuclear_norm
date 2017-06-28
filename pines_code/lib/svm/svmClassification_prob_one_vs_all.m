function [predict_label,acc,prob] = svmClassification_prob_one_vs_all(...
                                           trainData,trainLabels, testData, testLabels,indian_pines_vec,r,c,d,...
                                           numClasses)
             
            for k=1:numClasses
            % Select the optimal parameters for each model: gamma -g and cots -c from the lowest cross validation error value
                [bestG,bestC] = selectParams(trainData,double(trainLabels==k));
                cmd = ['-s 0 -t 2 -c ', num2str(bestC), ' -g ', num2str(bestG), ' -q ',' -b 1'];
                model{k} = libsvmtrain(double(trainLabels==k), trainData, cmd);
            end
            
            %# get probability estimates of test instances using each model
            numTest = size(testData,1);
            prob = zeros(numTest,numClasses);
            for k=1:numClasses
                [~,~,p] = libsvmpredict(double(testLabels==k), testData, model{k}, '-b 1');
                prob(:,k) = p(:,model{k}.Label==1);    %# probability of class==k
            end
            
            [maxProbVal,pred] = max(prob,[],2);
            acc = sum(pred == testLabels) ./ numel(testLabels);    %# accuracy
            
            testLabelsAll = zeros(size(indian_pines_vec,1),1);
            numTest = length(testLabelsAll);
            prob = zeros(numTest,numClasses);
            for k=1:numClasses
                [predict_label, accuracy, prob_values] = libsvmpredict(double(testLabelsAll==k), indian_pines_vec, model{k}, '-b 1');
                prob(:,k) = prob_values(:,model{k}.Label==1);    %# probability of class==k
            end
            
            prob = reshape(prob,r,c,numClasses);
            predict_label = reshape(predict_label,r,c);      
end


