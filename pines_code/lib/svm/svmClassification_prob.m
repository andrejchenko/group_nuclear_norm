function [predict_label,acc,prob_values] = svmClassification_prob(trainData,trainLabels, testData, testLabels,indian_pines_vec,r,c,d)
            %input: trainData,trainLabels, testData, testLabels
            % Select the optimal parameters: gamma -g and cots -c from the
            % lowest cross validation error value
            [bestG,bestC] = selectParams(trainData,trainLabels)
            
            % Train the LIB_SVM with the optimum parameters
            % C-SVM, RBF kernel, cost = ..., gamma = ..., -b - probabilistics
            
            %cmd = '-s 0 -t 2 -c 10 -g 0.07 -b 1';
            cmd = ['-s 0 -t 2 -c ', num2str(bestC), ' -g ', num2str(bestG), ' -q ',' -b 1'];
            model = libsvmtrain(trainLabels, trainData, cmd);
           
            % Use the SVM model to classify the data
            [predict_label, accuracy, prob_values] = libsvmpredict(testLabels, testData, model,'-b 1'); % run the SVM model on the test data
            acc = accuracy(1)/100;
            testLabelsAll = zeros(size(indian_pines_vec,1),1);
            [predict_label, accuracy, prob_values] = libsvmpredict(testLabelsAll, indian_pines_vec, model,'-b 1');
            [~,classes] = size(prob_values);
            
            prob_values = reshape(prob_values,r,c,classes);
            predict_label = reshape(predict_label,r,c);
end


