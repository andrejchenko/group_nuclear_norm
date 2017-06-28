function [predict_label,acc] = svmClassification(trainData,trainLabels, testData, testLabels,indian_pines_vec)
            %input: trainData,trainLabels, testData, testLabels
            % Select the optimal parameters: gamma -g and cots -c from the
            % lowest cross validation error value
            [bestG,bestC] = selectParams(trainData,trainLabels)
            
            % Train the LIB_SVM with the optimum parameters
            % C-SVM, RBF kernel, cost = ..., gamma = ..., -b - probabilistics
            
            %cmd = '-s 0 -t 2 -c 10 -g 0.07 -b 1';
            cmd = ['-s 0 -t 2 -c ', num2str(bestC), ' -g ', num2str(bestG), ' -q '];
            model = libsvmtrain(trainLabels, trainData, cmd);
           
            % Use the SVM model to classify the data
            [predict_label, accuracy, decisionVals] = libsvmpredict(testLabels, testData, model); % run the SVM model on the test data
            acc = accuracy(1)/100;
end


