function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

% Initialize variables
choice = [0 0.3 0.6 1.2 2.4 4.8 9.6 19.2 38.4]';

minError = inf;
curC = inf;
curSigma = inf;

steps = size(choice,1);

for i = 1:steps
    for j = 1:steps
        model = svmTrain(X, y, choice(i), @(x1, x2) gaussianKernel(x1, x2, choice(j)));
        predictions = svmPredict(model, Xval);
        error = mean(double(predictions ~= yval));
        if error < minError
            curC = choice(i);
            curSigma = choice(j);
        end
    end
end

C = curC;
sigma = curSigma;
        
% =========================================================================

end
