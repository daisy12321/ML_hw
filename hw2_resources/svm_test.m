function valid_score = svm_test(name, C, kernel, sigma2, toPlot)
disp('======Training======');
% load data from csv files
data = importdata(strcat('data/data_',name,'_train.csv'));
X = data(:,1:2);
Y = data(:,3);

% Carry out training, primal and/or dual
[w, w_0] = svm(X, Y, C, kernel, sigma2)

% Define the predictSVM(x) function, which uses trained parameters
predictSVM = @(x) w * x + w_0;

disp('======Validation======');
% load data from csv files
validate = importdata(strcat('data/data_',name,'_validate.csv'));
X_val = validate(:,1:2);
Y_val = validate(:,3);

Y_pred(predictSVM(X_val')>0) = 1;
Y_pred(predictSVM(X_val')<=0) = -1;
valid_score = sum(Y_pred == Y_val')/length(Y_val);
disp('======Accuracy======');
disp(valid_score);

if toPlot
    hold on;

    % plot training results
    plotDecisionBoundary(X, Y, predictSVM, [-1, 0, 1], ...
        strcat('SVM Training, C = ', num2str(C)), ...
        strcat('../hw2_writeup/figures/hw2_2_',name,'_a_',num2str(C),'.pdf'));
    
    % plot validation results
    plotDecisionBoundary(X_val, Y_val, predictSVM, [-1, 0, 1],...
        strcat('SVM Validate, C = ', num2str(C)), ...
        strcat('../hw2_writeup/figures/hw2_2_',name,'_b_',num2str(C),'.pdf'));
end
