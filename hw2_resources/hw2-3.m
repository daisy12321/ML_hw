addpath('srcs', 'hw2_resources', 'hw2_resources/data')
%%% 3.1 %%%%
name = 'titanic';
data = importdata(strcat('data/data_',name,'_train.csv'));

X = data(:,1:11); 
[n, p] = size(X);
X_scaled = scale(X);
Y = data(:,12);

[z, f] = lr_run(X_scaled, Y, 0.1, true)
predictLR = @(x) sigmoid(z(2:p+1) * x + z(1));

disp('======Validation======');
% load data from csv files
validate = importdata(strcat('data/data_',name,'_validate.csv'));
X_val = validate(:,1:11);
X_val_scaled = scale(X_val);
Y_val = validate(:,12);


Y_pred(predictLR(X_val_scaled')>.5) = 1;
Y_pred(predictLR(X_val_scaled')<=.5) = -1;
valid_score = sum(Y_pred == Y_val')/length(Y_val);
valid_score
