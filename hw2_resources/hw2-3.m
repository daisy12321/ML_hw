addpath('srcs', 'hw2_resources', 'hw2_resources/data')
%%% 3.1 %%%%
name = 'titanic';

[X, Y] = readdata(name, 'train', true);
data = importdata(strcat('data/data_',name,'_train.csv'));

X = data(:,1:11); 
[n, p] = size(X);
X_scaled = scale(X);
Y = data(:,12);

[z, f] = lr_run(X_scaled, Y, 0, true)
predictLR = @(x) sigmoid(z(2:p+1) * x + z(1));

disp('======Validation======');
% load data from csv files
validate = importdata(strcat('data/data_',name,'_validate.csv'));
X_val = validate(:,1:11);
X_val_scaled = scale(X_val);
Y_val = validate(:,12);
valid_score = get_accu(z, X_val_scaled, Y_val)


% find best lambda
LAMBDA_RANGE = (-3:1:6);
valid_scores = zeros(length(LAMBDA_RANGE), 4);
test_scores = zeros(length(LAMBDA_RANGE), 4);

for i = 1:length(LAMBDA_RANGE)
    lambda_i = 10^(LAMBDA_RANGE(i));
    [w_reg1_1, tmp, valid_scores(i, 1), test_scores(i, 1)]= lr_test('stdev1', lambda_i, false);
    [w_reg2_1, tmp, valid_scores(i, 2), test_scores(i, 2)]= lr_test('stdev2', lambda_i, false);
    [w_reg3_1, tmp, valid_scores(i, 3), test_scores(i, 3)]= lr_test('stdev4', lambda_i, false);
    [w_reg4_1, tmp, valid_scores(i, 4), test_scores(i, 4)]= lr_test('nonsep', lambda_i, false);
end
valid_scores


valid_score = get_accu(z, X_val_scaled, Y_val)
