function [X_train, Y_train_lab, Y_train, X_valid, Y_valid_lab, Y_valid, X_test, Y_test_lab, Y_test] = read_data(name)
    train = importdata(strcat('hw3_resources/data/',name,'_train.csv'));
    valid = importdata(strcat('hw3_resources/data/',name,'_validate.csv'));
    test = importdata(strcat('hw3_resources/data/',name,'_test.csv'));

    %%%% training %%%%
    [N, p] = size(train);
    X_train = [ones(N, 1), train(:,1:p-1)];
    Y_train_lab = train(:,p);
    Y_train = dummyvar(Y_train_lab);

    
    %%%% validation %%%%
    N_valid = size(valid, 1);
    X_valid = [ones(N_valid, 1), valid(:,1:p-1)];
    Y_valid_lab = valid(:,p);
    Y_valid = dummyvar(Y_valid_lab);
    
    
    %%%% testing %%%%
    N_test = size(test, 1);
    X_test = [ones(N_test, 1), test(:,1:p-1)];
    Y_test_lab = test(:,p);
    Y_test = dummyvar(Y_test_lab);
    
    