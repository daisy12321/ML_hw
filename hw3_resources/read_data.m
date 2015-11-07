function [X_train, Y_train_lab, Y_train, X_valid, Y_valid_lab, Y_valid, X_test, Y_test_lab, Y_test] = read_data(name)
    train = importdata(strcat('hw3_resources/data/',name,'_train.csv'));
    valid = importdata(strcat('hw3_resources/data/',name,'_validate.csv'));
    test = importdata(strcat('hw3_resources/data/',name,'_test.csv'));

    %%%% training %%%%
    N = size(train, 1);
    X_train = [ones(N, 1), train(:,1:2)];
    Y_train_lab = train(:,3);

    K = max(Y_train_lab);
    
    Y_train = zeros(N,K);
    Y_train(:,1) = [Y_train_lab == ones(N,1)];
    Y_train(:,2) = [Y_train_lab == 2*ones(N,1)];
    Y_train(:,3) = [Y_train_lab == 3*ones(N,1)];
    
    %%%% validation %%%%
    N_valid = size(valid, 1);
    X_valid = [ones(N_valid, 1), valid(:,1:2)];
    Y_valid_lab = valid(:,3);
    
    Y_valid = zeros(N_valid,K);
    Y_valid(:,1) = [Y_valid_lab == ones(N_valid,1)];
    Y_valid(:,2) = [Y_valid_lab == 2*ones(N_valid,1)];
    Y_valid(:,3) = [Y_valid_lab == 3*ones(N_valid,1)];
    
    
    
    %%%% testing %%%%
    N_test = size(test, 1);
    X_test = [ones(N_test, 1), test(:,1:2)];
    Y_test_lab = test(:,3);
    
    Y_test = zeros(N_test,K);
    Y_test(:,1) = [Y_test_lab == ones(N_test,1)];
    Y_test(:,2) = [Y_test_lab == 2*ones(N_test,1)];
    Y_test(:,3) = [Y_test_lab == 3*ones(N_test,1)];
    
    