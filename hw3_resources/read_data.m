function [X_train, Y_train, X_valid, Y_valid, X_test, Y_test] = read_data(name)
    train = importdata(strcat('hw3_resources/data/',name,'_train.csv'));
    valid = importdata(strcat('hw3_resources/data/',name,'_validate.csv'));
    test = importdata(strcat('hw3_resources/data/',name,'_test.csv'));

    %%%% training %%%%
    X_train = train(:,1:2);
    Y_init = train(:,3);
    [N, D] = size(X_train);
    K = max(Y_init);
    
    Y_train = zeros(N,K);
    Y_train(:,1) = [Y_init == ones(N,1)];
    Y_train(:,2) = [Y_init == 2*ones(N,1)];
    Y_train(:,3) = [Y_init == 3*ones(N,1)];
    
    %%%% validation %%%%
    X_valid = valid(:,1:2);
    Y_init = valid(:,3);
    [N, D] = size(X_valid);
    K = max(Y_init);
    
    Y_valid = zeros(N,K);
    Y_valid(:,1) = [Y_init == ones(N,1)];
    Y_valid(:,2) = [Y_init == 2*ones(N,1)];
    Y_valid(:,3) = [Y_init == 3*ones(N,1)];
    
    
    
    %%%% testing %%%%
    X_test = test(:,1:2);
    Y_init = test(:,3);
    [N, D] = size(X_valid);
    K = max(Y_init);
    
    Y_test = zeros(N,K);
    Y_test(:,1) = [Y_init == ones(N,1)];
    Y_test(:,2) = [Y_init == 2*ones(N,1)];
    Y_test(:,3) = [Y_init == 3*ones(N,1)];
    
    