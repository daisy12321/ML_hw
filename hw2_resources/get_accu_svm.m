function accu = get_accu_svm(w, w_0, X, Y)
    predictSVM = @(x) w * x + w_0;

    Y_pred(predictSVM(X')>0) = 1;
    Y_pred(predictSVM(X')<=0) = -1;
    
    accu = sum(Y_pred == Y')/length(Y);
end
