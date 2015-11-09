function accu = get_accu_ANN(W1, W2, X, Y)
    [predict_all, predict_class] = predict_multi_class(X, W1, W2);
    accu = sum(predict_class' == Y)/length(Y);
end