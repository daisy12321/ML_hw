function accu = get_accu(w, X, Y)
    predictLR = @(x) sigmoid(w(2:length(w)) * x + w(1));

    Y_pred(predictLR(X')>.5) = 1;
    Y_pred(predictLR(X')<=.5) = -1;
    
    accu = sum(Y_pred == Y')/length(Y);
end
