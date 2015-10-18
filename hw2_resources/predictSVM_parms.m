function y_svm = predictSVM_parms(x, kernel, w, w_0, sigma2, X, Y, alpha)
    n = length(Y);
    if kernel=='dot'
        y_svm = w * x + w_0;
    else
        y_svm = w_0;
        for i=1:n
            y_svm = y_svm + alpha(i)*Y(i)*rbf(X(i,:),x',sigma2);
        end
    end
end