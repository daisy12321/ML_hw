function sse = SSE(X, Y, M, w)
    w0 = w(1);
    w_other = w(2:length(w));
    yhat = zeros(1,length(X)) + w0;
    for i = 1:M
        yhat = yhat + w_other(i) * (X .^ i);
    end
    
    sse = (Y-yhat)*(Y-yhat)'
    