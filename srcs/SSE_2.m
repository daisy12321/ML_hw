function sse = SSE_2(X_full, Y, w)
    yhat = w' * X_full';
    sse = (Y-yhat)*(Y-yhat)';