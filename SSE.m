function sse = SSE(w)
    global X_full Y M;
    yhat = w' * X_full';
    sse = (Y-yhat)*(Y-yhat)';
    