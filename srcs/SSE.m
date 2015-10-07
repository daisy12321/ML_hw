function sse = SSE(w)
    global X_full Y;
    yhat = w' * X_full';
    sse = (Y-yhat)*(Y-yhat)';
    