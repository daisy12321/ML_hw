function sse_derivative = SSE_derivative(w)
    global X_full Y;
    yhat = w' * X_full';
    sse_derivative = -2*(X_full' * (Y-yhat)');