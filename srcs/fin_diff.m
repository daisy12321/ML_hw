function df = fin_diff(fun, x, h)
    dim = length(x);
    df = zeros(dim,1);
    for i = 1:dim
        xdelta = zeros(dim,1);
        xdelta(i) = h;
        df(i) = (fun(x+xdelta) - fun(x-xdelta))/(2*h);
    end