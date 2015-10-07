function curve = sine_curve(w, x)
    sum = 0;
    sum = sum + w(1)*x;
    for i=1:11
        sum = sum + w(i+1)*sin(0.4*pi*x*i);
    end
    curve = sum;