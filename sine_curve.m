function curve = sine_curve(w, x)
    sum = 0;
    sum = sum + w(2)*x;
    for i=1:11
        sum = sum + w(i+2)*sin(0.4*pi*x*i);
    end
    curve = sum;