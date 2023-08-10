function res = condexp_target(b, i, j, x, wr)
    % The argument of conditional expectation for the RO-PDF equation
    % defined for the line energy. 
    % The exact expression is:
    %
    %   2(bij^2)*(
    %   ( v(i)-v(j)cos(delta(i)-delta(j)) ) * (w(i)-wr) + 
    %   ( v(j)-v(i)cos(delta(i)-delta(j)) ) * (w(j)-wr)
    % )
    assert(mod(length(x),4)==0);
    [v,w,delta,~] = split_vector(x);
    res = 2*(b(i,j)^2)*( (v(i)-v(j)*cos(delta(i)-delta(j)))*(w(i)-wr) + ...
        (v(j)-v(i)*cos(delta(i)-delta(j)))*(w(j)-wr) );
end