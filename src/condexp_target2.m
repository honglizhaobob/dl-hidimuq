function res = condexp_target2(b, i, j, v, w, delta)
    % The argument of conditional expectation for the RO-PDF equation
    % defined for the line energy. 
    % The exact expression is:
    %
    %   2(bij^2)*(
    %   ( v(i)-v(j)cos(delta(i)-delta(j)) ) * (w(i)-wr) + 
    %   ( v(j)-v(i)cos(delta(i)-delta(j)) ) * (w(j)-wr)
    % )
    res = 2*(b(i,j)^2)*v(i)*v(j)*sin(delta(i)-delta(j))*(w(i)-w(j));
end