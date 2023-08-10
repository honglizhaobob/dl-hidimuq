function res = line_energy(b, i, j, x)
    % Helper function to compute the line energy given the states.
    % For the line corresponding to generator (i, j).
    assert(mod(length(x),4)==0);
    [v,~,delta,~] = split_vector(x);
    res = (b(i,j)^2)* ...
        ((v(i)^2)-2*(v(i)*v(j))*cos(delta(i)-delta(j))+(v(j)^2));
end