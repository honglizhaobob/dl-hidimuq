function res = line_energy2(b, i, j, v, delta)
    % Helper function to compute the line energy given the states.
    % For the line corresponding to generator (i, j).
    res = (b(i,j)^2)* ...
        ((v(i)^2)-2*(v(i)*v(j))*cos(delta(i)-delta(j))+(v(j)^2));
end