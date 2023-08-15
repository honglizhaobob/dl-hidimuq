function E = energy(x, B)
    % computes energy of state at a specific time.
    try 
        % verifies that B is positive definite by Cholesky
        F = chol(B);
    catch
        warning("Cholesky failed, energy is not meaningful. ");
        E = +inf;
    end
    y = F*x;
    E = norm(y)^2;
end