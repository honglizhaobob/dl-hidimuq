function E = energy(x, B)
    % computes energy of state at a specific time.
    try 
        % verifies that B is positive definite by Cholesky
        F = chol(B);
        y = F*x;
        E = norm(y)^2;
    catch
        warning("Cholesky failed, energy is not meaningful. ");
        E = +inf;
    end
end