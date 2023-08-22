function R = cov_model(case_number, mode, B, X)
    % Covariance models for buses, defined in: https://arxiv.org/abs/2207.13310
    % 
    % case_number                       problem case
    %
    % mode                              - id, uncorrelated / identity
    %                                   - exp, exponentially correlated
    %                                   - const, constant matrix
    %
    all_modes = ["id", "exp", "const"];
    assert(ismember(mode, all_modes));
    if case_number == 9
        n = 9;
        if mode == "id"
            R = eye(n);
        elseif mode == "const"
            R = 0.44.*ones(n,n);
            % reassign diagonal elements
            for i=1:n
                R(i,i)=1.0;
            end
        else
            % needs to use reactance and susceptance matrices
            distancemat = sqrt(B.*X);
            lambda = 82.0;      % constant scaling factor
            R = exp(-distancemat./lambda);
        end
    elseif case_number == 30
        n = 30;
        if mode == "id"
            R = eye(n);
        elseif mode == "const"
            R = 0.36.*ones(n,n);
            % reassign diagonal elements
            for i=1:n
                R(i,i)=1.0;
            end
        else
            error("not implemented. ");
        end
    elseif case_number == 57
        n = 57;
        if mode == "id"
            R = eye(n);
        elseif mode == "const"
            R = 0.36.*ones(n,n);
            % reassign diagonal elements
            for i=1:n
                R(i,i)=1.0;
            end
        else
            error("not implemented. ");
        end
    else
        error("Requires implementation. ")
    end
end