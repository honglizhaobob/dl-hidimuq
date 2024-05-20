function I = mutual_info(p_xy, dx, dy)
    % Given joint density, p_xy, computes mutual information
    % between x, y by numerically integrating with stepsizes 
    % dx, dy. p_xy(i,j) = p(x(i),y(j))

    % compute marginal in x
    p_x = trapz(dy,p_xy,2);
    % renormalize
    p_x = abs(p_x)/trapz(dx,p_x);       % column vector N_x

    % compute marginal in y
    p_y = trapz(dx,p_xy,1);
    % renormalize
    p_y = abs(p_y)/trapz(dy,p_y);       % row vector N_y

    % product measure
    p_xp_y = p_x.*p_y;

    % compute mutual info, set zero density to eps and renormalize
    p_xp_y(p_xp_y<eps) = eps;
    p_xp_y = abs(p_xp_y)/trapz(dy,trapz(dx,p_xp_y,1));
    p_xy(p_xy<eps) = eps;
    p_xy = abs(p_xy)/trapz(dy,trapz(dx,p_xy,1));
    I = trapz(dy,trapz(dx,p_xy.*log(p_xy./p_xp_y),1));

end