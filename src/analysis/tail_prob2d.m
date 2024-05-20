function [p_tail,F_x,F_y,F_xy] = tail_prob2d(p_xy, dx, dy)
    % Computes from gridded PDF p_xy the tail probability:
    % P(U1>u1, U2>u2) using inclusion-exclusion.
    %
    % returned as a grid for all combinations of u1, u2.

    % ensure joint density
    mass2d = trapz(dx,trapz(dy,p_xy,2));
    p_xy = abs(p_xy)/mass2d;
    % compute joint CDF
    F_xy = cumtrapz(dx,cumtrapz(dy,p_xy,2),1);

    % compute marginal density in x
    p_x = trapz(dy,p_xy,2);
    p_x = p_x(:);
    % normalize
    p_x = abs(p_x)/trapz(dx,p_x);
    % marginal CDF in x
    F_x = cumtrapz(dx,p_x);

    % compute marginal density in y
    p_y = trapz(dx,p_xy,1);
    p_y = p_y(:);
    % normalize
    p_y = abs(p_y)/trapz(dy,p_y);
    % marginal CDF in y
    F_y = cumtrapz(dy,p_y);

    % compute complement density (size Nx by Ny)
    complement = F_x + F_y' - F_xy;
    p_tail = abs(1.0-complement);

end