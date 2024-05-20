function finterp = empirical_cdf(xdata,xgrid)
    % computes empirical CDF then interpolate to 
    % regular grid, by default uses linear inter/extrapolation.
    [f,x] = ecdf(xdata);
    % first point of x is repeated
    x = x(2:end);
    f = f(2:end);
    finterp = interp1(x,f,xgrid,'nearest','extrap');
end