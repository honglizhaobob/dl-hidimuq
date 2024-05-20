function c = get_coeff2d(x_data,y_data,z_data,xgrid,ygrid,mode)
    % Estimate coefficient using [x, y] => z data:
    % Exact solution is E[z | X=x, Y=y], and evalute 
    % result on meshgrid formed by (xgrid, ygrid) representing 
    % cell edges. `mode` indicates the parameteric or nonparameteric 
    % model used for regression.
    assert((numel(x_data)==numel(y_data))&(numel(y_data)==numel(z_data)));
    xx=x_data(:); yy=y_data(:); zz=z_data(:);
    % number of samples
    ns = numel(xx);
    % number of grid points in each dimension
    nx = length(xgrid); ny = length(ygrid);
    dx = xgrid(2)-xgrid(1); dy = ygrid(2)-ygrid(1);
    if mode == "lin"
        c = zeros(nx,ny);
        % linear regression with bias
        X=[ones(ns,1) xx yy];
        coeffs = (X'*X)\(X'*zz);
        % evaluate coefficients at each point in meshgrid
        for i = 1:nx
            for j = 1:ny
                c(i,j) = coeffs(2)*xgrid(i)+coeffs(3)*ygrid(j)+coeffs(1);
            end
        end
    elseif mode == "lowess"
        fitsurface = fit([xx yy],zz,'lowess',"Normalize","on");
        [ymesh,xmesh] = meshgrid(ygrid,xgrid);
        % evaluate coefficients at each point in meshgrid
        c = reshape(fitsurface(xmesh(:),ymesh(:)),[nx ny]);
    else
        error("not implemented! ");
    end
    if any(any(isnan(c)))
        error("NaN encountered. ");
    end
end



