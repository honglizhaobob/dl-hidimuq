function p1 = prod_measure(p,dx,dy)
    % Forms product measure of marginals from p(x,y) by numerically 
    % integrating over uniform grid of sizes dx dy.
    
    % marginal in x
    px = trapz(dy,p,2);
    px = px(:);
    % normalize
    px = abs(px)/trapz(dx,px);

    % marginal in y
    py = trapz(dx,p,1);
    py = py(:);
    py = abs(py)/trapz(dy,py);

    % form product measure
    p1 = px.*py';
    % renormalize
    p1 = p1/trapz(dx,trapz(dy,p1,2));
end