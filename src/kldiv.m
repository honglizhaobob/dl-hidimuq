function div = kldiv(p1,p2,dx,dy)
    % Computes D_kl(p1|p2) given uniform grid sizes dx, dy
    % WARNING: nonsymmetric!
    
    % pad with eps
    p1(p1<eps) = eps;
    p2(p2<eps) = eps;
    % ensure density
    p1 = abs(p1)/trapz(dy,trapz(dx,p1));
    p2 = abs(p2)/trapz(dy,trapz(dx,p2));
    % compute KL 
    div = trapz(dy,trapz(dx,p1.*(log(p1)-log(p2)),1));
end