function p = normalize_pdf(pold, dx)
    % Normalizes the PDF to integrate to 1 given uniform step size (1d)
    pold = abs(pold);
    p = pold/trapz(dx,pold);
end