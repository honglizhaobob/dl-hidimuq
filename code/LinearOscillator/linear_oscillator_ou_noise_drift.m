function mu = linear_oscillator_ou_noise_drift(t, u, params)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Hongli Zhao, honglizhaobob@uchicago.edu
%          
% Drift term for the linear oscillator driven by OU noise.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
n = length(u);
assert(mod(n,3)==0);
d = n/3;

mu = zeros(n,1);
% unpack states, which includes OU process
XtYt = u(1:2*d); Thetat = u((2*d+1):end);
% deterministic part
mu1 = oscillator_rhs(t, XtYt, params.M, params.D, params.K, params.f);
mu(1:2*d) = mu1;

% add OU noise influence
mu((d+1):2*d) = mu((d+1):2*d) + Thetat;

% OU noise process
mu((2*d+1):end) = -params.delta .* Thetat;

end