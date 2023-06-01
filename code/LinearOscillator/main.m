% (05/31/2023) Main script for running simulations of the random
% linear oscillator problem.
clear; clc; rng("default");

% Set up dimensions
d = 200;
tstart = 0.0;
tend = 10.0;
dt = 1e-2;
tgrid = tstart:dt:tend;
