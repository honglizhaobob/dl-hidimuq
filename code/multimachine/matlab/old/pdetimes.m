%% Case 30
clear

load('errors_case30_con.mat', 'ICMCclocks','PDEclocks')
mc_vec = (500:500:(500*length(ICMCclocks)))';
pde_time_case30_con = sum(PDEclocks([2,5,13:16,19,23,26,27,29,31:52,54,55,57,58])-ICMCclocks(1)) + sum(PDEclocks([1,7,11,12,24,25,30])-ICMCclocks(2)) + sum(PDEclocks([18,20,28])-ICMCclocks(3)) + sum(PDEclocks([3,4,6,8:10,17,21,22,53,56,59,60])-ICMCclocks(7))+ ICMCclocks(7);
ll = sort([2,5,13:16,19,23,26,27,29,31:52,54,55,57,58,1,7,11,12,24,25,30,18,20,28,3,4,6,8:10,17,21,22,53,56,59,60]);
if ~isequal(ll,1:60)
    error('indices not matched')
end
pde_samples_case30_con = length([2,5,13:16,19,23,26,27,29,31:52,54,55,57,58])*mc_vec(1) + length([1,7,11,12,24,25,30])*mc_vec(2) + 3*mc_vec(3) + length([3,4,6,8:10,17,21,22,53,56,59,60])*mc_vec(7);
speed_case30_con = length([2,5,13:16,19,23,26,27,29])*mc_vec(1) + length([1,7,11,12,24,25,30])*mc_vec(2) + 3*mc_vec(3) + length([3,4,6,8:10,17,21,22])*mc_vec(7)
phase_case30_con = length([31:52,54,55,57,58])*mc_vec(1) + length([53,56,59,60])*mc_vec(7)


load('errors_case30_exp.mat', 'ICMCclocks','PDEclocks')
mc_vec = (500:500:(500*length(ICMCclocks)))';
pde_time_case30_exp = sum(PDEclocks([1,5,11,18,24:27,31:41,45:52,54,57,58])-ICMCclocks(1)) + sum(PDEclocks([7,12,13,15,16,29,30,53])-ICMCclocks(2)) + sum(PDEclocks([2,9,19,20,23])-ICMCclocks(3)) + sum(PDEclocks([14,17,42])-ICMCclocks(4))+ sum(PDEclocks([3,4,6,8,10,21,22,28,43,44,55,56,59,60])-ICMCclocks(6))+ ICMCclocks(6);
ll = sort([1,5,11,18,24:27,31:41,45:52,54,57,58,7,12,13,15,16,29,30,53,2,9,19,20,23,14,17,42,3,4,6,8,10,21,22,28,43,44,55,56,59,60]);
if ~isequal(ll,1:60)
    error('indices not matched')
end
pde_samples_case30_exp = length([1,5,11,18,24:27,31:41,45:52,54,57,58])*mc_vec(1) + length([7,12,13,15,16,29,30,53])*mc_vec(2) + 5*mc_vec(3) + 3*mc_vec(4) + length([3,4,6,8,10,21,22,28,43,44,55,56,59,60])*mc_vec(6);
speed_case30_exp = length([1,5,11,18,24:27])*mc_vec(1) + length([7,12,13,15,16,29,30])*mc_vec(2) + 5*mc_vec(3) + 2*mc_vec(4) + length([3,4,6,8,10,21,22,28])*mc_vec(6)
phase_samples_case30_exp = length([31:41,45:52,54,57,58])*mc_vec(1) + mc_vec(2) + mc_vec(4) + length([43,44,55,56,59,60])*mc_vec(6)

load('errors_case30_uncorr.mat', 'ICMCclocks','PDEclocks')
mc_vec = (500:500:(500*length(ICMCclocks)))';
pde_time_case30_uncorr = sum(PDEclocks([5,11,13,23,27,29:54,57,58])-ICMCclocks(1)) + sum(PDEclocks([1,14,18,24:26,55,59])-ICMCclocks(2)) + sum(PDEclocks([2:4,6:10,12,16,17,19:22,28,56,60])-ICMCclocks(6)) + PDEclocks(15)-ICMCclocks(5)+ ICMCclocks(6);
ll = sort([5,11,13,23,27,29:54,57,58,1,14,18,24:26,55,59,2:4,6:10,12,16,17,19:22,28,56,60,15]);
if ~isequal(ll,1:60)
    error('indices not matched')
end
pde_samples_case30_uncorr = length([5,11,13,23,27,29:54,57,58])*mc_vec(1) + length([1,14,18,24:26,55,59])*mc_vec(2) + length([2:4,6:10,12,16,17,19:22,28,56,60])*mc_vec(6) + mc_vec(5);
speed_case30_uncorr = length([5,11,13,23,27,29,30])*mc_vec(1) + length([1,14,18,24:26])*mc_vec(2) + length([2:4,6:10,12,16,17,19:22,28])*mc_vec(6) + mc_vec(5)
phase_case30_uncorr = length([31:54,57,58])*mc_vec(1) + length([55,59])*mc_vec(2) + length([56,60])*mc_vec(6) 

%% Case 9
clear

load('errors_case9_con.mat', 'ICMCclocks','PDEclocks')
mc_vec = (500:500:(500*length(ICMCclocks)))';
pde_time_case9_con = sum(PDEclocks([1,2,7,8,10:18])-ICMCclocks(1)) + sum(PDEclocks([5,6,9])-ICMCclocks(2)) + sum(PDEclocks([3,4])-ICMCclocks(4)) + ICMCclocks(4);
ll = sort([1,2,7,8,10:18,5,6,9,3,4]);
if ~isequal(ll,1:18)
    error('indices not matched')
end
pde_samples_case9_con = length([1,2,7,8,10:18])*mc_vec(1) + 3*mc_vec(2) + 2*mc_vec(4);
speed_case9_con = length([1,2,7,8])*mc_vec(1) + 3*mc_vec(2) + 2*mc_vec(4)
phase_case9_con = length(10:18)*mc_vec(1)


load('errors_case9_exp.mat', 'ICMCclocks', 'PDEclocks')
mc_vec = (500:500:(500*length(ICMCclocks)))';
pde_time_case9_exp = sum(PDEclocks([1,2,4,7,8,10:18])-ICMCclocks(1)) + sum(PDEclocks([3,6,9])-ICMCclocks(2)) + sum(PDEclocks(5)-ICMCclocks(3)) + ICMCclocks(3);
ll = sort([1,2,4,7,8,10:18,3,6,9,5]);
if ~isequal(ll,1:18)
    error('indices not matched')
end
pde_samples_case9_exp = length([1,2,4,7,8,10:18])*mc_vec(1) + 3*mc_vec(2) + mc_vec(3);
speed_case9_exp = length([1,2,4,7,8])*mc_vec(1) + 3*mc_vec(2) + mc_vec(3)
phase_samples_case9_exp = length(10:18)*mc_vec(1)

load('errors_case9_uncorr.mat', 'ICMCclocks', 'PDEclocks')
mc_vec = (500:500:(500*length(ICMCclocks)))';
pde_time_case9_uncorr = sum(PDEclocks([1,2,4:9])-ICMCclocks(7)) + sum(PDEclocks(10:18)-ICMCclocks(1)) + sum(PDEclocks(3)-ICMCclocks(2)) + ICMCclocks(7);
ll = sort([1,2,4:9,10:18,3]);
if ~isequal(ll,1:18)
    error('indices not matched')
end
pde_samples_case9_uncorr = length(10:18)*mc_vec(1) + mc_vec(2) + length([1,2,4:9])*mc_vec(7);
speed_case9_uncorr = mc_vec(2) + length([1,2,4:9])*mc_vec(7)
phase_case9_uncorr = length(10:18)*mc_vec(1) 

%% Case 9 8-9 fail
clear

load('errors_case9_con_89fail.mat', 'ICMCclocks','PDEclocks')
mc_vec = (500:500:(500*length(ICMCclocks)))';
pde_time_case9_con_89fail = sum(PDEclocks([2,7,8,10:18])-ICMCclocks(1)) + sum(PDEclocks([1,3,4])-ICMCclocks(2)) + PDEclocks(5)-ICMCclocks(6)+ sum(PDEclocks([6,9])-ICMCclocks(8)) + ICMCclocks(8);
ll = sort([2,7,8,10:18,1,3,4,5,6,9]);
if ~isequal(ll,1:18)
    error('indices not matched')
end
pde_samples_case9_con_89fail = length([2,7,8,10:18])*mc_vec(1) + 3*mc_vec(2) + mc_vec(6) + 2*mc_vec(8);
speeds_case9_con_89fail = length([2,7,8])*mc_vec(1) + 3*mc_vec(2) + mc_vec(6) + 2*mc_vec(8)
phase_case9_con_89fail = length(10:18)*mc_vec(1) 


load('errors_case9_exp_89fail.mat', 'ICMCclocks', 'PDEclocks')
mc_vec = (500:500:(500*length(ICMCclocks)))';
pde_time_case9_exp_89fail = sum(PDEclocks([2,3,6,7,10:18])-ICMCclocks(1)) + sum(PDEclocks([1,4,8])-ICMCclocks(2)) + sum(PDEclocks([5,9])-ICMCclocks(5)) + ICMCclocks(5);
ll = sort([2,3,6,7,10:18,1,4,8,5,9]);
if ~isequal(ll,1:18)
    error('indices not matched')
end
pde_samples_case9_exp_89fail = length([2,3,6,7,10:18])*mc_vec(1) + 3*mc_vec(2) + 2*mc_vec(5);
speed_case9_exp_89fail = length([2,3,6,7])*mc_vec(1) + 3*mc_vec(2) + 2*mc_vec(5)
phase_case9_exp_89fail = length(10:18)*mc_vec(1)


load('errors_case9_uncorr_89fail.mat', 'ICMCclocks', 'PDEclocks')
mc_vec = (500:500:(500*length(ICMCclocks)))';
pde_time_case9_uncorr_89fail = sum(PDEclocks([1,4,5,8,10:18])-ICMCclocks(1)) + sum(PDEclocks([2,3,7])-ICMCclocks(3)) + PDEclocks(6)-ICMCclocks(9) + PDEclocks(9)-ICMCclocks(11) + ICMCclocks(11);
ll = sort([1,4,5,8,10:18,2,3,7,6,9]);
if ~isequal(ll,1:18)
    error('indices not matched')
end
pde_samples_case9_uncorr_89fail = length([1,4,5,8,10:18])*mc_vec(1) + 3*mc_vec(3) + mc_vec(9) + mc_vec(11);
speed_case9_uncorr_89fail = length([1,4,5,8])*mc_vec(1) + 3*mc_vec(3) + mc_vec(9) + mc_vec(11)
phase_case9_uncorr_89fail = length(10:18)*mc_vec(1) 

