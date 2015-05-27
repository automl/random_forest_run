function [means, vars, tree_means] = compute_from_leaves(modules, Theta_idx, X, Theta_is_idx)
if nargin < 5
    Theta_is_idx=0;
end

M = length(modules);
if Theta_is_idx
    global ThetaUniqSoFar;
    numThetas = length(Theta_idx);
    Theta = ThetaUniqSoFar(Theta_idx,:); % currently all of them, but can use subset and then we have the indices.
else
    numThetas = size(Theta_idx,1);
    Theta = Theta_idx;
end

%=== For each tree, get the leaves.
cell_of_leaves = cell(1,M);
%leaf_count = 0;

for m=1:M
    %cell_of_leaves{m} = fh_treeval_thetas_pis_matlab(RF{m},Theta,X);
    cell_of_leaves{m} = fh_treeval_thetas_pis(modules{m}.T,Theta,X);
%    leaf_count = leaf_count + length(cell_of_leaves{m});
end

[means, vars, tree_means] = compute_from_leaves_part(cell_of_leaves, int32(numThetas), int32(size(X,1)));

% test of compute_from_leaves
% incsum=0; for i=1:5, incsum = incsum + length(cell_of_leaves{1}{i}{2}) *
% sum(cell_of_leaves{1}{i}{3})/length(cell_of_leaves{1}{i}{3}); incsum/2000, end

if any(isnan(vars))
    vars
    error('some variance is NaN.')
end
if ~all(vars>-1e-6)
    minvar = min(vars)
    vars
    warning('some variance is < 10^-6. Numerical problems can make variance a tiny bit negative, but not much - more would probably be due to a bug.')
end