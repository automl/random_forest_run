function [cell_of_leaves, objPredNoTrafo, firstObjPredNoTrafo, secondObjPredNoTrafo] = fh_treeval_thetas_pis(Tree,Theta,X,strategyForMissing, trans, kappa)
%   FH_TREEVAL_THETAS_PIS 
%   Propagate the parameter configruations in Theta's rows and the 
%   instances with the features in X's row down the tree.
%   Return a cell array with entries [theta_idx, pi_idx, value, nodenumber]
%   for each leaf that is compatible with a subset of both Theta and X.
%
%   NaN values in Theta or X are forbidden.
if nargin < 6
    if nargout > 1
        error 'Need to specify kappa in call to fh_treeval_thetas_pis';
    else
        kappa = 5; % result not used anyways.
    end
    if nargin < 5
        trans = 'log';
    end
end
if strcmp(trans, 'id')
    trafoInt = int32(0);
else
    trafoInt = int32(1);
end

% if ~isstruct(Tree) | ~isfield(Tree,'method')
%    error('stats:treeval:BadTree',...
%          'The first argument must be a decision tree.');
% end
[N,nt] = size(Theta);
[n,nx] = size(X);

if nt+nx ~= Tree.npred
   error('stats:treeval:BadInput',...
         'The X and Theta matrices must have %d columns together.',Tree.npred);
end

%[cell_of_leaves, numCellsFilled] = collect_leaves_theta_pis(Tree.var,Tree.cut,Tree.class,Tree.children(:,1),Tree.children(:,2),Tree.catsplit,Theta,X);
%seed = ceil(rand*100000000);
seed = 1234;
%strategyForMissing = 0; % subtree average (Matlab regression tree default)
%strategyForMissing = 1; % 50-50 random subtree selection
%strategyForMissing = 4; % optimistic subtree selection
% if strategyForMissing == -1
%     strategyForMissing=0; % collecting all three values into the cell array
% end
%%[cell_of_leaves, numCellsFilled, objPredNoTrafo, firstObjPredNoTrafo, secondObjPredNoTrafo] = collect_leaves_theta_pis_sample_missing(Tree.var,Tree.cut,Tree.class,Tree.children(:,1),Tree.children(:,2),Tree.minval,Tree.maxval,Tree.catsplit,Theta,X,int32(seed),int32(strategyForMissing),trafoInt, kappa);

% %[cell_of_leaves, numCellsFilled, objPredNoTrafo] = collect_marginal_mean_var_flat_theta_pis(Tree.var,Tree.cut,Tree.ysub,Tree.children(:,1),Tree.children(:,2),Tree.catsplit,Theta,X,int32(seed),Tree.nodesize,trafoInt, kappa);
% %[cell_of_leaves, numCellsFilled, objPredNoTrafo] = collect_marginal_mean_var_theta_pis(Tree.var,Tree.cut,Tree.class,Tree.nodeerr,Tree.children(:,1),Tree.children(:,2),Tree.catsplit,Theta,X,int32(seed),Tree.nodesize,trafoInt, kappa);
% %[objPredNoTrafo] = collect_marginal_mean_var_theta_pis_simple(Tree.var,Tree.cut,Tree.class,Tree.nodeerr,Tree.children(:,1),Tree.children(:,2),Tree.catsplit,Theta,X,int32(seed),Tree.nodesize,trafoInt, kappa);

% [cell_of_leaves, numCellsFilled, objPredNoTrafo] = collect_big_leaves_weight_theta_pis_sample_missing(Tree.var,Tree.cut,Tree.ysub,Tree.children(:,1),Tree.children(:,2),Tree.catsplit,Tree.nodesize,Theta,X,int32(seed),trafoInt);
% 
% [cell_of_leaves, numCellsFilled, objPredNoTrafo, firstObjPredNoTrafo, secondObjPredNoTrafo] = collect_big_leaves_theta_pis_sample_missing(Tree.var,Tree.cut,Tree.class,Tree.nodeerr,Tree.children(:,1),Tree.children(:,2),Tree.catsplit,Theta,X,int32(seed),int32(strategyForMissing),trafoInt, kappa);
%[cell_of_leaves, numCellsFilled, objPredNoTrafo] = new_collect_big_leaves_theta_pis_sample_missing(Tree.var,Tree.cut,Tree.ysub,Tree.children(:,1),Tree.children(:,2),Tree.catsplit,Tree.leftsample,Theta,X,int32(seed));

%[cell_of_leaves, numCellsFilled] = collect_big_leaves_theta_pis_nomissing(Tree.var,Tree.cut,Tree.ysub,Tree.children(:,1),Tree.children(:,2),Tree.catsplit,Theta,X);
%[cell_of_leaves, numCellsFilled] = collect_big_leaves_theta_pis_distinleaf_nomissing(Tree.var,Tree.cut,Tree.ysub,Tree.children(:,1),Tree.children(:,2),Tree.catsplit, Theta, X ,Tree.m_n, Tree.kappa_n, Tree.alpha_n, Tree.beta_n, Tree.mu_n, Tree.var_n, Tree.lambdas_n, Tree.mus_n);
% if isfield(Tree, 'leaf_median')
%     % only to get the ids after all.
%     [cell_of_leaves, numCellsFilled] = collect_big_leaves_theta_pis_distinleaf_nomissing(Tree.var,Tree.cut,Tree.ysub,Tree.children(:,1),Tree.children(:,2),Tree.catsplit, Theta, X ,Tree.leaf_g, Tree.leaf_m, Tree.leaf_n, Tree.leaf_median, Tree.leaf_median);
% else
    if isfield(Tree, 'leaf_mean')
        [cell_of_leaves, numCellsFilled] = collect_big_leaves_theta_pis_distinleaf_nomissing(Tree.var,Tree.cut,Tree.ysub,Tree.children(:,1),Tree.children(:,2),Tree.catsplit, Theta, X ,Tree.leaf_g, Tree.leaf_m, Tree.leaf_n, Tree.leaf_mean, Tree.leaf_var);
    else
    % only to get the ids after all.
        [cell_of_leaves, numCellsFilled] = collect_big_leaves_theta_pis_distinleaf_nomissing(Tree.var,Tree.cut,Tree.ysub,Tree.children(:,1),Tree.children(:,2),Tree.catsplit, Theta, X ,Tree.leaf_g, Tree.leaf_m, Tree.leaf_n, -ones(size(Tree.leaf_n)), -ones(size(Tree.leaf_n)));
    end
% end


%[cell_of_leaves, numCellsFilled, objPredNoTrafo, firstObjPredNoTrafo,
%secondObjPredNoTrafo] =
%collect_weighted_samples_from_leaves_theta_pis(Tree.var,Tree.cut,Tree.clas
%s,Tree.nodeerr,Tree.children(:,1),Tree.children(:,2),Tree.catsplit,Tree.ysub,Theta,X,int32(seed),Tree.nodesize,int32(k_samples),trafoInt, kappa);
%k_samples = 1;
%[cell_of_leaves, numCellsFilled, objPredNoTrafo] = collect_samples_from_leaves_theta_pis(Tree.var,Tree.cut,Tree.class,Tree.nodeerr,Tree.ysub,Tree.children(:,1),Tree.children(:,2),Tree.catsplit,Theta,X,int32(seed),Tree.nodesize,int32(k_samples),trafoInt, kappa);
%k_samples
%[cell_of_leaves_m, numCellsFilled_m, objPredNoTrafo_m] = collect_leaves_theta_pis_sample_missingKM(Tree.var,Tree.cut,Tree.class,int32(Tree.is_censored),Tree.children(:,1),Tree.children(:,2),Tree.minval,Tree.maxval,Tree.catsplit,Theta,X,int32(seed),int32(strategyForMissing),trafoInt, kappa);
%assertVectorEq(objPredNoTrafo, objPredNoTrafo_m);
% assert(length(cell_of_leaves)==length(cell_of_leaves_m));
% assert(numCellsFilled==numCellsFilled_m);
% 
% for i=1:numCellsFilled
%     assertVectorEq(cell_of_leaves{i}{1}, cell_of_leaves_m{i}{1});
%     assertVectorEq(cell_of_leaves{i}{2}, cell_of_leaves_m{i}{2});
%     assertVectorEq(cell_of_leaves{i}{3}(1), cell_of_leaves_m{i}{3}(1));
%     assertVectorEq(cell_of_leaves{i}{3}(2), cell_of_leaves_m{i}{3}(2));
%     assertVectorEq(cell_of_leaves{i}{3}(3), cell_of_leaves_m{i}{3}(3));
%     assertVectorEq(cell_of_leaves{i}{4}, cell_of_leaves_m{i}{4});
% end

% % Super slow! [cell_of_leaves_m, numCellsFilled_m] = collect_leaves_theta_pis_m(Tree.var,Tree.cut,Tree.class,Tree.children(:,1),Tree.children(:,2),Tree.catsplit,Theta,X);
% cell_of_leaves_m = fh_treeval_thetas_pis_matlab(Tree,Theta,X);

cell_of_leaves = cell_of_leaves(1:numCellsFilled);

% Function I used for MEXing.
function [cell_of_leaves, numCellsFilled] = collect_leaves_theta_pis_m(var,cut,class,leftchildren,rightchildren,catsplit,Theta,X)
%=== Create result cell array once and pass it to be filled.
cell_of_leaves = cell(1,length(var)); % upper bound on # leaves, allocate space in the beginning.

N = size(Theta,1);
n = size(X,1);
[cell_of_leaves, numCellsFilled] = fwd_theta_pis(var,cut,class,leftchildren,rightchildren,catsplit,Theta,X,1:N,N,1:n,n,1,0,cell_of_leaves);


%------------------------------------------------
function [cell_of_leaves, numCellsFilled] = fwd_theta_pis(var,cut,class,leftchildren,rightchildren,catsplit,Theta,X,Theta_rows,N,x_rows,n,thisnode,numCellsFilled,cell_of_leaves)
%DOAPPLY Apply classification rule to specified rows starting at a node.
%   This is a recursive function.  Starts at top node, then recurses over
%   child nodes.  THISNODE is the current node at each step.
%   Theta_rows are the idxs of Theta left at this node, x_rows the idxs of X.
%   cell_of_leaves is passed from above, and we simply fill the leaves from
%   this subtree into it starting at index numCellsFilled+1.

splitvar      = var(thisnode);
cutoff        = cut(thisnode);
leftchild     = leftchildren(thisnode,:);
rightchild    = rightchildren(thisnode,:);

% Declare variables.
idx_goingleft = zeros(max(N,n),1);
idx_goingright = zeros(max(N,n),1);
idx_missing = zeros(max(N,n),1);
 
% Terminal case
if splitvar==0
   id = class(thisnode);
   numCellsFilled = numCellsFilled+1;
   cell_of_leaves{numCellsFilled} = {Theta_rows, x_rows, id, thisnode};
   return;
end

%%%% Now deal with non-terminal nodes %%%%

% Determine whether splitting on a parameter or an instance feature.
if abs(splitvar) <= size(Theta,2)
    split_on_param = 1;
    xLen = N;
    x = Theta(Theta_rows,abs(splitvar));
else 
    split_on_param = 0;
    xLen = n;
    x = X(x_rows,abs(splitvar)-size(Theta,2));
end

numLeft = 0;
numRight = 0;
numMissing = 0;
% Determine if this point goes left, goes right, or stays here
if splitvar>0                % continuous variable
    % Matlab: isleft = (x < cutoff);
    % Matlab: isright = ~isleft;
    for i=1:xLen
        if x(i) < cutoff
%            fprintf(strcat(['Continuous ', num2str(splitvar), ' left: ', num2str(x(i)), ' < ', num2str(cutoff), '\n']));
            numLeft = numLeft+1;
            idx_goingleft(numLeft) = i;
        else
%            fprintf(strcat(['Continuous ', num2str(splitvar), ' right: ', num2str(x(i)), ' >= ', num2str(cutoff), '\n']));
            numRight = numRight+1;
            idx_goingright(numRight) = i;
        end
	end
else                         % categorical variable
    % Matlab: isleft = ismember(x,catsplit{cutoff,1});
    % Matlab: isright = ismember(x,catsplit{cutoff,2});
    % Matlab: ismissing = ~(isleft | isright);
    nCatLeft = length(catsplit{cutoff,1});
    nCatRight = length(catsplit{cutoff,2});
    for i=1:xLen
        goleft = 0;
        for j=1:nCatLeft
%            fprintf(strcat(['categorical  ', num2str(-splitvar), ' left; ', num2str(x(i)), ' == ', num2str(catsplit{cutoff,1}(j)), '?\n']));
            if catsplit{cutoff,1}(j) == int32(x(i))
                goleft = 1;
                break;
            end
        end
        if goleft>0
            numLeft = numLeft+1;
            idx_goingleft(numLeft) = i;
        else % if we don't go left, maybe we go right ...
            goright = 0;
            for j=1:nCatRight
%                fprintf(strcat(['categorical  ', num2str(-splitvar), ' right; ', num2str(x(i)), ' == ', num2str(catsplit{cutoff,2}(j)), '?\n']));
                if catsplit{cutoff,2}(j) == int32(x(i))
                    goright = 1;
                    break;
                end
            end
            if goright>0
                numRight = numRight+1;
                idx_goingright(numRight) = i;
            else % we go neither left nor right => missing
                numMissing = numMissing + 1;
                idx_missing(numMissing) = i;
            end
        end
    end
end

%thisnode
%idx_goingleft_out = idx_goingleft(1:numLeft)
%idx_goingright_out = idx_goingright(1:numRight)
%idx_missing_out = idx_missing(1:numMissing)

if numLeft>0  % going left 
    if split_on_param
        [cell_of_leaves, numCellsFilled] = fwd_theta_pis(var,cut,class,leftchildren,rightchildren,catsplit,Theta,X,Theta_rows(idx_goingleft(1:numLeft)),numLeft,x_rows,n,leftchild,numCellsFilled,cell_of_leaves);
    else
        [cell_of_leaves, numCellsFilled] = fwd_theta_pis(var,cut,class,leftchildren,rightchildren,catsplit,Theta,X,Theta_rows,N,x_rows(idx_goingleft(1:numLeft)),numLeft,leftchild,numCellsFilled,cell_of_leaves);
    end
end

if numRight>0  % going right 
    if split_on_param
        [cell_of_leaves, numCellsFilled] = fwd_theta_pis(var,cut,class,leftchildren,rightchildren,catsplit,Theta,X,Theta_rows(idx_goingright(1:numRight)),numRight,x_rows,n,rightchild,numCellsFilled,cell_of_leaves);
    else
        [cell_of_leaves, numCellsFilled] = fwd_theta_pis(var,cut,class,leftchildren,rightchildren,catsplit,Theta,X,Theta_rows,N,x_rows(idx_goingright(1:numRight)),numRight,rightchild,numCellsFilled,cell_of_leaves);
    end
end

if numMissing>0  % staying here
    if split_on_param
        numCellsFilled = numCellsFilled+1;
        cell_of_leaves{numCellsFilled} = {Theta_rows(idx_missing(1:numMissing)), x_rows, class(thisnode), thisnode};
    else
        error 'No empty features (continuous) allowed - a value is either bigger or smaller/equal another one.'
    end
end