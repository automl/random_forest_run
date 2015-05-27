function [pred, predvar] = fh_simple_one_treeval(Tree,X,strategyForMissing)
%TREEVAL Compute fitted value for decision tree applied to data.
%   YFIT = TREEVAL(TREE,X) takes a classification or regression tree TREE
%   as produced by the TREEFIT function, and a matrix X of predictor
%   values, and produces a vector pred of predicted response values.
%
%   NaN values in the X matrix are not allowed.

% if ~isstruct(Tree) || ~isfield(Tree,'method')
%    error('stats:treeval:BadTree',...
%          'The first argument must be a decision tree.');
% end
if nargin < 3
    strategyForMissing = 0;
end

if nargout>=3 && ~isequal(Tree.method,'regression')
   error('stats:treeval:TooManyOutputs',...
         'Only 2 output arguments available for regression trees.');
end
[nr,nc] = size(X);
if nc~=Tree.npred
   error('stats:treeval:BadInput',...
         'The X matrix must have %d columns.',Tree.npred);
end
 
%=== Call MEX file.
seed = ceil(rand*100000000);
%strategyForMissing = 0; % subtree average (Matlab regression tree default)
%strategyForMissing = 1; % 50-50 random subtree selection
%strategyForMissing = 4; % optimistic subtree selection

[pred, predvar] = fwd_big_var(X, int32(Tree.var), Tree.cut, Tree.class, Tree.nodeerr, int32(Tree.children), Tree.catsplit, int32(seed), int32(strategyForMissing)); 

%------------------------------------------------
function pred = fwd_one_row(X, var, cut, class, children, catsplit, thisnode)
%DOAPPLY Apply classification rule to x.
%   This is a recursive function for propagating a single row X.
%   Starts at top node, then recurses over
%   child nodes.  THISNODE is the current node at each step. All other
%   variables are the same at each node.

splitvar      = var(thisnode);
cutoff        = cut(thisnode);
kids          = children(thisnode,:);

% Terminal case
if splitvar==0
   pred = class(thisnode);
   return
end

%%%% Now deal with non-terminal nodes %%%%

% Determine if this point goes left, goes right, or stays here
x = X(abs(splitvar));

if splitvar>0                % continuous variable
    if x < cutoff
        pred = fwd_one_row(X, var, cut, class, children, catsplit, kids(1));
        return
    else
        pred = fwd_one_row(X, var, cut, class, children, catsplit, kids(2));
        return
    end        
else
    for i=1:length(catsplit{cutoff,1})
        if catsplit{cutoff,1}(i) == x
            pred = fwd_one_row(X, var, cut, class, children, catsplit, kids(1));
            return
        end
    end
    for i=1:length(catsplit{cutoff,2})
        if catsplit{cutoff,2}(i) == x
            pred = fwd_one_row(X, var, cut, class, children, catsplit, kids(2));
            return
        end
    end
    pred = class(thisnode);
    return
end