function Tree = maketree_ysub_dist(X, catcols, nodenumber, parent, ysub, censsub, cutvar, cutpoint, leftchildren, rightchildren, resuberr, nodesize, catsplit, numNodes, ncatsplit, overallobj)
    Tree.node      = nodenumber(1:numNodes);
    Tree.parent    = parent(1:numNodes);
    Tree.ysub      = ysub(1:numNodes);
    Tree.is_censored=censsub(1:numNodes);
    Tree.var       = cutvar(1:numNodes);
    Tree.cut       = cutpoint(1:numNodes);
    Tree.children  = [leftchildren(1:numNodes), rightchildren(1:numNodes)];
    Tree.nodeerr   = resuberr(1:numNodes);
    Tree.nodesize  = nodesize(1:numNodes);
    Tree.npred     = size(X, 2);

    Tree.catcols   = catcols;
    Tree.method    = 'regression';
    Tree.catsplit  = catsplit(1:ncatsplit,:); % list of all categorical predictor splits

    for i=1:numNodes
        %=== Not using a distribution at all for mean and variance in leaf.
        if isempty(ysub{i})
            Tree.leaf_mean(i,1) = 0;
        else
            Tree.leaf_mean(i,1) = mean(ysub{i});
        end
        
        % TODO: remove leaf_n/g/m
        if (Tree.children(i) == 0) 
            Tree.leaf_n(i,1) = Tree.nodesize(i);
        else
            Tree.leaf_n(i,1) = 0;
        end
        Tree.leaf_g(i,1) = 0; % not used
        Tree.leaf_m(i,1) = Tree.leaf_mean(i,1);
        Tree.leaf_n = int32(Tree.leaf_n);
        
        Tree.leaf_var(i,1) = 0;
    end
    Tree.class = zeros(numNodes, 1);
    if strcmp(overallobj, 'median')
        Tree.leaf_median = zeros(numNodes,1);
        for i=1:numNodes 
            Tree.leaf_median(i) = median(ysub{i});
            Tree.class(i) = Tree.leaf_median(i);
        end
    else
        Tree.emp_mean_at_leaf = zeros(numNodes,1);
        for i=1:numNodes
            Tree.emp_mean_at_leaf(i) = Tree.leaf_mean(i,1); % mean(ysub{i});
            Tree.class(i) = Tree.emp_mean_at_leaf(i);
        end
    end

    % not implemented - just here for compatibility!!!
    Tree.nodeprob= -ones(numNodes,1); % not supported
    Tree.minval  = -ones(numNodes,1); % not supported
    Tree.maxval  = -ones(numNodes,1); % not supported
    Tree.risk    = -ones(numNodes,1); % not supported
end
