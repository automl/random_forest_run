function outfig = fh_treedisp(Tree,varargin)
% treedisp doesn't work with nodesize and parent being integer 
% (which they totally should be ...)
Tree.nodesize = double(Tree.nodesize);
Tree.parent = double(Tree.parent);
treedisp(Tree,varargin{:});
