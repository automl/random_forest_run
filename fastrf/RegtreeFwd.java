package ca.ubc.cs.beta.models.fastrf;

import java.util.*;
public class RegtreeFwd {
    /**
     * Propogates data points down the regtree and returns a 1*X.length vector of node #s
     * specifying which node each data point falls into.
     * @param tree the regtree to use
     * @param X a numdatapoints*numvars matrix
     */
    public static int[] fwd(Regtree tree, double[][] X) {  
        int numdata = X.length;
        int numnodes = tree.node.length;
        if (tree.cut.length != numnodes) {
            throw new RuntimeException("cut must be Nx1 vector.");
        }
        if (tree.nodepred.length != numnodes) {
            throw new RuntimeException("nodepred must be Nx1 vector.");
        }
        if (tree.children.length != numnodes) {
            throw new RuntimeException("children must be Nx2 matrix.");
        }
        
        int[] result = new int[numdata];
        
        for (int i=0; i < numdata; i++) {
            int thisnode = 0;
            while(true) {
                int splitvar = tree.var[thisnode];
                if (splitvar == 0) {
                    // This node not split, store results.
                    result[i] = thisnode;
                    break;
                }
                double cutoff = tree.cut[thisnode];
                int left_kid = tree.children[thisnode][0];
                int right_kid = tree.children[thisnode][1];
                // Determine if the points goes left or goes right
                if (splitvar > 0) { 
                    // continuous variable
                    thisnode = (X[i][splitvar-1] <= cutoff ? left_kid : right_kid);
                } else { 
                    // categorical variable
                    int x = (int)X[i][-splitvar-1];
                    int split = tree.catsplit[(int)cutoff][x-1];
                    if (split == 0) thisnode = left_kid;
                    else if (split == 1) thisnode = right_kid;
                    else throw new RuntimeException("Missing value -- not allowed in this implementation.");
                }
            }
        }
        return result;
    }
    
    /**
     * Propogates configurations(Theta) and instances(X) down the tree, and returns a 1*Theta.length vector of 
     * marginal prediction for each configuration (summed across each of the specified instances).
     * @param tree the regtree to use
     * @param Theta a vector of configuration parameters
     * @param X a vector of instance parameters. If the tree has already been preprocessed, this argument is ignored.
     */
    public static double[] marginalFwd(Regtree tree, double[][] Theta, double[][] X) {
        if (Theta == null || Theta.length == 0) {
            throw new RuntimeException("Theta must not be empty");
        }
        int thetarows = Theta.length;
        int thetacols = Theta[0].length;
        int numnodes = tree.node.length;
        if (numnodes == 0) {
            throw new RuntimeException("Tree must exist.");
        }
        if (tree.cut.length != numnodes) {
            throw new RuntimeException("cut must be Nx1 vector.");
        }
        if (tree.nodepred.length != numnodes) {
            throw new RuntimeException("nodepred must be Nx1 vector.");
        }
        if (tree.parent.length != numnodes) {
            throw new RuntimeException("parent must be Nx1 matrix.");
        }
        if (tree.children.length != numnodes) {
            throw new RuntimeException("children must be Nx2 matrix.");
        }
        
        Regtree preprocessed;
        if (tree.preprocessed) {
            preprocessed = tree;
        } else {
            preprocessed = preprocess_inst_splits(tree, X);
        }
        
        double[] result = new double[thetarows];
        
        LinkedList<Integer> queue = new LinkedList<Integer>();
        
        for (int i=0; i < thetarows; i++) {
            queue.add(0);
            while(!queue.isEmpty()) {
                int thisnode = queue.poll();
                while(true) {
                    int splitvar = preprocessed.var[thisnode];
                    double cutoff = preprocessed.cut[thisnode];
                    int left_kid = preprocessed.children[thisnode][0];
                    int right_kid = preprocessed.children[thisnode][1];

                    if (splitvar == 0) {
                        // We are in leaf node. store results.
                        result[i] += preprocessed.weightedpred[thisnode];
                        break;
                    } else if (Math.abs(splitvar) > thetacols) {
                        // Splitting on instance - pass this instance down both children
                        queue.add(right_kid);
                        thisnode = left_kid;
                    } else {
                        if (splitvar > 0) { // continuous
                            thisnode = (Theta[i][splitvar-1] <= cutoff ? left_kid : right_kid);
                        } else { // categorical
                            int x = (int)Theta[i][-splitvar-1];
                            int split = preprocessed.catsplit[(int)cutoff][x-1];
                            if (split == 0) thisnode = left_kid;
                            else if (split == 1) thisnode = right_kid;
                            else throw new RuntimeException("Missing value -- not allowed in this implementation.");
                        }
                    }
                }
            }
        }
        
        if (tree.logModel == 1) {
            for (int i=0; i < thetarows; i++) {
                result[i] = Math.log10(result[i]);
            }
        }
        return result;
    }
    
    
    /**
     * Propogates configurations(Theta) and instances(X) down the tree, and returns a size 2 array containing
     * leafIdxs : vector. indices of leaves that at least one given configuration made it to
     * ThetaIdxs : matrix of size leafIdxs.length*unknown. ThetaIdxs[i][j] means Theta[j] made it into leaf[leafIdxs[i]].
     * @param tree the regtree to use
     * @param Theta a vector of configuration parameters
     * @param X a vector of instance parameters. If the tree has already been preprocessed, this argument is ignored.
     */
    public static Object[] fwdThetas(Regtree tree, double[][] Theta) {
        if (Theta == null || Theta.length == 0) {
            throw new RuntimeException("Theta must not be empty");
        }
        int thetarows = Theta.length;
        int thetacols = Theta[0].length;
        int numnodes = tree.node.length;
        if (numnodes == 0) {
            throw new RuntimeException("Tree must exist.");
        }
        if (tree.cut.length != numnodes) {
            throw new RuntimeException("cut must be Nx1 vector.");
        }
        if (tree.nodepred.length != numnodes) {
            throw new RuntimeException("nodepred must be Nx1 vector.");
        }
        if (tree.parent.length != numnodes) {
            throw new RuntimeException("parent must be Nx1 matrix.");
        }
        if (tree.children.length != numnodes) {
            throw new RuntimeException("children must be Nx2 matrix.");
        }
        
        Regtree preprocessed;
        if (tree.preprocessed) {
            preprocessed = tree;
        } else {
            throw new RuntimeException("fwdThetas can only be called on a preprocessed tree.");
        }
        
        int numLeavesWithTheta = 0;
        int[][] thetaIdxs = new int[numnodes][thetarows];
        int[] thetaIdxsLens = new int[numnodes];
        Arrays.fill(thetaIdxsLens, 0);
        
        LinkedList<Integer> queue = new LinkedList<Integer>();
        for (int i=0; i < thetarows; i++) {
            queue.add(0);
            while(!queue.isEmpty()) {
                int thisnode = queue.poll();
                while(true) {
                    int splitvar = preprocessed.var[thisnode];
                    double cutoff = preprocessed.cut[thisnode];
                    int left_kid = preprocessed.children[thisnode][0];
                    int right_kid = preprocessed.children[thisnode][1];

                    if (splitvar == 0) {
                        // We are in leaf node. store results.
                        if (thetaIdxsLens[thisnode]==0) {
                            numLeavesWithTheta++;                            
                        }
                        thetaIdxs[thisnode][thetaIdxsLens[thisnode]++] = i;
                        break;
                    } else if (Math.abs(splitvar) > thetacols) {
                        // Splitting on instance - pass this instance down both children
                        queue.add(right_kid);
                        thisnode = left_kid;
                    } else {
                        if (splitvar > 0) { // continuous
                            thisnode = (Theta[i][splitvar-1] <= cutoff ? left_kid : right_kid);
                        } else { // categorical
                            int x = (int)Theta[i][-splitvar-1];
                            int split = preprocessed.catsplit[(int)cutoff][x-1];
                            if (split == 0) thisnode = left_kid;
                            else if (split == 1) thisnode = right_kid;
                            else throw new RuntimeException("Missing value -- not allowed in this implementation.");
                        }
                    }
                }
            }
        }
        
        int[] leafIdxs = new int[numLeavesWithTheta];
        int[][] condensedThetaIdxs = new int[numLeavesWithTheta][];
        for (int i=0, counter=0; i < numnodes; i++) {
            if (thetaIdxsLens[i] == 0) continue;
            leafIdxs[counter] = i;
            condensedThetaIdxs[counter] = new int[thetaIdxsLens[i]];
            System.arraycopy(thetaIdxs[i], 0, condensedThetaIdxs[counter], 0, thetaIdxsLens[i]);
            counter++;
        }
        
        Object[] retn = new Object[2];
        retn[0] = leafIdxs;
        retn[1] = condensedThetaIdxs;
        return retn;
    }
    
    /**
     * Preprocesses the regtree for marginal predictions using the specified instances. 
     * A call to preprocess_inst_splits(tree,X) followed by marginalFwd(tree,Theta,null) 
     * is equivalent to the call marginalFwd(tree,Theta,X)
     * except that marginalFwd preprocesses the tree internally.
     */
    public static Regtree preprocess_inst_splits(Regtree tree, double[][] X) {
        Regtree newtree = new Regtree(tree);
        
        int numnodes = newtree.node.length;
        if (numnodes == 0) {
            throw new RuntimeException("Tree must exist.");
        }
        if (newtree.cut.length != numnodes) {
            throw new RuntimeException("cut must be Nx1 vector.");
        }
        if (newtree.nodepred.length != numnodes) {
            throw new RuntimeException("nodepred must be Nx1 vector.");
        }
        if (newtree.parent.length != numnodes) {
            throw new RuntimeException("parent must be Nx1 matrix.");
        }
        if (newtree.children.length != numnodes) {
            throw new RuntimeException("children must be Nx2 matrix.");
        }
        
        newtree.weights = new double[numnodes];
        newtree.weightedpred = new double[numnodes];
            
        if (X == null) {
            for (int i=0; i < numnodes; i++) {
                newtree.weights[i] = 0;
                if (newtree.logModel == 1) {
                    newtree.weightedpred[i] = Math.pow(10, newtree.nodepred[i]);
                } else {
                    newtree.weightedpred[i] = newtree.nodepred[i];
                }
            }
            newtree.preprocessed = true;
            return newtree;
        }
       
        int numinsts = X.length;       
        int thetacols = newtree.npred - X[0].length;

        LinkedList<Integer> queue = new LinkedList<Integer>();

        // Preprocess tree by forwarding instances and getting weights
        Arrays.fill(newtree.weights, 0.0);
        Arrays.fill(newtree.weightedpred, 0.0);

        for (int i=0; i < numinsts; i++) {
            queue.add(0);
            while(!queue.isEmpty()) {
                int thisnode = queue.poll();
                while(true) {
                    int splitvar = newtree.var[thisnode];
                    double cutoff = newtree.cut[thisnode];
                    int left_kid = newtree.children[thisnode][0];
                    int right_kid = newtree.children[thisnode][1];

                    if (splitvar == 0) {
                        // We are in leaf node. 
                        if (newtree.logModel == 1) {
                            newtree.weightedpred[thisnode]+=Math.pow(10, newtree.nodepred[thisnode])/numinsts;
                        } else {
                            newtree.weightedpred[thisnode]+=newtree.nodepred[thisnode]/numinsts;
                        }
                        newtree.weights[thisnode] += 1.0/numinsts;
                        break;
                    } else if (Math.abs(splitvar) <= thetacols) {
                        // Splitting on Theta - pass this instance down both children
                        queue.add(right_kid);
                        thisnode = left_kid;
                    } else {
                        if (splitvar > 0) { // continuous
                            thisnode = (X[i][splitvar-1-thetacols] <= cutoff ? left_kid : right_kid);
                        } else { // categorical
                            int x = (int)X[i][-splitvar-1-thetacols];
                            int split = newtree.catsplit[(int)cutoff][x-1];
                            if (split == 0) thisnode = left_kid;
                            else if (split == 1) thisnode = right_kid;
                            else throw new RuntimeException("Missing value -- not allowed in this implementation.");
                        }
                    }
                }
            }
        }

        // cut leaf splits on instances. 
        cut_instance_leaf_split_helper(newtree, thetacols, 0);

        newtree.preprocessed = true;
        return newtree;
    }
    
    // Returns whether thisnode is a leaf at the end of the function
    private static int cut_instance_leaf_split_helper(Regtree tree, int thetacols, int thisnode) {
        int left_kid = tree.children[thisnode][0], right_kid = tree.children[thisnode][1];
        int ret = 0;
        
        if (tree.var[thisnode] == 0) {
            ret = 1;
        } else if (
                   cut_instance_leaf_split_helper(tree, thetacols, left_kid)
                   + 
                   cut_instance_leaf_split_helper(tree, thetacols, right_kid) == 2 // This is so we don't short-circuit
                   && Math.abs(tree.var[thisnode]) > thetacols
                ) {
                // both children are leaves, and this is a split on an instance
                make_into_leaf(tree, thisnode);
                ret = 1;
        }
        
        if (thisnode != 0) {
            tree.weightedpred[tree.parent[thisnode]] += tree.weightedpred[thisnode];
            tree.weights[tree.parent[thisnode]] += tree.weights[thisnode];
        }
        if (ret == 0 && tree.weights[thisnode] == 0) {
            make_into_leaf(tree, thisnode);
            ret = 1;
        }
        return ret;
    }
    
    private static void make_into_leaf(Regtree tree, int thisnode) {
        tree.var[thisnode] = 0;
        int left_kid = tree.children[thisnode][0], right_kid = tree.children[thisnode][1];
        tree.children[thisnode][0] = 0;
        tree.children[thisnode][1] = 0;
        // combine stats of children
        if (tree.resultsStoredInLeaves) {
            tree.ysub[thisnode] = new double[tree.nodesize[thisnode]];
            System.arraycopy(tree.ysub[left_kid], 0, tree.ysub[thisnode], 0, tree.nodesize[left_kid]);
            System.arraycopy(tree.ysub[right_kid], 0, tree.ysub[thisnode], tree.nodesize[left_kid], tree.nodesize[right_kid]);

            tree.is_censored[thisnode] = new boolean[tree.nodesize[thisnode]];
            System.arraycopy(tree.is_censored[left_kid], 0, tree.is_censored[thisnode], 0, tree.nodesize[left_kid]);
            System.arraycopy(tree.is_censored[right_kid], 0, tree.is_censored[thisnode], tree.nodesize[left_kid], tree.nodesize[right_kid]);
        } else {
            for (int i=0; i < tree.ysub[thisnode].length; i++) {
                tree.ysub[thisnode][i] = tree.ysub[left_kid][i] + tree.ysub[right_kid][i];
            }
        }
        tree.recalculateStats(thisnode);
    }
}