package ca.ubc.cs.beta.models.fastrf;

import java.util.*;

public class Regtree implements java.io.Serializable {    
    private static final long serialVersionUID = -7861532246973394126L;
    
    public int numNodes;
    public int[] node;
    public int[] parent;
    public double[][] ysub;
    public boolean[][] is_censored;
    public int[] var;
    public double[] cut;
    public int[][] children;
    public int[] nodesize;
    public int npred;
    public int[][] catsplit;
    
    public double[] nodepred;
    public double[] nodevar;
    
    public boolean resultsStoredInLeaves;
    
    public boolean preprocessed;
    public double[] weightedpred;
    public double[] weights;
    
    public int logModel;

    public Regtree(int numNodes, int logModel) {
        this.numNodes = numNodes;
        this.logModel = logModel;
        preprocessed = false;
    }
    
    public Regtree(int numNodes, int ncatsplit, boolean storeResultsInLeaves, int logModel) {
        this(numNodes, logModel);
        
        resultsStoredInLeaves = storeResultsInLeaves;
        
        node = new int[numNodes];
        parent = new int[numNodes];
        var = new int[numNodes];
        cut = new double[numNodes];
        children = new int[numNodes][2];
        nodesize = new int[numNodes];
        catsplit = new int[ncatsplit][];
        
        if (resultsStoredInLeaves) {
            ysub = new double[numNodes][];
            is_censored = new boolean[numNodes][];
        } else {
            ysub = new double[numNodes][2]; // sum, sumofsq
        }
    }
    
    public Regtree(Regtree t) {
        this(t.numNodes, t.catsplit.length, t.resultsStoredInLeaves, t.logModel);
        
        npred = t.npred;     

        System.arraycopy(t.node, 0, node, 0, numNodes);
        System.arraycopy(t.parent, 0, parent, 0, numNodes);
        System.arraycopy(t.var, 0, var, 0, numNodes);
        System.arraycopy(t.cut, 0, cut, 0, numNodes);
        System.arraycopy(t.nodesize, 0, nodesize, 0, numNodes);
        
        for (int i=0; i < t.catsplit.length; i++) {
            catsplit[i] = new int[t.catsplit[i].length];
            catsplit[i] = new int[t.catsplit[i].length];
            System.arraycopy(t.catsplit[i], 0, catsplit[i], 0, t.catsplit[i].length);
        }
        
        for (int i=0; i < numNodes; i++) {
            children[i][0] = t.children[i][0];
            children[i][1] = t.children[i][1];
            
            if (resultsStoredInLeaves) {
                int Nnode = children[i][0] == 0 ? nodesize[i] : 0;
                if (Nnode != 0) {
                    ysub[i] = new double[Nnode];
                    is_censored[i] = new boolean[Nnode];
                
                    System.arraycopy(t.ysub[i], 0, ysub[i], 0, Nnode);
                    System.arraycopy(t.is_censored[i], 0, is_censored[i], 0, Nnode);
                }
            } else {
                System.arraycopy(t.ysub[i], 0, ysub[i], 0, t.ysub[i].length);
            }
        }
        
        preprocessed = t.preprocessed;
        if (preprocessed) {
            weights = new double[numNodes];
            weightedpred = new double[numNodes];
            System.arraycopy(t.weights, 0, weights, 0, numNodes);
            System.arraycopy(t.weightedpred, 0, weightedpred, 0, numNodes);
        }
        
        recalculateStats();
    }
    
    /**
     * Gets a prediction for the given instantiations of features
     * @returns a matrix of size X.length*2 where index (i,0) is the prediction for X[i] 
     * and (i,1) is the variance of that prediction.
     */
    public static double[][] apply(Regtree tree, double[][] X) {
        int[] nodes = RegtreeFwd.fwd(tree, X);
        
        double[][] retn = new double[X.length][2]; // mean, var
        for (int i=0; i < X.length; i++) {
            retn[i][0] = tree.nodepred[nodes[i]];
            retn[i][1] = tree.nodevar[nodes[i]];
        }
        return retn;
    }
    
    /**
     * Gets a prediction for the given configurations and instances
     * @see RegtreeFwd.marginalFwd
     */
    public static double[] applyMarginal(Regtree tree, double[][] Theta, double[][] X) {
        return RegtreeFwd.marginalFwd(tree, Theta, X);
    }
    
    /**
     * Recalculate statistic (mean, var) of the entire tree
     */
    public void recalculateStats() {
        nodepred = new double[numNodes];
        nodevar = new double[numNodes];
        
        for (int i=0; i < numNodes; i++) {
            recalculateStats(i);
        }
    }
    
    /**
     * Recalculate statistic (mean, var) of the specified node
     */
    public void recalculateStats(int node) {
        if (children[node][0] != 0) {
            return;
        }
        if (resultsStoredInLeaves) {
            double[] results = Arrays.copyOf(ysub[node], ysub[node].length);
            if (logModel == 1) {
                nodepred[node] = Math.log10(Utils.mean(results));
                nodevar[node] = 0;
            } else {
                nodepred[node] = Utils.mean(results);
                nodevar[node] = Utils.var(results);
            }
        } else {
            double sum = ysub[node][0], sumOfSq = ysub[node][1];
            int N = nodesize[node];
            
            if (logModel == 1) {
                nodepred[node] = Math.log10(sum/N);
                nodevar[node] = 0;
            } else {
                nodepred[node] = sum/N;
                nodevar[node] = (sumOfSq - sum*sum/N) / (N-1);
            }
        }
    }
    
    /**
     * Marginalizes over toBeMarginalized and returns a marginal prediction for each row of X.
     * Different from applyMarginalModel in that toBeMarginalized is not a list of data points but we use the cartesian product of the rows of toBeMarginalized.
     * @params allvars a cell array with numvars rows where each row is the domain of a variable.
     * @params variablesToKeep a d*1 vector of the variable numbers of the rows of allvars that we don't want to marginalize out.
     * I will refer to Xvars as the variables to keep and toBeMarginalized as the variables to marginalize out.
     * @returns a prod(Xvars[i].length)*1 vector of marginalized predictions which results from taking the variable instantiations from Xvars in increasing order.
     * eg. Xvars = [[1,2,3],[1,2],[1]], then return[0] is prediction for [1,1,1] = Xvars[0,0,0], return[1] is prediction for [1,2,1] = Xvars[0,1,0] since there is no Xvars[0,0,1], etc
     * until return[5] is prediction for [3,2,1] which is Xvars[2,1,0].
     */
    public static double[] getMarginal(Regtree tree, double[][] allvars, int[] variablesToKeep) {  
        if (variablesToKeep == null) {
            variablesToKeep = new int[0];
        }
        if (allvars.length != tree.npred) {
            throw new RuntimeException("variable domains for all variables must be provided");
        }
       
        // Fill in auxillary arrays
        int[] toBeMarginalizedVariables = new int[tree.npred-variablesToKeep.length];
        int[] variablesToKeep_IndexOf = new int[tree.npred+1];
        Arrays.fill(variablesToKeep_IndexOf, -1);
        int counter = 0, next = 1;
        for (int i=0; i < variablesToKeep.length; i++) {
            while (variablesToKeep[i] != next) {
                toBeMarginalizedVariables[counter++] = next++;
            }
            variablesToKeep_IndexOf[variablesToKeep[i]] = i;
            next++;
        }
        while(next <= tree.npred) {
            toBeMarginalizedVariables[counter++] = next++;
        }
        
        int logModel = tree.logModel;
        
        LinkedList<Integer> queue = new LinkedList<Integer>();
  
        // element i of result is # of entries of cartesian product of toBeMarginalized that fall into node i.
        double[] weights = new double[tree.numNodes];
        Arrays.fill(weights, 1);

        // For each row of toBeMarginalized, forward it and for each leaf store # of elements that fall into it.
        for (int i=0; i < toBeMarginalizedVariables.length; i++) {
            int nextvar = toBeMarginalizedVariables[i];

            // counts[j] is # of times some element from row i falls into node j.
            int[] counts = new int[tree.numNodes];
            int numValues = allvars[nextvar].length;

            for (int j=0; j < numValues; j++) {
                queue.add(0);
                while(!queue.isEmpty()) {
                    int thisnode = queue.poll();
                    while(true) {
                        int splitvar = tree.var[thisnode];
                        double cutoff = tree.cut[thisnode];
                        int left_kid = tree.children[thisnode][0];
                        int right_kid = tree.children[thisnode][1];

                        if (splitvar == 0) {
                            // We are in leaf node. store weights.
                            counts[thisnode]++;
                            break;
                        } else if (Math.abs(splitvar) != nextvar) {
                            // Splitting on different var - pass this down both children
                            queue.add(right_kid);
                            thisnode = left_kid;
                        } else {
                            if (splitvar > 0) { // continuous
                                thisnode = (allvars[nextvar][j] <= cutoff ? left_kid : right_kid);
                            } else { // categorical
                                int x = (int)allvars[nextvar][j];
                                int split = tree.catsplit[(int)cutoff][x-1];
                                if (split == 0) thisnode = left_kid;
                                else if (split == 1) thisnode = right_kid;
                                else throw new RuntimeException("Missing value -- not allowed in this implementation.");
                            }
                        }
                    }
                }
            }

            // Multiply with existing weights (the cardinality of the cartesian product of the actual values so far)
            for (int j=0; j < tree.numNodes; j++) {
                weights[j] *= counts[j]*1.0/numValues;
            }
        }

        //=== Calculate marginal means for X.
        double[] retn;
        if (variablesToKeep.length==0) {
            // Special case if X is empty.
            retn = new double[1];
            double sum = 0;
            for (int i=0; i < weights.length; i++) sum += weights[i];
            if (sum - 1 > 1e-6) {
                throw new RuntimeException("Something is wrong. Sum: " + sum + "(" + Arrays.toString(weights) + ")");
            }
            for (int i=0; i < weights.length; i++) {
                double pred = (logModel == 1 ? Math.pow(10, tree.nodepred[i]) : tree.nodepred[i]);
                retn[0] += weights[i]*pred;
            }
        } else {
            int[] state = new int[variablesToKeep.length];
            int numData = 1;
            for (int i=0; i < variablesToKeep.length; i++) {
                state[i] = 0;
                numData *= allvars[variablesToKeep[i]].length;
            }
            if (numData == 0) throw new RuntimeException("Cannot have a variable with an empty domain!");
            
            retn = new double[numData];
            // For each combination of variables to keep, multiply prediction with weights.
            for (int i=0; i < numData; i++) {
                queue.add(0);
                while(!queue.isEmpty()) {
                    int thisnode = queue.poll();
                    while(true) {
                        int splitvar = tree.var[thisnode];
                        double cutoff = tree.cut[thisnode];
                        int left_kid = tree.children[thisnode][0];
                        int right_kid = tree.children[thisnode][1];
                        int varidx = variablesToKeep_IndexOf[Math.abs(splitvar)];

                        if (splitvar == 0) {
                            // We are in leaf node. store weights.
                            double pred = (logModel == 1 ? Math.pow(10, tree.nodepred[thisnode]) : tree.nodepred[thisnode]);
                            retn[i] += weights[thisnode]*pred;
                            break;
                        } else if (varidx == -1) {
                            // Splitting on different var - pass this down both children
                            queue.add(right_kid);
                            thisnode = left_kid;
                        } else {
                            if (splitvar > 0) { // continuous
                                thisnode = (allvars[variablesToKeep[varidx]][state[varidx]] <= cutoff ? left_kid : right_kid);
                            } else { // categorical
                                int x = (int)allvars[variablesToKeep[varidx]][state[varidx]];
                                int split = tree.catsplit[(int)cutoff][x-1];
                                if (split == 0) thisnode = left_kid;
                                else if (split == 1) thisnode = right_kid;
                                else throw new RuntimeException("Missing value -- not allowed in this implementation.");
                            }
                        }
                    }
                }
            }
            for (int j = variablesToKeep.length-1; j >= 0; j--) {
                state[j]++;
                if (state[j] == allvars[variablesToKeep[j]].length) {
                    state[j] = 0;
                } else {
                    break;
                }
            }
        }
        for (int i=0; i < retn.length; i++) {
            if (logModel == 1) {
                retn[i] = Math.log10(retn[i]);
            }
        }
        return retn;
    }
}