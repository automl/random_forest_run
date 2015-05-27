package ca.ubc.cs.beta.models.rf;

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
    public String method = "regression";
    
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
        method = t.method;

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
                nodepred[node] = Math.log10(mean(results));
                nodevar[node] = 0;
            } else {
                nodepred[node] = mean(results);
                nodevar[node] = var(results);
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
     * Propogates toBeMarginalizedObj down the tree and collect the means and weights of the leaves that each row lands in.
     * @params toBeMarginalizedObj a numvars-dimensional matrix numvars is the dimensions of the training data. Results will be returned for each row.
     * @returns a number of leaves*(1+numvars) matrix where 
          result[j][0] is the mean of leaf j, and
          result[j][i] for i>=1 is the ratio of elements of row i (also variable i) of toBeMarginalized that end up in leaf j
     */
    public static double[][] getLeafInfoForANOVA(Regtree tree, Object[] toBeMarginalizedObj) {
        if (toBeMarginalizedObj == null) {
            return null;
        }
        if (toBeMarginalizedObj.length != tree.npred) {
            throw new RuntimeException("toBeMarginalizedObj.length must be equal to numvars.");
        }
        double[][] toBeMarginalized = new double[toBeMarginalizedObj.length][];
        for (int i=0; i < toBeMarginalized.length; i++) {
            if (toBeMarginalizedObj[i] instanceof double[]) {
                toBeMarginalized[i] = (double[])toBeMarginalizedObj[i];
            } else {
                toBeMarginalized[i] = new double[1];
                toBeMarginalized[i][0] = (Double)toBeMarginalizedObj[i];
            }
        }
        
        int logModel = tree.logModel;
        
        int leafNodeIdx[] = new int[tree.numNodes];
        int numleaves = 0;
        for (int i=0; i < tree.numNodes; i++) {
            if (tree.var[i] == 0) {
                leafNodeIdx[i] = numleaves++;
            }
        }
        
        // Create output array
        double[][] retn = new double[numleaves][toBeMarginalized.length+1];
        
        // Initialize output array
        int counter = 0;            
        for (int i=0; i < tree.numNodes; i++) {
            if (tree.var[i] == 0) {
                // Store mean of this leaf
                retn[counter++][0] = (logModel == 1 ? Math.pow(10, tree.nodepred[i]) : tree.nodepred[i]);
            }
        }
        
        LinkedList<Integer> queue = new LinkedList<Integer>();
        for (int i=0; i < toBeMarginalized.length; i++) {
            int nextvar = i+1;
            int numValues = toBeMarginalized[i].length;
            
            counter = 0;

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
                            // We are in leaf node. store results.
                            int leaf = leafNodeIdx[thisnode];
                            retn[leaf][nextvar] += 1.0;
                            break;
                        } else if (Math.abs(splitvar) != nextvar) {
                            // Splitting on different var - pass this down both children
                            queue.add(right_kid);
                            thisnode = left_kid;
                        } else {
                            if (splitvar > 0) { // continuous
                                thisnode = (toBeMarginalized[i][j] <= cutoff ? left_kid : right_kid);
                            } else { // categorical
                                int x = (int)toBeMarginalized[i][j];
                                int split = tree.catsplit[(int)cutoff][x-1];
                                if (split == 0) thisnode = left_kid;
                                else if (split == 1) thisnode = right_kid;
                                else throw new RuntimeException("Missing value -- not allowed in this implementation.");
                            }
                        }
                    }
                }
            }
            
            for (int j=0; j < numleaves; j++) {
                retn[j][nextvar] /= numValues;
            }
        }
        return retn;
    }    
    
    // FOR TESTING PURPOSES ONLY!
    public void setysub(Object[][] y, Object[][] cens) {
        numNodes = y.length;
        this.ysub = new double[numNodes][];
        this.is_censored = new boolean[numNodes][];
        
        for (int i=0; i < numNodes; i++) {
            Object censsub = cens[i][0];
            if (censsub == null) continue;
            if (censsub instanceof int[]) {
                int[] t = (int[])censsub;
                this.is_censored[i] = new boolean[t.length];
                for (int j=0; j < t.length; j++) is_censored[i][j] = t[j] == 1;
            } else {
                this.is_censored[i] = new boolean[1];
                this.is_censored[i][0] = (Integer)censsub == 1;
            }
        }
        
        for (int i=0; i < numNodes; i++) {
            Object ysub = y[i][0];
            if (ysub == null) continue;
            if (ysub instanceof double[]) {
                double[] t = (double[])ysub;
                this.ysub[i] = new double[t.length];
                System.arraycopy(t, 0, this.ysub[i], 0, t.length);
            } else {
                this.ysub[i] = new double[1];
                this.ysub[i][0] = (Double)ysub;
            }
        }
        recalculateStats();
    }
    
    // FOR TESTING PURPOSES ONLY!
    public void setcatsplit(int[] splitvars, Object[][] catsplit, int[] domain_sizes) {
        this.catsplit = new int[catsplit.length][];
        
        int nextcatsplit = -1;
        for (int i=0; i < catsplit.length; i++) {
            while(splitvars[++nextcatsplit] >= 0);
            int[] cs = new int[domain_sizes[-splitvars[nextcatsplit]-1]];
            Arrays.fill(cs, -1);
            for (int a = 0; a <= 1; a++) {
                Object c = catsplit[i][a];
                int[] t;
                if (c instanceof int[][]) {
                    t = ((int[][])c)[0];
                } else if (c instanceof int[]) {
                    t = (int[])c;
                } else {
                    t = new int[1];
                    t[0] = (Integer)c;
                }
                for (int j=0; j < t.length; j++) {
                    cs[t[j]-1] = a;
                }
            }
            this.catsplit[i] = cs;
        }
    }
    
    public static double mean(double[] arr) {
        if (arr == null) return 0;
        int l = arr.length;
        double res = 0;
        for (int i=0; i < l; i++) {
            res += arr[i] / l;
        }
        return res;
    }
    
    public static double var(double[] arr) {
        if (arr == null) return 0;
        int l = arr.length;
        if (l <= 1) return 0;
        
        double m = mean(arr);
        double res = 0;
        for (int i=0; i < l; i++) {
            res += (arr[i] - m) * (arr[i] - m) / (l-1);
        }
        return res;
    }
    
    public static double median(double[] arr) {
        if (arr == null) return Double.NaN;
        int l = arr.length;
        if (l == 0) return Double.NaN;
        Arrays.sort(arr);
        return arr[(int)Math.floor(l/2.0)] / 2 + arr[(int)Math.ceil(l/2.0)] / 2;
    }
}