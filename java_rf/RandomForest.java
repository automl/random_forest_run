package ca.ubc.cs.beta.models.rf;

import java.util.*;

public class RandomForest implements java.io.Serializable {
    private static final long serialVersionUID = 5204746081208095703L;
    
    public int numTrees;
    public Regtree[] Trees;
    
    public RandomForest(int numtrees) {
        if (numtrees <= 0) {
            throw new RuntimeException("Invalid number of regression trees in forest: " + numtrees);
        }
        numTrees = numtrees;
        Trees = new Regtree[numtrees];
    }
  
    /**
     * Learns a random forst using all data points in each tree.
     */
    public static RandomForest learnModel(int numTrees, RegtreeBuildParams params) {
        Random r = params.random;
        if (r == null) {
            r = new Random();
            if (params.seed != -1) {
                r.setSeed(params.seed);
            }
            params.random = r;
        }
        
        int numData = params.X.length;
        int[][] treeDataIdxs = new int[numTrees][numData];
        for (int i=0; i < numTrees; i++) {
            for (int j=0; j < numData; j++) {
                treeDataIdxs[i][j] = r.nextInt(numData);
            }
        }
        return learnModel(numTrees, treeDataIdxs, params);
    }
    
    /**
     * Learns a random forest putting the specified data points into each tree.
     * @params tree_data_idxs a vector of size numtrees specifying which indices of X to use in building that tree.
     * @see RegtreeFit.fit
     */
    public static RandomForest learnModel(int numTrees, int[][] treeDataIdxs, RegtreeBuildParams params) {       
        if (treeDataIdxs.length != numTrees) {
            throw new RuntimeException("length(treeDataIdxs) must be equal to numtrees.");
        }
        
        ArrayList<Integer> censored_idxs = new ArrayList<Integer>();
		for (int i = 0; i < params.cens.length; i++) {
			// find model.cens
			if (params.cens[i]) censored_idxs.add(i);
		}
		
        boolean storeResponses = params.storeResponses;
        RandomForest rf = new RandomForest(numTrees);
        
        if (censored_idxs.size() > 0) {
            params.storeResponses = true;
            for (int m = 0; m < numTrees; m++) {
                rf.Trees[m] = RegtreeFit.fit(treeDataIdxs[m], params);
            }
            
            double[] lowerBoundForSamples = new double[censored_idxs.size()];
            double[][] censoredX = new double[censored_idxs.size()][];
			for (int i = 0; i < lowerBoundForSamples.length; i++) {
                int idx = censored_idxs.get(i);
				lowerBoundForSamples[i] = params.y[idx];
                censoredX[i] = params.X[idx];
			}
			
			double maxY = params.y[0];
			for (int i = 1; i < params.y.length; i++) {
				maxY = Math.max(maxY, params.y[i]);
			}
			
			if (params.logModel == 1) {
				for (int i = 0; i < lowerBoundForSamples.length; i++) {
					lowerBoundForSamples[i] = Math.pow(10, lowerBoundForSamples[i]);
				}
				maxY = Math.pow(10, maxY);
			}
            
            if (params.logModel == 3) {
                params.kappa = Math.log10(params.kappa);
            }
            
			double valueForAllCens = params.cutoffPenaltyFactor * params.kappa;
			valueForAllCens = Math.max(valueForAllCens, maxY);
			
			int[][] numOccurrence = new int[censored_idxs.size()][numTrees];
            int[] numCensoredBefore = new int[params.y.length];
            for (int i=1; i < params.y.length; i++) {
                numCensoredBefore[i] = numCensoredBefore[i-1] + (params.cens[i-1] ? 1 : 0);
            }
            
            for (int m=0; m < numTrees; m++) {
                for (int i=0; i < treeDataIdxs[m].length; i++) {
                    int idx = treeDataIdxs[m][0];
                    if (params.cens[idx]) {
                        numOccurrence[numCensoredBefore[idx]][m]++;
                    }
                }
            }
            
            Object[][] result = RandomForest.hallucinateData(rf, censoredX, numOccurrence, lowerBoundForSamples, valueForAllCens, params.logModel);
            double[][][] convertedResult = new double[censored_idxs.size()][numTrees][];
            for (int i=0; i < censored_idxs.size(); i++) {
                for (int m=0; m < numTrees; m++) {
                    convertedResult[i][m] = (double[])(result[i][m]);
                }
            }
            
            double[] origy = params.y;
            boolean[] origcens = params.cens;
            params.cens = new boolean[params.cens.length]; // no censored data now - all imputed.
            
            for (int m=0; m < numTrees; m++) {
                params.y = origy.clone();
                for (int i=0; i < treeDataIdxs[m].length; i++) {
                    int idx = treeDataIdxs[m][i];
                    if (params.cens[idx]) {
                        double[] samples = convertedResult[numCensoredBefore[idx]][m];
                        params.y[idx] = samples[numOccurrence[numCensoredBefore[idx]][m]--];
                    }
                }
            }
        }
        params.storeResponses = storeResponses;
        for (int m = 0; m < numTrees; m++) {
            rf.Trees[m] = RegtreeFit.fit(treeDataIdxs[m], params);
            java.lang.System.gc();
        }
        return rf;
    }
    
    /**
     * Propogates data points down the regtree and returns a numtrees*X.length vector of node #s
     * specifying which node each data point falls into.
     * @params X a numdatapoints*numvars matrix
     */
    public static int[][] fwd(RandomForest forest, double[][] X) {
        int[][] retn = new int[forest.numTrees][X.length];
        for (int i=0; i < forest.numTrees; i++) {
            int[] result = RegtreeFwd.fwd(forest.Trees[i], X);
            System.arraycopy(result, 0, retn[i], 0, result.length);
        }
        return retn;
    }
    
    /**
     * Propogates configurations down the regtree and returns a numtrees*2 matrix containing 
     * leafIdxs and thetaIdxs.
     * @see RegtreeFwd.fwdThetas
     */
    public static Object[][] fwdThetas(RandomForest forest, double[][] Theta) {
        Object[][] retn = new Object[forest.numTrees][2];
        for (int i=0; i < forest.numTrees; i++) {
            Object[] result = RegtreeFwd.fwdThetas(forest.Trees[i], Theta);
            int[] leafIdxs = (int[]) result[0];
            int[][] ThetaIdxs = (int[][]) result[1];
            
            retn[i][0] = leafIdxs;
            retn[i][1] = ThetaIdxs;
        }
        return retn;
    }
    
    /**
     * Gets a prediction for the given instantiations of features
     * @returns a matrix of size X.length*2 where index (i,0) is the prediction for X[i] 
     * and (i,1) is the variance of that prediction.
     */
    public static double[][] apply(RandomForest forest, double[][] X) {
        double[][] treemeans = new double[X.length][forest.numTrees];
        double[][] treevars = new double[X.length][forest.numTrees];
        for (int i=0; i < forest.numTrees; i++) {
            int[] result = RegtreeFwd.fwd(forest.Trees[i], X);
            for (int j=0; j < result.length; j++) {
                treemeans[j][i] = forest.Trees[i].nodepred[result[j]];
                treevars[j][i] = forest.Trees[i].nodevar[result[j]];
            }
        }
        double[][] retn = new double[X.length][2]; // mean, var
        for (int i=0; i < X.length; i++) {
            retn[i][0] = Regtree.mean(treemeans[i]);
            retn[i][1] = Regtree.var(treemeans[i]);
        }
        return retn;
    }
    
    public static double[][] applyMarginal(RandomForest forest, double[][] Theta) {
        return applyMarginal(forest, Theta, null);
    }
    
    /**
     * Gets a prediction for the given configurations and instances
     * @returns a matrix of size Theta.length*2 where index (i,0) is the prediction for Theta[i] 
     * and (i,1) is the variance of that prediction.
     * @see RegtreeFwd.marginalFwd
     */
    public static double[][] applyMarginal(RandomForest forest, double[][] Theta, double[][] X) {
        int nTheta = Theta.length;
        
        double[][] treemeans = new double[nTheta][forest.numTrees];
        for (int i=0; i < forest.numTrees; i++) {
            double[] result = RegtreeFwd.marginalFwd(forest.Trees[i], Theta, X);

            for (int j=0; j < nTheta; j++) {
                treemeans[j][i] = result[j];
            }
        }
        
        double[][] retn = new double[nTheta][2]; // mean, var
        for (int i=0; i < nTheta; i++) {
            retn[i][0] = Regtree.mean(treemeans[i]);
            retn[i][1] = Regtree.var(treemeans[i]);
        }
        return retn;
    }
    
    /**
     * Marginalizes over toBeMarginalized and returns a marginal prediction for each row of X.
     * Different from applyMarginalModel in that toBeMarginalized is not a list of data points but we use the cartesian product of the rows of toBeMarginalized.
     * @params X a N*d matrix (N d-variable sets) to get the marginal for
     * @params toBeMarginalized a cell array with p rows where each row specifies the values for a variable to marginalize over.
     * @params variableIndicesForColumnsOfX a d*1 vector of the variable numbers each column of X represents. 1-indexed.
       d+p = numvars. The columns of toBeMarginalized represent the variables {0:numvars-1}-variableIndicesForColumnsOfX, in sorted increasing order.
     * @returns a N*1 vector of marginalized predictions.
     */
    public static double[] getMarginal(RandomForest forest, double[][] X, Object[] toBeMarginalizedObj, int[] variableIndicesForColumnsOfX) {  
        if (X == null) {
            X = new double[1][0];
        }
        if (toBeMarginalizedObj == null) {
            toBeMarginalizedObj = new Object[0];
        }
        if (variableIndicesForColumnsOfX == null) {
            variableIndicesForColumnsOfX = new int[0];
        }
        if (X[0].length + toBeMarginalizedObj.length != forest.Trees[0].npred) {
            throw new RuntimeException("d+p must be equal to numvars.");
        }
        if (X[0].length != variableIndicesForColumnsOfX.length) {
            throw new RuntimeException("The number of columns of X and the length of variableIndicesForColumnsOfX must match up.");
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
        
        // Fill in auxillary arrays
        int[] toBeMarginalizedVariables = new int[forest.Trees[0].npred-variableIndicesForColumnsOfX.length];
        int[] isXvar = new int[forest.Trees[0].npred+1];
        int counter = 0, next = 1;
        for (int i=0; i < variableIndicesForColumnsOfX.length; i++) {
            while (variableIndicesForColumnsOfX[i] != next) {
                toBeMarginalizedVariables[counter++] = next++;
            }
            isXvar[variableIndicesForColumnsOfX[i]] = i+1;
            next++;
        }
        while(next <= forest.Trees[0].npred) {
            toBeMarginalizedVariables[counter++] = next++;
        }
        
        int logModel = forest.Trees[0].logModel;
        
        // marginalized means for each X in each tree.
        double[][] treeMeans = new double[forest.numTrees][X == null ? 1 : X.length];
        
        LinkedList<Integer> queue = new LinkedList<Integer>();
        
        // Loop through each tree, for each tree calculate treeMeans
        for (int m=0; m < forest.numTrees; m++) {
            Regtree tree = forest.Trees[m];
        
            // element i of result is # of entries of cartesian product of toBeMarginalized that fall into node i.
            double[] results = new double[tree.numNodes];
            Arrays.fill(results, 1);
            
            // For each row of toBeMarginalized, forward it and for each leaf store # of elements that fall into it.
            for (int i=0; i < toBeMarginalized.length; i++) {
                int nextvar = toBeMarginalizedVariables[i];

                // counts[j] is # of times some element from row i falls into node j.
                int[] counts = new int[tree.numNodes];
                int numValues = toBeMarginalized[i].length;
                
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
                                counts[thisnode]++;
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
                
                // Multiply with existing results (the cardinality of the cartesian product of the actual values so far)
                for (int j=0; j < tree.numNodes; j++) {
                    results[j] *= counts[j]*1.0/numValues;
                }
            }
            
            //=== Calculate marginal means for X.
            if (X[0].length==0) {
                // Special case if X is empty.
                double sum = 0;
                for (int i=0; i < results.length; i++) sum += results[i];
                if (sum - 1 > 1e-6) {
                    throw new RuntimeException("Something is wrong. Sum: " + sum + "(" + Arrays.toString(results) + ")");
                }
                for (int i=0; i < results.length; i++) {
                    double pred = (logModel == 1 ? Math.pow(10, tree.nodepred[i]) : tree.nodepred[i]);
                    treeMeans[m][0] += results[i]*pred;
                }
            } else {
                // For each row of X, get the marginal mean from this tree.
                for (int i=0; i < X.length; i++) {
                    queue.add(0);
                    while(!queue.isEmpty()) {
                        int thisnode = queue.poll();
                        while(true) {
                            int splitvar = tree.var[thisnode];
                            double cutoff = tree.cut[thisnode];
                            int left_kid = tree.children[thisnode][0];
                            int right_kid = tree.children[thisnode][1];
                            int varidx = isXvar[Math.abs(splitvar)];

                            if (splitvar == 0) {
                                // We are in leaf node. store results.
                                double pred = (logModel == 1 ? Math.pow(10, tree.nodepred[thisnode]) : tree.nodepred[thisnode]);
                                treeMeans[m][i] += results[thisnode]*pred;
                                break;
                            } else if (varidx == 0) {
                                // Splitting on different var - pass this down both children
                                queue.add(right_kid);
                                thisnode = left_kid;
                            } else {
                                if (splitvar > 0) { // continuous
                                    thisnode = (X[i][varidx-1] <= cutoff ? left_kid : right_kid);
                                } else { // categorical
                                    int x = (int)X[i][varidx-1];
                                    int split = tree.catsplit[(int)cutoff][x-1];
                                    if (split == 0) thisnode = left_kid;
                                    else if (split == 1) thisnode = right_kid;
                                    else throw new RuntimeException("Missing value -- not allowed in this implementation.");
                                }
                            }
                        }
                    }
                }
            }
        }
        
        // calculate the mean of treeMeans
        double[] retn = new double[X.length];
        for (int i=0; i < retn.length; i++) {
            double sum = 0;
            for (int m=0; m < forest.numTrees; m++) sum += treeMeans[m][i];
            retn[i] = sum * 1.0 / forest.numTrees;
            if (logModel == 1) {
                retn[i] = Math.log10(retn[i]);
            }
        }
        return retn;
    }
    
    /** 
     * Prepares the random forest for marginal predictions.
     * @see RegtreeFwd.preprocess_inst_splits
     */
    public static RandomForest preprocessForest(RandomForest forest, double[][] X) {
        RandomForest prepared = new RandomForest(forest.numTrees);
        for (int i=0; i < forest.numTrees; i++) {
            prepared.Trees[i] = RegtreeFwd.preprocess_inst_splits(forest.Trees[i], X);
        }
        return prepared;
    }
    
    /**
     * Collect response values from leaves for the given data points.
     * @returns a 3*X.length*totalNumberOfDataPointsInLeaves matrix of y_pred, cens_pred, and weights.
     */
    public static Object[] collectData(RandomForest forest, double[][] X) {
        int[][] leafNodes = fwd(forest, X);
        return collectData(forest, leafNodes);
    }
    
    /**
     * Collect response values from the given leaves.
     * @paramss leafNodes a numTrees*X.length matrix of the nodes of the data points. (eg. from fwd)
     * @returns a 3*X.length*totalNumberOfDataPointsInLeaves matrix of y_pred, cens_pred, and weights.
     */
    public static Object[] collectData(RandomForest forest, int[][] leafNodes) {
        if (!forest.Trees[0].resultsStoredInLeaves) {
            throw new RuntimeException("Cannot collect data if they were not stored.");
        }
        
        Object[] retn = new Object[3];

        int numdata = leafNodes[0].length;
        
        double[][] y_pred_all = new double[numdata][];
        boolean[][] cens_pred_all = new boolean[numdata][];
        double[][] weights_all = new double[numdata][];
        
        
        for (int i=0; i < numdata; i++) {
            int size = 0;
            for (int j=0; j < forest.numTrees; j++) {
                size += forest.Trees[j].nodesize[leafNodes[j][i]];
            }
            
            double[] y_pred = new double[size];
            boolean[] cens_pred = new boolean[size];
            double[] weights = new double[size];
            
            int counter = 0;
            for (int j=0; j < forest.numTrees; j++) {
                Regtree tree = forest.Trees[j];
                int node = leafNodes[j][i];
                int nodesize = tree.nodesize[node];
                for (int k=0; k < nodesize; k++) {
                    y_pred[counter+k] = tree.ysub[node][k];
                    cens_pred[counter+k] = tree.is_censored[node][k];
                    if (!cens_pred[counter+k]) y_pred[counter+k] += 1e-10;
                }
                Arrays.fill(weights, counter, counter+nodesize, 1.0/nodesize);
                counter += nodesize;
            }
            
            y_pred_all[i] = y_pred;
            cens_pred_all[i] = cens_pred;
            weights_all[i] = weights;
        }
        retn[0] = y_pred_all;
        retn[1] = cens_pred_all;
        retn[2] = weights_all;
        return retn;
    }
    
    /**
     * Hallucinate response values for the given data points based on the distribution in the leaves that the data point falls into.
     * @returns a X.length*forest.numTrees*numOccurrence[i][j] matrix of hallucinated response values for each data point X[i] in tree[j].
     */
    public static Object[][] hallucinateData(RandomForest forest, double[][] X, int[][] numOccurrence, double[] lowerBoundForSamples, double valueForAllCens, int logModel) {
        Object[][] retn = new Object[X.length][forest.numTrees];
        Object[] collectedFromLeaves = collectData(forest, X);
        double[][] y_pred_all = (double[][])collectedFromLeaves[0];
        boolean[][] cens_pred_all = (boolean[][])collectedFromLeaves[1];
        double[][] weights_all = (double[][])collectedFromLeaves[2];
        
        int numdata = X.length;
        for (int i=0; i < numdata; i++) {
            int totalOccurrences = 0;
            for (int j=0; j < forest.numTrees; j++) {
                totalOccurrences += numOccurrence[i][j];
            }
            if (totalOccurrences == 0) continue;

            double[] y_pred = y_pred_all[i];
            boolean[] cens_pred = cens_pred_all[i];
            double[] weights = weights_all[i];
            
            double[] samples = WeibullFit.fit_dist_and_sample(y_pred, cens_pred, weights, totalOccurrences, lowerBoundForSamples[i], valueForAllCens);       
 
            int counter = 0;
            for (int j=0; j < forest.numTrees; j++) {
                int numOccurrenceHere = numOccurrence[i][j];
                double[] result = new double[numOccurrenceHere];
                for (int k=0; k < numOccurrenceHere; k++) {
                    if (logModel == 1) {
                        result[k] = Math.log10(samples[counter++]);                       
                    } else {
                        result[k] = samples[counter++];
                    }
                }
                retn[i][j] = result;
            }
        }
        return retn;
    }
    
    /**
     * Updates the random forest with new data points, some of which are censored.
     * @params tree The tree to update
     * @params newx a numnewdatapoints*numvars matrix of new data points
     * @params newy a 1*numnewdatapoints vector
     * @params newcens a 1*numnewdatapoints vector
     * @params logModel whether the passed in newy is logged.
     */
    public static void update(RandomForest forest, double[][] newx, double[] newy, boolean[] newcens, double valueForAllCens, int logModel) {
        if (forest.Trees[0].preprocessed) {
            throw new RuntimeException("Cannot update preprocessed forests.");
        }
        if (newx.length != newy.length || newx.length != newcens.length) {
            throw new RuntimeException("Argument sizes mismatch.");
        }
        
        int[][] nodes = fwd(forest, newx);
        
        if (logModel == 1 || logModel == 2) {
            double[] tmp = new double[newy.length];
            for (int i=0; i < newy.length; i++) {
                tmp[i] = Math.pow(10, newy[i]);
            }
            newy = tmp;
        }
        
        boolean hascens = false;
        for (int i=0; i < newcens.length; i++) {
            if (newcens[i]) {
                hascens = true;
                break;
            }
        }
        
        double[] samples = null;
        double[][] y_pred_all = null;
        boolean[][] cens_pred_all = null;
        double[][] weights_all = null;
        if (hascens) {
            Object[] collectedFromLeaves = collectData(forest, nodes);
            y_pred_all = (double[][])collectedFromLeaves[0];
            cens_pred_all = (boolean[][])collectedFromLeaves[1];
            weights_all = (double[][])collectedFromLeaves[2];
        }
        
        boolean[][] nodeChanged = new boolean[forest.numTrees][];
        for (int i=0; i < forest.numTrees; i++) {
            nodeChanged[i] = new boolean[forest.Trees[i].node.length];
        }
        
        for (int i = 0; i < newx.length; i++) {
            if(newcens[i]) {
                double lowerBoundForSamples = newy[i];

                double[] y_pred = y_pred_all[i];
                boolean[] cens_pred = cens_pred_all[i];
                double[] weights = weights_all[i];
                
                samples = WeibullFit.fit_dist_and_sample(y_pred, cens_pred, weights, forest.numTrees, lowerBoundForSamples, valueForAllCens);
            }
            
            for (int m=0; m < forest.numTrees; m++) {
                // For now, update every tree with the same data point.
                Regtree tree = forest.Trees[m];
                
                int node = nodes[m][i];
                nodeChanged[m][node] = true;
                
                int Nnode = tree.nodesize[node];
                
                if (tree.resultsStoredInLeaves) {
                    double[] newysub = new double[Nnode+1];
                    boolean[] newcenssub = new boolean[Nnode+1];
                    if (Nnode != 0) {
                        System.arraycopy(tree.ysub[node], 0, newysub, 0, Nnode);
                        System.arraycopy(tree.is_censored[node], 0, newcenssub, 0, Nnode);
                    }
                    
                    if (newcens[i]) {
                        if (logModel == 1) {
                            newysub[Nnode] = Math.log10(samples[m]);
                        } else {
                            newysub[Nnode] = samples[m];
                        }
                    } else {
                        newysub[Nnode] = newy[i];
                    }
                    tree.ysub[node] = newysub;
                    
                    newcenssub[Nnode] = false;
                    tree.is_censored[node] = newcenssub;
                } else {
                    tree.ysub[node][0] += newy[i]; // sum
                    tree.ysub[node][1] += newy[i]*newy[i]; // sum of squares
                }
                
                tree.nodesize[node]++;
                while(node != 0) {
                    node = tree.parent[node];
                    tree.nodesize[node]++;
                }
            }
        }
        
        for (int m=0; m < forest.numTrees; m++) {
            Regtree tree = forest.Trees[m];
            for (int i=0; i < nodeChanged[m].length; i++) {
                if (nodeChanged[m][i]) {
                    tree.recalculateStats(i);
                }
            }
        }
    }
}
