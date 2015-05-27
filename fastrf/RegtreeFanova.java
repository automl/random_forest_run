package ca.ubc.cs.beta.models.fastrf;

import java.util.*;

public class RegtreeFanova {
    /**
     *  @returns a map from a state to a multidimensional array of f values.
     */
    public static Map<String, Object> calculateFValues(Regtree tree, double[][] allvars, int degree) {
        if (allvars == null || allvars.length == 0) {
            throw new RuntimeException("empty X.");
        }
        if (degree > allvars.length) {
            throw new RuntimeException("Cannot analyze to a degree greater than the number of variables there are!");
        }
        
        double[][] leafInfo = getLeafInfoForANOVA(tree, allvars);
        
        int numvars = allvars.length;
        int mapsize = Utils.sum(pascalTriangleSlice(numvars, degree));
        HashMap<String, Object> map = new HashMap<String, Object>(mapsize);
        map.put("degree", degree);
        
        double f0 = 0;
        for (int i=0; i < leafInfo.length; i++) {
            double ypred = leafInfo[i][0];
            f0 += Utils.prod(leafInfo[i], 1, leafInfo[i].length) * ypred;
        }
        map.put(stateToString(numvars, new int[0]), f0);
        
        int state[] = new int[degree];
        for (int i=0; i < degree; i++) state[i] = i;
        
        do {
            calcFValue(tree, allvars, state, map);
        } while(incrState(state, degree-1, numvars));
        
        return map;
    }
    
    private static void calcFValue(Regtree tree, double[][] allvars, int[] state, Map<String, Object> map) {
        int numvars = allvars.length;
        String stateString = stateToString(numvars, state);
        if (map.containsKey(stateString)) return;
        
        double[] ypreds = Regtree.getMarginal(tree, allvars, state);
        double f0 = (Double)map.get(stateToString(numvars, new int[0]));
        
        // TODO: finish this.
    }
    
    public static Map<String, Object> fvals_to_a(Map<String, Object> fvals, int numvars) {
        //TODO
        return null;
    }
    
    public static String stateToString(int numvars, int[] state) {
        StringBuilder sb = new StringBuilder(numvars);
        for (int i=0; i < numvars; i++) {
            sb.append('0');
        }
        for (int i=0; i < state.length; i++) {
            sb.setCharAt(state[i], '1');
        }
        return sb.toString();
    }
    
    private static boolean incrState(int[] state, int position, int maxVal) {
        boolean retn = true;
        state[position]++;
        while (state[position] >= maxVal) {
            if (position == 0) return false;
            retn = incrState(state, position-1, maxVal);
            state[position] = state[position-1]+1;
        }
        return retn;
    }
    
    /** Finds the total variance of the data points in the given tree. **/
    public static double getTotalVariance(Regtree tree, double[][] allvars) {
        double[][] leafInfo = getLeafInfoForANOVA(tree, allvars);
        
        double f0 = 0;
        for (int i=0; i < leafInfo.length; i++) {
            double ypred = leafInfo[i][0];
            f0 += Utils.prod(leafInfo[i], 1, leafInfo[i].length) * ypred;
        }
        double total_variance = 0;
        for (int i=0; i < leafInfo.length; i++) {
            double ypred = leafInfo[i][0];
            total_variance += Utils.prod(leafInfo[i], 1, leafInfo[i].length) * (ypred-f0) * (ypred-f0);
        }
        return total_variance;
    }
    
    public static double getExplainedVarianceAtdegree(int degree, Map<String, Object> fvals) {
        if (degree > (Double)(fvals.get("degree"))) throw new RuntimeException("Can only calculate variance up to degree " + (Double)(fvals.get("degree")));
        // TODO
        return 0;
    }
    
    public static double getExplainedVarianceUpTodegree(int degree, Map<String, Object> fvals) {
        double retn = 0;
        for (int i=0; i < degree; i++) retn += getExplainedVarianceAtdegree(i, fvals);
        return retn;
    }
    
  
    /**
     * Propogates allvars down the tree and collect the means and weights of the leaves that each row lands in.
     * @params allvars a numvars-dimensional matrix numvars is the dimensions of the training data. Results will be returned for each row.
     * @returns a number of leaves*(1+numvars) matrix where 
          result[j][0] is the mean of leaf j, and
          result[j][i] for i>=1 is the ratio of elements of row i (also variable i) of allvars that end up in leaf j
     */
    public static double[][] getLeafInfoForANOVA(Regtree tree, double[][] allvars) {
        if (allvars == null) {
            return new double[0][];
        }
        if (allvars.length != tree.npred) {
            throw new RuntimeException("allvars.length must be equal to numvars.");
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
        double[][] retn = new double[numleaves][allvars.length+1];
        
        // Initialize output array
        int counter = 0;            
        for (int i=0; i < tree.numNodes; i++) {
            if (tree.var[i] == 0) {
                // Store mean of this leaf
                retn[counter++][0] = (logModel == 1 ? Math.pow(10, tree.nodepred[i]) : tree.nodepred[i]);
            }
        }
        
        LinkedList<Integer> queue = new LinkedList<Integer>();
        for (int i=0; i < allvars.length; i++) {
            int nextvar = i+1;
            int numValues = allvars[i].length;
            
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
                                thisnode = (allvars[i][j] <= cutoff ? left_kid : right_kid);
                            } else { // categorical
                                int x = (int)allvars[i][j];
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
    
    // Doesn't check for overflow. Your fault if you pass huge inputs.
    public static int[] pascalTriangleSlice(int N, int k) {
        int[] b = new int[N + 1];
        b[0] = 1;
        for (int i = 1; i <= N; i++) {
            b[i] = 1;
            for (int j = i-1; j > 0; j--) {
                b[j] += b[j-1];
            }
        }
        int[] retn = new int[k];
        for (int i = 0; i < k; i++) {
            retn[i] = b[i];
        }
        return retn;
    }
}