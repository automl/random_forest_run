package ca.ubc.cs.beta.models.rf;

import java.util.Random;

/**@params X a vector of instantiations of features
 * @params y a vector of size X.length of response values
 * @params cens a vector of size X.length indicating whether the corresponding response is right-censored.
 * @params catDomainSizes a vector of size X[0].length indicating the size of the domain for the corresponding categorical feature, or 0 if the feature is continuous.
 * @params condParents the conditional parents of the given variable (column index into X). Empty if no parents. NOT SORTED
 * @params condParentVals the ok values of all the conditional parents of the given variable (each element is a matrix). Indexed wrt condParents.
 * @params splitMin the minimum number of data points in each node.
 * @params ratioFeatures the percentage (0, 1] of the features to use in deciding the split at each node.
 * @params kappa the cutoff for censoring
 * @params logModel whether the data y is logged. Predictions will be done with untransformed data.
 * @params storeResponses whether to store the responses themselves in the leaves or just to store some statistic (sum, sumofsquares, and leaf size)
 * @params seed -1 means don't use a seed (ie. create a new Random but don't call setSeed). 
 * @params random will be used instead of seed if it's not null.
 */
public class RegtreeBuildParams implements java.io.Serializable {    
    public double[][] X;
    public double[] y;
    public boolean[] cens = null;
    public int[] catDomainSizes;
    public int[][] condParents = null;
    public int[][][] condParentVals = null;
    
    public int splitMin;
    public double ratioFeatures;
    public double kappa;
    public double cutoffPenaltyFactor;
    public int logModel;
    public boolean storeResponses;
    
    public int seed = -1;
    public Random random = null;
    
    public void conditionalsFromMatlab(int[] cond, int[] condParent, Object[] condParentValsObj, int nvars) {
        condParents = new int[nvars][];
        condParentVals = new int[nvars][][];
        
        for (int i=0; i < nvars; i++) {
            int count = 0;
            for (int j=0; j < cond.length; j++) {
                if (cond[j]-1 == i) {
                    count++;
                }
            }
            condParents[i] = new int[count];
            condParentVals[i] = new int[count][];
            
            count = 0;
            for (int j=0; j < cond.length; j++) {
                if (cond[j]-1 == i) {
                    condParents[i][count] = condParent[j]-1;
                    if(condParentValsObj[j] instanceof int[]) {
                        condParentVals[i][count] = (int[])condParentValsObj[j];
                    } else {
                        condParentVals[i][count] = new int[1];
                        condParentVals[i][count][0] = (Integer)condParentValsObj[j];
                    }
                    count++;
                }
            }
        }
    }
}