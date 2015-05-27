package ca.ubc.cs.beta.models.rf;

import java.util.*;

public class RegtreeFitNoCensFix {    
    private static int seed;
    
    //*
    private static final int RAND_MAX = Integer.MAX_VALUE - 1;
    private static int rand(Random r) {
        int retn = r.nextInt(Integer.MAX_VALUE);
        return retn;
    }
    /*/
    private static final int RAND_MAX = 2147483646;
    private static int rand(Random r) {
        long a=22695477;
        int c=1;
        return seed = (int)((a*seed+c)%(RAND_MAX+1));
    }
    //*/
    
    /**
     * Fits a regression tree.
     * @params dataIdxs indices into params.X,y,cens of data points to use for building this tree.
     * @params params see RegtreeBuildParams
     */
    public static Regtree fit(int[] dataIdxs, RegtreeBuildParams params) {
        int N = dataIdxs.length;
        assert(N > 0);
        int nvars = params.X[0].length;
        
        double[][] X = new double[N][nvars];
        double[] y = new double[N];
        boolean[] cens = new boolean[N];

        for (int i=0; i < N; i++) {
            int idx = dataIdxs[i];
            X[i] = params.X[idx];
            y[i] = params.y[idx];
        }
        
        if (params.cens != null) {
            for (int i=0; i < N; i++) {
                int idx = dataIdxs[i];
                cens[i] = params.cens[idx];
            }
        }
        
        int[] catDomainSizes = params.catDomainSizes;
        
        int[][] condParents = params.condParents;
        int[][][] condParentVals = params.condParentVals; 
        
        double ratioFeatures = params.ratioFeatures;
        double kappa = params.kappa;
        int splitMin = params.splitMin;

        Random r = params.random;
        if (r == null) {
            r = new Random();
            if (params.seed != -1) {
                r.setSeed(params.seed);
            }
        }
        
        if (catDomainSizes.length != nvars) {
            throw new RuntimeException("catDomainSizes must be of the same length as size(X, 2)");
        }
        
        //=== Build the tree ===      
        // which data points are assigned to the node
        Vector<Vector<Integer>> assignedtonode = new Vector<Vector<Integer>>(2*N);
        assignedtonode.add(new Vector<Integer>(N));
        for (int i=0; i < N; i++) {
            assignedtonode.get(0).add(i);
        }
        
        int[] noderows = new int[N];
        double[] ynode = new double[N];
        boolean[] censnode = new boolean[N];
        double[] xvars = new double[N];
        double[] x = new double[N];
        int[] rows = new int[N];
        double[] ycum = new double[N];
        int[] randomPermutation = new int[nvars];
        double resuberr = 0;
        
        int[] nodenumber = new int[2*N];
        nodenumber[0] = 0;
        
        int[] nodesize = new int[2*N];
        int[] cutvar = new int[2*N];
        double[] cutpoint = new double[2*N];
        int[] leftchildren = new int[2*N];
        int[] rightchildren = new int[2*N];
        int[] parent = new int[2*N];
        double[][] ysub = new double[2*N][];
        boolean[][] censsub = new boolean[2*N][];
        
        int[] period = new int[N];
        double[] period_time = new double[N+1];
        int[] N_all = new int[N];
        int[] C_all = new int[N];
        int[] O_all = new int[N];
        
        double[] ynode_idx = new double[N];
        boolean[] censnode_idx = new boolean[N];
        int[] period_idx = new int[N];
        
        int[] xleft = new int[N];
        int[] xright = new int[N];
        int[] numLeftPointer = new int[1];
        int[] numRightPointer = new int[1];
        double[] cutvalPointer = new double[1];
        double[] critvalPointer = new double[1];
        
        int[] bestleft = new int[N];
        int[] bestright = new int[N];
        
        int[] leftside = new int[N];
        int[] rightside = new int[N];
        
         // number of categorical splits
        int ncatsplit = 0;
        int[][] catsplit = new int[2*N][];
        
        boolean hascens = false;
        for (int i=0; i < N; i++) {
            if (cens[i]) {
                hascens = true;
                break;
			}
		}

        if (!hascens) {
            //=== Initialize variables for Rcritval_cat/cont ===//
            t = new int[N];
            B = new double[N];
            diff_t = new int[N];
            catmeans = new double[N];
            Ysplit1 = new double[N-1];
            n1 = new int[N-1];
            allx = new double[N];
            mu1 = new double[N-1];
            mu2 = new double[N-1];
            maxlocs = new int[N-1];
            ssx = new double[N-1];
        } else {
            //=== Initialize variables for Rcritval_cat/cont_logrank ===//
            y_in = new double[N];
            c_in = new double[N];
            km_mean = new double[N];
            N_1 = new int[N];
            O_1 = new int[N];
            N_add = new int[N];
            logrank_stat = new double[N-1];
            E = new double[N];
            V = new double[N];
            allx = new double[N];
            maxlocs = new int[N-1];
            //=== Initialize variables for kaplan-meier mean ===//
            y_km = new double[N];
            c_km = new double[N];
            rows_km = new int[N];
        }
        sorder = new int[N];

        int[] stack = new int[N];
        stack[0] = 0;
        int stacktop = 0;
        int numNodes = 1;
        while (stacktop >= 0) {
            int tnode = stack[stacktop--];
            // the number of training data points inside this node
            int Nnode = assignedtonode.get(tnode).size();
            if (Nnode == 0) {
                throw new RuntimeException("Nnode is 0!!");
            }
            
            int numUncensNode = 0;
            // average of responses from training data in this node
            double ysum = 0, ysumOfSq = 0;
            for (int i=0; i < Nnode; i++) {
                int idx = assignedtonode.get(tnode).get(i);
                // which rows of the data are in this node
                noderows[i] = idx;
                // response values in this node
                ynode[i] = y[idx];
                ysum += y[idx];
                ysumOfSq += y[idx]*y[idx];
                
                censnode[i] = cens[idx];
                if (!cens[idx]) {
                    numUncensNode++;
                }
            }
            double ybar = ysum / Nnode;
            
            double mincost = (Nnode == 1 ? 0 : (ysumOfSq - ysum*ysum/Nnode) / (Nnode-1));
            boolean impure = (mincost > 1e-20 * resuberr);
            
            if (tnode == 0) {
                resuberr = mincost;
            }
            
            nodesize[tnode] = Nnode;
            cutvar[tnode] = 0;
            cutpoint[tnode] = 0;
            leftchildren[tnode] = 0;
            rightchildren[tnode] = 0;
            
            // Critical value associated with current best found split var and point
            double bestcrit = -1e12;
            int numBestLeft = 0, numBestRight = 0;
            
            // Consider splitting this node
            if (impure && numUncensNode >= splitMin) { // split only impure nodes with more than a threshold of uncensored values
                int bestvar = -1;
                double bestcut = 0;
                
                int nvarshere = 0;
                
                if (condParents == null) {
                    nvarshere = nvars;
                    for (int i=0; i < nvars; i++) {
                        randomPermutation[i] = i;
                    }
                } else {
                    for (int i=0; i < nvars; i++) {
                        boolean isenabled = true;
                        if (condParents[i] != null) {
                            for (int j=0; j < condParents[i].length; j++) {
                                int[] compatibleValues = getCompatibleValues(tnode, condParents[i][j], N, parent, cutvar, cutpoint, leftchildren, rightchildren, catsplit, catDomainSizes);

                                for (int k=0; k < compatibleValues.length; k++) {
                                    boolean isokvalue = false;
                                    for (int l=0; l < condParentVals[i][j].length; l++) {
                                        if (compatibleValues[k] == condParentVals[i][j][l]) {
                                            isokvalue = true;
                                            break;
                                        }
                                    }
                                    if (!isokvalue) {
                                        isenabled = false;
                                        break;
                                    }
                                }
                                if (!isenabled) break;
                            }
                        }
                        if (isenabled) {
                            randomPermutation[nvarshere++] = i;
                        }
                    }
                }
                shuffle(randomPermutation, nvarshere, r);
                
                //=== Begin special logrank code ===
                // First, compute period for uncensored data. (Censored data with same y value could be before or after the first uncensored one in the ordering, thus we need 2 passes.)                 
                int num_periods = 0;
                if (hascens) {
                    rankSort(ynode, Nnode);
                    
                    for (int j = 0; j < Nnode; j++) {
                        period[j] = 0;
                    }
                    
                    double last_time = -1e10;
                    for (int j=0; j < Nnode; j++) {
                        int i = sorder[j];
                        if (!censnode[i]) {
                            if (ynode[i] > last_time + 1e-6) {
                                num_periods++;
                                period_time[num_periods] = ynode[i];
                                last_time = ynode[i];
                            }
                            period[i] = num_periods;
                        }
                    }
                    
                    // Then, fill in period for censored data.
                    int curr_period = 0;
                    double this_time = 0;
                    for (int i=0; i < Nnode; i++) {
                        if (censnode[i]) {
                            if (curr_period == num_periods) {
                                this_time = 1e9;
                            } else {
                                this_time = period_time[curr_period+1];
                            }
                            while (ynode[i] > this_time - 1e-6) {
                                curr_period++;
                                if (curr_period == num_periods) {
                                    this_time = 1e9;
                                } else {
                                    this_time = period_time[curr_period+1];
                                }
                            }
                            period[i] = curr_period;
                        }
                    }
                    
                    // Compute initial logrank counters when all data in 2nd group.
                    for (int p=0; p < num_periods; p++) {
                        O_all[p] = 0;
                        C_all[p] = 0;
                    }
                    
                    // Go through data points, and collect the events.
                    for (int i=0; i < Nnode; i++) {
                        int p = period[i];
                        if (!censnode[i]) {
                            O_all[p-1]++;
                        } else {
                            if (p > 0) {
                                C_all[p-1]++;
                            }
                        }
                        // else the point counts as if it hadn't even happened
                    }
                    
                    // Go through periods, and collect N.
                    N_all[0] = Nnode;
                    for (int p = 1; p < num_periods; p++) {
                        N_all[p] = N_all[p-1] - O_all[p-1] - C_all[p-1];
                    }
                }
                //=== End of this part of special logrank code. ===
                
                //=== Try splitting each variable at every split point and pick best split ===
                for (int i=0; i < nvarshere; i++) {
                    int nextvar = randomPermutation[i];
                    boolean xcat = (catDomainSizes[nextvar] != 0);
                    
                    for (int j=0; j < Nnode; j++) {
                        // xvars[j] contains the value of nextvar of data sample[j] in the node
                        xvars[j] = X[noderows[j]][nextvar];
                    }
                    
                    rankSort(xvars, Nnode);
                    
                    for (int j=0; j < Nnode; j++) {
                        x[j] = xvars[sorder[j]];
                    }
                    
                    //=== Determine if there's anything to split along this variable. 
                    if (x[Nnode-1] - x[0] < 1e-10) {
                        continue;
                    }
                    
                    //These are the rows we can possibly split on
                    //WATCH OUT: rows holds the indices as original in Matlab, not C style
                    int numrows = 0;
                    for (int j=0; j < Nnode-1; j++) {
                        if (x[j] + 1e-10 < x[j+1]) {
                            rows[numrows++] = j+1; // the +1 here is to make this compatible with calling Rcritval_cat & cont from Matlab directly.
                        }
                    }
                    if (numrows == 0) continue;
                    
                    //=== Do the core work: get the best split of the variable and its quality.
                    if (hascens) {
                        for (int j=0; j < Nnode; j++) {
                            ynode_idx[j] = ynode[sorder[j]];
                            censnode_idx[j] = censnode[sorder[j]];
                            period_idx[j] = period[sorder[j]];
                        }
                        if (xcat) {
                            Rcritval_cat_logrank(x, Nnode, ynode_idx, censnode_idx, rows, numrows, period_idx, num_periods, N_all, O_all, kappa, critvalPointer, xleft, xright, numLeftPointer, numRightPointer, r);
                        } else {
                            Rcritval_cont_logrank(x, Nnode, censnode_idx, rows, numrows, period_idx, num_periods, N_all, O_all, critvalPointer, cutvalPointer, r);                    
                        }
                    } else {
                        ycum[0] = ynode[sorder[0]] - ybar;
                        for (int j=1; j < Nnode; j++) {
                            ycum[j] = ycum[j-1] + ynode[sorder[j]] - ybar; // centered response cum. sum
                        }
                        if (xcat) {
                            Rcritval_cat(x, ycum, rows, Nnode, numrows, critvalPointer, xleft, xright, numLeftPointer, numRightPointer, r);
                        } else {
                            Rcritval_cont(x, ycum, rows, Nnode, numrows, critvalPointer, cutvalPointer, r);
                        }
                    }
                    //=== Change best split if this one is best so far.
                    if (critvalPointer[0] > bestcrit + 1e-10) {
                        bestcrit = critvalPointer[0];
                        bestvar = nextvar;
                        if (xcat) {
                            numBestLeft = numLeftPointer[0];
                            numBestRight = numRightPointer[0];
                            for (int j=0; j < numBestLeft; j++) {
                                bestleft[j] = xleft[j];
                            }
                            for (int j=0; j < numBestRight; j++) {
                                bestright[j] = xright[j];
                            }
                        } else {
                            bestcut = cutvalPointer[0];
                        }
                    }
                    
                    if (i >= Math.max(1, (int)(Math.floor(ratioFeatures*nvarshere))) - 1 && bestcrit > -1e11) {
                        break;
                    }
                }
                
                //=== Best split point has been found. Split this node using the best rule found.
            
                // Number of data points in the children nodes
                int nleft = 0, nright = 0;
                
                if (bestvar == -1) {
                    // Terminal node
                } else {
                    for (int i=0; i < Nnode; i++) {
                        x[i] = X[noderows[i]][bestvar];
                    }
                    
                    if (catDomainSizes[bestvar]!=0) {
                        cutvar[tnode] = -(bestvar+1); // negative indicates cat. var. split
                        cutpoint[tnode] = ncatsplit; // index into catsplit cell array. 0-indexed!!!
                  
                        /* 1: To get all compatible values, walk up the tree, looking
                         * for a split on the same parameter. If none is found
                         * take the initial domain of that parameter.
                         */
                        int[] compatibleValues = getCompatibleValues(tnode, bestvar, N, parent, cutvar, cutpoint, leftchildren, rightchildren, catsplit, catDomainSizes);
                        
                        int[] missing_values_for_left = new int[compatibleValues.length];
                        int[] missing_values_for_right = new int[compatibleValues.length];
                        
                        // 2: For each compatible but missing value choose a side u.a.r.
                        int num_missing_to_left = 0, num_missing_to_right = 0;
                        for (int i=0; i < compatibleValues.length; i++) {
                            int nextValue = compatibleValues[i];
                            int j;
                            for (j=0; j < numBestLeft; j++) {
                                if (nextValue == bestleft[j]) break;
                            }
                            if (j == numBestLeft) {
                                for (j=0; j < numBestRight; j++) {
                                    if (nextValue == bestright[j]) break;
                                }
                                if (j == numBestRight) {
                                    // Missing but compatible value: choose side u.a.r.
                                    if (rand(r) % 2 == 0) {
                                        missing_values_for_left[num_missing_to_left++] = nextValue;
                                    } else {
                                        missing_values_for_right[num_missing_to_right++] = nextValue;
                                    }
                                }
                            }
                        }
                        
                        // 3: Merge the determined and the randomly assigned missing values
                        for (int i=num_missing_to_left; i<num_missing_to_left+numBestLeft; i++) {
                            missing_values_for_left[i] = bestleft[i-num_missing_to_left];
                        }
                        sort(missing_values_for_left, num_missing_to_left+numBestLeft);
                        
                        for (int i=num_missing_to_right; i<num_missing_to_right+numBestRight; i++) {
                            missing_values_for_right[i] = bestright[i-num_missing_to_right];
                        }
                        sort(missing_values_for_right, num_missing_to_right+numBestRight);
                        
                        // 4: Store the information
                        catsplit[ncatsplit] = new int[num_missing_to_left + numBestLeft];
                        for (int i=0; i<num_missing_to_left+numBestLeft; i++) {
                            catsplit[ncatsplit][i] = missing_values_for_left[i];
                        }
                        catsplit[ncatsplit+N] = new int[num_missing_to_right + numBestRight];
                        for (int i=0; i<num_missing_to_right+numBestRight; i++) {
                            catsplit[ncatsplit+N][i] = missing_values_for_right[i];
                        }
                        ncatsplit++;
                        
                        for (int i=0; i < Nnode; i++) {
                            boolean onleft = false;
                            for (int j=0; j < numBestLeft; j++) {
                                if ((int)(Math.floor(x[i]+0.5)) == bestleft[j]) {
                                    onleft = true;
                                    break;
                                }
                            }
                            if (onleft) {
                                leftside[nleft++] = i;
                            } else {
                                rightside[nright++] = i;
                            }
                        }
                    } else { // splitting on cont. var.
                        cutvar[tnode] = bestvar + 1;
                        cutpoint[tnode] = bestcut;
                        
                        for (int i=0; i < Nnode; i++) {
                            if (x[i] <= bestcut) {
                                leftside[nleft++] = i;
                            } else {
                                rightside[nright++] = i;
                            }
                        }
                    }
                    if (nleft == 0 || nright == 0) {
                        throw new RuntimeException("Empty side after splitting!");
                    }
                    
                    leftchildren[tnode] = numNodes;
                    rightchildren[tnode] = numNodes+1;
                    
                    parent[numNodes] = tnode;
                    parent[numNodes+1] = tnode;
                    
                    nodenumber[numNodes]  = numNodes;
                    nodenumber[numNodes+1] = numNodes+1;
                    
                    assignedtonode.add(new Vector<Integer>(nleft));
                    Vector<Integer> leftchildnode = assignedtonode.get(numNodes);
                    for (int i=0; i < nleft; i++) {
                        leftchildnode.add(noderows[leftside[i]]);
                    }
                    assignedtonode.add(new Vector<Integer>(nright));
                    Vector<Integer> rightchildnode = assignedtonode.get(numNodes+1);
                    for (int i=0; i < nright; i++) {
                        rightchildnode.add(noderows[rightside[i]]);
                    }
                    
                    stack[++stacktop] = numNodes;
                    stack[++stacktop] = numNodes+1;
                    numNodes += 2;
                }
            }
            if (leftchildren[tnode] == 0) {
                // Leaf => store results falling here (don't store them everywhere to avoid O(N^2) storage)
                // Save *runtimes*, not losses. 
                ysub[tnode] = new double[Nnode];
                censsub[tnode] = new boolean[Nnode];
                System.arraycopy(ynode, 0, ysub[tnode], 0, Nnode);
                System.arraycopy(censnode, 0, censsub[tnode], 0, Nnode);
            }
            tnode++;
        }
        
        Regtree tree = new Regtree(numNodes, ncatsplit, params.storeResponses, params.logModel);
        
        System.arraycopy(nodenumber, 0, tree.node, 0, numNodes);
        System.arraycopy(parent, 0, tree.parent, 0, numNodes);
        System.arraycopy(cutvar, 0, tree.var, 0, numNodes);
        System.arraycopy(cutpoint, 0, tree.cut, 0, numNodes);
        System.arraycopy(nodesize, 0, tree.nodesize, 0, numNodes);
        tree.npred = nvars;
        
        int nextnode=-1;
        for (int i=0; i < ncatsplit; i++) {
            while(cutvar[++nextnode] >= 0);
            int[] tmp = new int[catDomainSizes[-cutvar[nextnode]-1]];
            Arrays.fill(tmp, -1);
            int[] cs = catsplit[(int)cutpoint[nextnode]];
            for (int j=0; j < cs.length; j++) {
                tmp[cs[j]-1] = 0;
            }
            cs = catsplit[(int)cutpoint[nextnode] + N];
            for (int j=0; j < cs.length; j++) {
                tmp[cs[j]-1] = 1;
            }
            tree.catsplit[(int)cutpoint[nextnode]] = tmp;
        }
        
        for (int i=0; i < numNodes; i++) {
            tree.children[i][0] = leftchildren[i];
            tree.children[i][1] = rightchildren[i];
            
            int Nnode = leftchildren[i] == 0 ? nodesize[i] : 0;
            
            if (params.logModel == 1 || params.logModel == 2) {
                for (int j=0; j < Nnode; j++) {
                    ysub[i][j] = Math.pow(10, ysub[i][j]);
                }
            }
            
            if (Nnode != 0) {
                if (params.storeResponses) {
                    tree.ysub[i] = new double[Nnode];
                    tree.is_censored[i] = new boolean[Nnode]; 
                    System.arraycopy(ysub[i], 0, tree.ysub[i], 0, Nnode);
                    System.arraycopy(censsub[i], 0, tree.is_censored[i], 0, Nnode);
                } else {
                    double sum = 0, sumOfSq = 0, sumOfLog = 0, sumOfLogSq = 0;
                    for (int j=0; j < Nnode; j++) {
                        double next = ysub[i][j];
                        sum += next;
                        sumOfSq += next * next;
                    }
                    tree.ysub[i][0] = sum;
                    tree.ysub[i][1] = sumOfSq;
                }
            }
        }
        tree.recalculateStats();
        
        t = null;
        B = null;
        diff_t = null;
        catmeans = null;
        Ysplit1 = null;
        n1 = null;
        allx = null;
        mu1 = null;
        mu2 = null;
        maxlocs = null;
        ssx = null;
        y_in = null;
        c_in = null;
        km_mean = null;
        N_1 = null;
        O_1 = null;
        N_add = null;
        logrank_stat = null;
        E = null;
        V = null;
        allx = null;
        maxlocs = null;
        y_km = null;
        c_km = null;
        rows_km = null;
        sorder = null;
        
        return tree;
    }
    
    private static int[] getCompatibleValues(int currnode, int var, int N, int[] parent, int[] cutvar, double[] cutpoint, int[] leftchildren, int[] rightchildren, int[][] catsplit, int[] catDomainSizes) {
        int[] compatibleValues = null;
        while (currnode > 0) {
            int parent_node = parent[currnode];
            if (-cutvar[parent_node]-1 == var) {
                int catsplit_index = (int)cutpoint[parent_node];

                if (leftchildren[parent_node] == currnode) {
                    compatibleValues = catsplit[catsplit_index];
                } else if (rightchildren[parent_node] == currnode) {
                    compatibleValues = catsplit[catsplit_index+N];
                } else {
                    throw new RuntimeException("currnode must be either left or right child of its parent.");
                }
                break;
            }
            currnode = parent_node;
        }
        if (currnode == 0) {
            compatibleValues = new int[catDomainSizes[var]];
            for (int i=0; i < compatibleValues.length; i++) compatibleValues[i] = i+1;
        }
        return compatibleValues;
    }
        
    //=== Variables for Rcritval_cat/cont ===//
    private static int[] t;
    private static double[] B;
    private static int[] diff_t;
    private static double[] catmeans;
    private static double[] Ysplit1;
    private static int[] n1;
    private static double[] allx;
    private static double[] mu1;
    private static double[] mu2;
    private static int[] maxlocs;
    private static double[] ssx;
    //=== END Variables for Rcritval_cat/cont ===//
    private static void Rcritval_cat(double[] x, double[] Ycum, int[] rows, int nX, int nrows, double[] critval_res, int[] xleft, int[] xright, int[] numLeftPointer, int[] numRightPointer, Random r) {
        int n = nrows + 1;

        //=== First get all possible split points. 
        //=== t are the changepoints + the last index
        for (int i=0; i<n-1; i++) {
            t[i] = rows[i]-1;
        }
        t[n-1] = nX-1;

        //=== B contains the category sums.     Matlab: B = Ycum(t,:); B(2:end,:) = B(2:end,:) - B(1:end-1,:);
        B[0] = Ycum[t[0]];
        for (int i=1; i<n; i++) {
            B[i] = Ycum[t[i]] - Ycum[t[i-1]];
        }

        //=== diff_t are the number of points in a category
        diff_t[0] = t[0]+1;
        for (int i=1; i<n; i++) {
            diff_t[i] = t[i]-t[i-1]; 
        }

        //=== catmeans contains the means for the categories.
        for (int i=0; i<n; i++) {
            catmeans[i] = B[i] / Math.max(1, diff_t[i]);
        }

        //=== Sort categories by mean response.
        rankSort(catmeans, n);
        
        //=== Ysplit1 is Ycum[rows sorted by mean response]
        //=== n1(i) is the number of points going left when splitting using the ith subset*/
        Arrays.fill(Ysplit1, 0, n-1, 0);
        Arrays.fill(n1, 0, n-1, 0);
        for (int i=0; i<n-1; i++) {
            for (int j=0; j<=i; j++) {
                Ysplit1[i] = Ysplit1[i] + B[sorder[j]];
                n1[i] = n1[i] + diff_t[sorder[j]];
            }
        }

        //=== Take one x value from each unique set
        for (int i=0; i<n; i++) {
            allx[i] = x[t[i]];
        }
        
        //=== Get left/right means (same for cat/cont)       
        for (int i=0; i<n-1; i++) {
            mu1[i] = Ysplit1[i] / n1[i];
        }
        for (int i=0; i<n-1; i++) {
            mu2[i] = (Ycum[nX-1] - Ysplit1[i]) / (nX - n1[i]);
        }

        //=== Get best split, val and location.
        int maxnumlocs = 0;
        critval_res[0] = -1e13;
        for (int i=0; i<n-1; i++) {
            ssx[i] = n1[i]*mu1[i]*mu1[i] + (nX-n1[i])*mu2[i]*mu2[i];
            if (ssx[i] > critval_res[0]-1e-10) {
                if (ssx[i] > critval_res[0]+1e-10) {
                    critval_res[0] = ssx[i];
                    maxnumlocs = 0;
                }
                maxlocs[maxnumlocs] = i;
                maxnumlocs++;
            }
        }

        int maxloc = maxlocs[rand(r) % maxnumlocs];
        numLeftPointer[0] = maxloc+1;
        numRightPointer[0] = n-numLeftPointer[0];

        // Now we can fill the result arrays xleft and xright as usual.
        for (int i=0; i<maxloc+1; i++) {
            xleft[i] = (int) allx[sorder[i]];
        }
        sort(xleft, maxloc+1);
        
        for (int i=maxloc+1; i<n; i++) {
            xright[i-(maxloc+1)] = (int) allx[sorder[i]];
        }
        sort(xright, n-(maxloc+1));
    }

    private static void Rcritval_cont(double[] x, double[] Ycum, int[] rows, int nX, int nrows, double[] critval_res, double[] cutval_res, Random r) {
        //=== Ysplit1 is Ycum(rows sorted by mean response)
        for (int i=0; i<nrows; i++) {
            Ysplit1[i] = Ycum[rows[i]-1];
        }

        //=== Get left/right means (same for cat/cont)
        for (int i=0; i<nrows; i++) {
            mu1[i] = Ysplit1[i] / rows[i];
        }
        for (int i=0; i<nrows; i++) {
            mu2[i] = (Ycum[nX-1] - Ysplit1[i]) / (nX - rows[i]);
        }

        //=== Get best split, val and location.
        int maxnumlocs = 0;
        critval_res[0] = -1e13;
        for (int i=0; i<nrows; i++) {
            ssx[i] = rows[i]*mu1[i]*mu1[i] + (nX-rows[i])*mu2[i]*mu2[i];
            if (ssx[i] > critval_res[0]-1e-10) {
                if (ssx[i] > critval_res[0]+1e-10) {
                    critval_res[0] = ssx[i];
                    maxnumlocs = 0;
                }
                maxlocs[maxnumlocs] = i;
                maxnumlocs = maxnumlocs + 1;
            }
        }
        int maxloc = maxlocs[rand(r) % maxnumlocs];

        //=== Get cutval.
        int cutloc = rows[maxloc]-1;
        double u = rand(r) * 1.0 / RAND_MAX;

        // if points are close the just take average. If points are farthur sample randomly from lerp
        if (x[cutloc+1] - x[cutloc] < 1.9*1e-6) {
            cutval_res[0] = (x[cutloc] + x[cutloc+1])/2;
        } else {
            cutval_res[0] = ((1-u)*(x[cutloc]+1e-6) + u*(x[cutloc+1]-1e-6));
            if (cutval_res[0] < x[cutloc]+1e-8 || cutval_res[0] > x[cutloc+1]-1e-8) {
                throw new RuntimeException("random splitpoint has to lie in between the upper and lower limit");
            }
        }
    }


    /* Kaplan-Meier estimator. If the highest value is censored K-L is
     * undefined for values above that.
     * For computing the mean we need an estimate, though. Fortunately, we
     * have an upper bound, and we assume that there are no deaths until
     * that upper bound.
     */
    private static double[] y_km;
    private static double[] c_km;
    private static int[] rows_km;
    private static double kaplan_meier_mean(int N, int M, double[] y_in, double[] c_in, double upper_bound) {
        int N_i = N+M;
        if (N_i==0) {
            throw new RuntimeException("Kaplan-Meier estimator undefined for empty population.");
        }
        
        for (int i=0; i<N; i++) y_km[i] = y_in[i];
        for (int i=0; i<M; i++) c_km[i] = c_in[i];
        sort(y_km, N);
        sort(c_km, M);

        double km_mean = 0;
        
        if (N>0) {
            int num_rows = 0;
            for (int i=0; i<N-1; i++) {
                if (y_km[i] + 1e-10 < y_km[i+1]) {
                    rows_km[num_rows++] = i;
                }
            }
            rows_km[num_rows++] = N-1;
            
            double prod = 1;
			int c_idx = 0;
            for (int i=0; i<num_rows; i++) {
                double t_i = y_km[rows_km[i]];
                while (c_idx < M && c_km[c_idx] < t_i) {
                    N_i--;
                    c_idx++;
                }
                int d_i;
                if (i==0) {
                    d_i = rows_km[i]+1;
                    km_mean += prod * y_km[rows_km[i]];
                } else {
                    d_i = rows_km[i] - rows_km[i-1];
                    km_mean += prod * (y_km[rows_km[i]] - y_km[rows_km[i-1]]);
                }
                prod *= (N_i-d_i+0.0)/(N_i+0.0);
                N_i -= d_i;
            }
            /* Deal with remaining censored values (if the highest value is
             * uncensored, then prod=0, so nothing happens then)
             */
            /*km_mean += prod * (upper_bound-y[N-1]);*/
            km_mean += prod * (upper_bound-y_km[N-1])/2;
        } else {
            km_mean = upper_bound;
        }        
        return km_mean;
    }

    private static double logrank_statistic(int num_periods, int[] N, int[] O, int[] N_1, int[] O_1, double[] E, double[] V) {
        /* Compute logrank statistic for the specified "numbers at risk" (N),
         * observed events (O), "numbers at risk" in group 1 (N_1), and observed
         * events in group 1 (O_1)
         *
         * If N(i) <= 1, we don't include it in the sum (undefined variance, 0/0)
         */

        double sum_V=0, numerator=0;
        for (int p=0; p<num_periods; p++) {
            E[p] = O[p] * ((N_1[p]+0.0)/N[p]);
            if (N[p]<= 1) {
                V[p] = 0;        
            } else {
                V[p] = E[p] * (1 - ((N_1[p]+0.0)/N[p])) * (N[p]-O[p]) / (N[p]-1.0);
                sum_V += V[p];
                numerator += (O_1[p]-E[p]);
            }
        }

        double denominator = Math.sqrt(sum_V);
        
        if (Math.abs(denominator) < 1e-6) {
            if (Math.abs(numerator) < 1e-6) {
                return 0;
            } else {
                throw new RuntimeException("Division by zero in function logrank_statistic.");
            }
        }
        
        return numerator/denominator;
    }


    //=== Variables for Rcritval_cat/cont_logrank ===//
    private static double[] y_in;
    private static double[] c_in;
    private static double[] km_mean;
    private static int[] N_1;
    private static int[] O_1;
    private static int[] N_add;
    private static double[] logrank_stat;
    private static double[] E;
    private static double[] V;
    //double[] allx; // declared in Rcritval_cat/cont
    //int[] maxlocs; // declared in Rcritval_cat/cont
    //=== END Variables for Rcritval_cat/cont_logrank ===//
    private static void Rcritval_cat_logrank(double[] x, int nX, double[] y, boolean[] cens, int[] rows, int nRows, int[] period, int num_periods, int[] N_all, int[] O_all, double kappa, double[] critval_res, int[] xleft, int[] xright, int[] numLeftPointer, int[] numRightPointer, Random r) {
        // Get Kaplan-Meier estimates of the mean & median of each group.
        for (int i=0; i<=nRows; i++) {
            int low = (i==0 ? 0 : rows[i-1]);
            int high = (i==nRows ? nX : rows[i]);

            int numNonCens = 0, numCens = 0;
            for (int j=low; j<high; j++) {
                if (cens[j]) {
                    c_in[numCens++] = y[j];
                } else {
                   y_in[numNonCens++] = y[j];
                }
            }
            km_mean[i] = kaplan_meier_mean(numNonCens, numCens, y_in, c_in, kappa);
        }

        //=== Sort categories by km_mean.
        rankSort(km_mean, nRows+1);

        for (int i=0; i<num_periods; i++) {
            N_1[i] = 0;
            O_1[i] = 0;
        }
        
        //=== For adding one group at a time, compute resulting logrank statistic.
        for (int s=0; s<nRows; s++) {
            int group_no = sorder[s];

            // Get indices of group_no'th group.
            int low = (group_no == 0 ? 0 : rows[group_no-1]);
            int high = (group_no == nRows ? nX : rows[group_no]);

            int maxperiod = -1;
            for (int i=low; i<high; i++) {
                maxperiod = Math.max(maxperiod, period[i]);
            }

            for (int p=0; p<num_periods; p++) {
                N_add[p] = 0;
            }
            
            // Collect observations in O_1, and helper N_add to collect numbers at risk, N_1
            for (int i=low; i<high; i++) {
                int p = period[i]-1;
                if (p >= 0) {
                    if (!cens[i]) {
                        O_1[p]++;
                    }
                    N_add[p]++;
                }
            }

            int N_scalar = 0;
            for (int p=num_periods-1; p >= 0; p--) {
                N_scalar += N_add[p];
                N_1[p] += N_scalar;
                if (N_1[p] < O_1[p]) {
                    throw new RuntimeException("O_1[p] > N_1[p] -- should be impossible");
                }
            }
            logrank_stat[s] = logrank_statistic(num_periods, N_all, O_all, N_1, O_1, E, V);
        }

        critval_res[0] = -1;
        int maxnumlocs = 0;
        for (int i=0; i<nRows; i++) {
            if (Math.abs(logrank_stat[i]) > critval_res[0]-1e-6) {
                if (Math.abs(logrank_stat[i]) > critval_res[0]+1e-6) {
                    critval_res[0] = Math.abs(logrank_stat[i]);
                    maxnumlocs = 0;
                }
                maxlocs[maxnumlocs] = i;
                maxnumlocs = maxnumlocs + 1;
            }
        }

        int maxloc = maxlocs[rand(r) % maxnumlocs];
        
        // Take one x value from each unique set
        for (int i=0; i<nRows; i++) {
            allx[i] = x[rows[i]-1];
        }
        allx[nRows] = x[nX-1];

        numLeftPointer[0] = maxloc+1;
        numRightPointer[0] = nRows+1-maxloc-1;

        // Now we can fill the result arrays xleft and xright as usual.
        for (int i=0; i<maxloc+1; i++) {
            xleft[i] = (int) allx[sorder[i]];
        }
        sort(xleft, maxloc+1);
        for (int i=maxloc+1; i<nRows+1; i++) {
            xright[i-(maxloc+1)] = (int) allx[sorder[i]];
        }
        sort(xright, nRows+1-(maxloc+1));
    }

    private static void Rcritval_cont_logrank(double[] x, int nX, boolean[] cens, int[] rows, int nRows, int[] period, int num_periods, int[] N_all, int[] O_all, double[] critval_res, double[] cutval_res, Random r) {
        for (int i=0; i<num_periods; i++) {
            N_1[i] = 0;
            O_1[i] = 0;
        }

        //=== For adding one group at a time, compute resulting logrank statistic.
        for (int group_no=0; group_no<nRows; group_no++) {
            // Get indices of group_no'th group.
            int low = (group_no == 0 ? 0 : rows[group_no-1]);
            int high = (group_no == nRows ? nX : rows[group_no]);

            int maxperiod = -1;
            for (int i=low; i<high; i++) {
                maxperiod = Math.max(maxperiod, period[i]);
            }        
            
            for (int p=0; p<maxperiod; p++) {
                N_add[p] = 0;
            }

            // Collect observations in O_1, and helper N_add to collect numbers at risk, N_1
            for (int i=low; i<high; i++) {
                int p = period[i]-1;
                if (p >= 0) {
                    if (!cens[i]) {
                        O_1[p]++;
                    }
                    N_add[p]++;
                }
            }

            int N_scalar=0;
            for (int p=maxperiod-1; p >= 0; p--) {
                N_scalar += N_add[p];
                N_1[p] += N_scalar;
            }
            logrank_stat[group_no] = logrank_statistic(num_periods, N_all, O_all, N_1, O_1, E, V);
        }

        critval_res[0] = -1;
        int maxnumlocs = 0;
        for (int i=0; i<nRows; i++) {
            if (Math.abs(logrank_stat[i]) > critval_res[0]-1e-10) {
                if (Math.abs(logrank_stat[i]) > critval_res[0]+1e-10) {
                    critval_res[0] = Math.abs(logrank_stat[i]);
                    maxnumlocs = 0;
                }
                maxlocs[maxnumlocs] = i;
                maxnumlocs = maxnumlocs + 1;
            }
        }
        
        int maxloc = maxlocs[rand(r) % maxnumlocs];
        int cutloc = rows[maxloc]-1;

        double u = rand(r) * 1.0 / RAND_MAX;

        // if points are close the just take average. If points are farthur sample randomly from lerp
        if (x[cutloc+1] - x[cutloc] < 1.9*1e-6) {
            cutval_res[0] = (x[cutloc] + x[cutloc+1])/2;
        } else {
            cutval_res[0] = ((1-u)*(x[cutloc]+1e-6) + u*(x[cutloc+1]-1e-6));
            if (cutval_res[0] < x[cutloc]+1e-8 || cutval_res[0] > x[cutloc+1]-1e-8) {
                throw new RuntimeException("random splitpoint has to lie in between the upper and lower limit");
            }
        }
    }
    
    private static void shuffle(int[] arr, int n, Random r) {
        for (int i=0; i < n-1; i++) {
            int j = i + rand(r) / (RAND_MAX / (n - i) + 1);
            int t = arr[j];
            arr[j] = arr[i];
            arr[i] = t;
        }
    }
    
    private static int[] sorder;
    private static void rankSort(double[] arr, int len) {
        for (int i=0; i<len; i++) {
            sorder[i] = i;
        }
        dp_quick(arr, sorder, 0, len-1);
    }
    
    private static void dp_quick(double[] input, int[] sorder, int min, int max) {
      if (max - min > 0) {
        int i = min;
        int j = max;
        double pivot = input[sorder[(i+j) >> 1]];
        do {
          while(input[sorder[i]] < pivot) i++;
          while(input[sorder[j]] > pivot) j--;
          if (i >= j) break;
          int t = sorder[i];
          sorder[i] = sorder[j];
          sorder[j] = t;
        } while(++i < --j);

        while (min < j && input[sorder[j]] == pivot) j--;
        if (min < j) dp_quick(input, sorder, min, j);

        while (i < max && input[sorder[i]] == pivot) i++;
        if (i < max) dp_quick(input, sorder, i, max);
      }
    }
    
    private static void sort(double[] arr, int len) {
        quickSort(arr, 0, len - 1);
    }
    
    private static void quickSort(double[] input, int min, int max) {
      if (max - min > 0) {
        int i = min;
        int j = max;
        double pivot = input[(i+j) >> 1];
        do {
          while(input[i] < pivot) i++;
          while(input[j] > pivot) j--;
          if (i >= j) break;
          double t = input[i];
          input[i] = input[j];
          input[j] = t;
        } while(++i < --j);

        while (min < j && input[j] == pivot) j--;
        if (min < j) quickSort(input, min, j);

        while (i < max && input[i] == pivot) i++;
        if (i < max) quickSort(input, i, max);
      }
    }
    
    private static void sort(int[] arr, int len) {
        quickSort(arr, 0, len - 1);
    }
    
    private static void quickSort(int[] input, int min, int max) {
      if (max - min > 0) {
        int i = min;
        int j = max;
        int pivot = input[(i+j) >> 1];
        do {
          while(input[i] < pivot) i++;
          while(input[j] > pivot) j--;
          if (i >= j) break;
          int t = input[i];
          input[i] = input[j];
          input[j] = t;
        } while(++i < --j);

        while (min < j && input[j] == pivot) j--;
        if (min < j) quickSort(input, min, j);

        while (i < max && input[i] == pivot) i++;
        if (i < max) quickSort(input, i, max);
      }
    }
}