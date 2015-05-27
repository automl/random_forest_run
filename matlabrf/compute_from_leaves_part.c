#include "mex.h"
#include <math.h>

#if !defined(MAX)
#define	MAX(A, B)	((A) > (B) ? (A) : (B))
#endif

#if !defined(MIN)
#define	MIN(A, B)	((A) < (B) ? (A) : (B))
#endif

void mexFunction( int nlhs, mxArray *plhs[],
                  int nrhs, const mxArray *prhs[] )
{
    int i, m, M, mrows, ncols, numTheta, num_leaves, numInstances, node_idx, l_idx, num_theta_idx_here, num_entries_for_leaf, numInstancesHere;
    double ratio_inst_idx, leaf_mean, leaf_var;
    double *means, *vars, *tmp_double_ptr, *tree_means, *tree_vars, *stats;
    int *tmp_int_ptr, *theta_idx_here;
    int dims[2];
    mxArray *mx_leaves, *mx_leaf, *mx_node_idx, *mx_theta_idx_here, *mx_pi_idx, *mx_leaf_stats; 
    
    /* Check for proper number of arguments. */
    if(nrhs!=3 || nlhs> 3) {
        mexErrMsgTxt("Usage: [means, vars, tree_means] = compute_from_leaves_part(cell_of_leaves, numTheta, numInstances).");
    }

    if (prhs[0] == NULL || !mxIsCell(prhs[0])){
        mexErrMsgTxt("cell_of_leaves must be a nonempty cell array");
    }
    M = mxGetNumberOfElements(prhs[0]);
    
    mrows = mxGetM(prhs[1]);
    ncols = mxGetN(prhs[1]);
    if( !mxIsInt32(prhs[1]) || mxIsComplex(prhs[1]) || !(ncols==1) || !(mrows==1)) {
        mexErrMsgTxt("numTheta must be a noncomplex int scalar.");
    }
    tmp_int_ptr = (int*) mxGetData(prhs[1]);
    numTheta = tmp_int_ptr[0];

    mrows = mxGetM(prhs[2]);
    ncols = mxGetN(prhs[2]);
    if( !mxIsInt32(prhs[2]) || mxIsComplex(prhs[2]) || !(ncols==1) || !(mrows==1)) {
        mexErrMsgTxt("numInstances must be a noncomplex int scalar.");
    }
    tmp_int_ptr = (int*) mxGetData(prhs[2]);
    numInstances = tmp_int_ptr[0];
 
    /* Outputs: mean and var, both scalar doubles */
    dims[0] = numTheta;
    dims[1] = 1;

    plhs[0] = mxCreateNumericArray(2, dims, mxDOUBLE_CLASS, mxREAL);
    means = mxGetPr(plhs[0]);

    plhs[1] = mxCreateNumericArray(2, dims, mxDOUBLE_CLASS, mxREAL);
    vars = mxGetPr(plhs[1]);

    dims[1] = M;
    plhs[2] = mxCreateNumericArray(2, dims, mxDOUBLE_CLASS, mxREAL);
    tree_means = mxGetPr(plhs[2]);
    
/*    tree_means = mxCalloc(numTheta*M, sizeof(double));*/
    tree_vars  = mxCalloc(numTheta*M, sizeof(double));

    
    
    /* The work to be done */
    /* Compute predictive distributions at each leaf, at each tree, and in total. */
    for( m=1; m<=M; m++ ){
        
        for (i=0; i<numTheta; i++){
            tree_means[ (m-1)*numTheta + i ] = 0;
             tree_vars[ (m-1)*numTheta + i ] = 0;
        }
        
        
/*        printf("  m=%d\n",m);*/
        mx_leaves = mxGetCell(prhs[0], m-1);
        if (mx_leaves == NULL || !mxIsCell(mx_leaves)){
            mexErrMsgTxt("entries of cell_of_leaves must be a nonempty cell arrays");
        }
        num_leaves = mxGetNumberOfElements(mx_leaves);
        
        /* Do the work for each leaf. */
        for( l_idx=1; l_idx<=num_leaves; l_idx++ ){
/*            printf("    l_idx=%d\n",l_idx);*/
            mx_leaf = mxGetCell( mx_leaves, l_idx-1 );
            if (mx_leaf == NULL || !mxIsCell(mx_leaf)){
                mexErrMsgTxt("entries of entries of cell_of_leaves must be nonempty cell arrays");
            }
            num_entries_for_leaf = mxGetNumberOfElements(mx_leaf);
            if (num_entries_for_leaf != 5){
                mexErrMsgTxt("Leaves must have 5 values.");
            }

            
            /* Using mean & var from parametric fit stored in leaf's statistics. */
            mx_leaf_stats = mxGetCell(mx_leaf, 5-1);
            if (mx_leaf_stats == NULL){
                 mexErrMsgTxt("mx_leaf_stats is null.");
            }
            mrows = mxGetM(mx_leaf_stats);
            ncols = mxGetN(mx_leaf_stats);
            if( !mxIsDouble(mx_leaf_stats) || mxIsComplex(mx_leaf_stats) || !(mrows==1) || !(ncols==5) ) {
                mexErrMsgTxt("mx_leaf_stats must be a noncomplex 1x5 double vector.");
            }
            stats = mxGetPr(mx_leaf_stats);            
            leaf_mean = stats[4-1];
            leaf_var  = stats[5-1];

            
            mx_theta_idx_here = mxGetCell(mx_leaf, 1-1);
            if (mx_theta_idx_here == NULL){
                 mexErrMsgTxt("mx_theta_idx_here is null.");
            }
            mrows = mxGetM(mx_theta_idx_here);
            num_theta_idx_here = mxGetN(mx_theta_idx_here);
            if( !mxIsInt32(mx_theta_idx_here) || mxIsComplex(mx_theta_idx_here) || !(mrows==1) ) {
                mexErrMsgTxt("mx_theta_idx_here must be a noncomplex int row vector.");
            }
            theta_idx_here = (int*) mxGetData(mx_theta_idx_here);
            
            mx_pi_idx = mxGetCell(mx_leaf, 2-1);
            if (mx_pi_idx == NULL){
                 mexErrMsgTxt("mx_pi_idx is null.");
            }                
            mrows = mxGetM(mx_pi_idx);
            numInstancesHere = mxGetN(mx_pi_idx);
            if( !mxIsInt32(mx_pi_idx) || mxIsComplex(mx_pi_idx) || !(mrows==1) ) {
                mexErrMsgTxt("mx_pi_idx must be a noncomplex int row vector.");
            }

            ratio_inst_idx = numInstancesHere/(numInstances+0.0);

/*                printf("leaf_mean=%lf, ratio_inst_idx=%lf \n", leaf_mean,ratio_inst_idx);*/
            /* Matlab indexing is: column_idx * total_num_rows + row_idx */
            for (i=0; i<num_theta_idx_here; i++){
                tree_means[ (m-1)*numTheta + (theta_idx_here[i]-1) ] += ratio_inst_idx * leaf_mean;
                 tree_vars[ (m-1)*numTheta + (theta_idx_here[i]-1) ] += ratio_inst_idx*ratio_inst_idx * leaf_var;
            }
        }
    }

    for (i=0; i<numTheta; i++){
/*        printf("  i=%d\n",i);*/
        /* means = mean(tree_means,2); */
        means[i] = 0;
        for (m=0; m<M; m++){
            means[i] += tree_means[ m*numTheta + i ];
        }
        means[i] /= M;
        
        /* vars = mean(tree_vars + tree_means.^2,2) - means.^2; */
        vars[i] = 0;
        for (m=0; m<M; m++){
            vars[i] += tree_vars[ m*numTheta + i ] + tree_means[ m*numTheta + i ]*tree_means[ m*numTheta + i ] - means[i]*means[i];
        }
        vars[i] /= M;
    }
        
/*    mxFree(tree_means);*/
    mxFree(tree_vars);
}