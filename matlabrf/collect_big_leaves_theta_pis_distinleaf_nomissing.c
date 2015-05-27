/* fwd_theta_pis.c
	Propagates data down a regression tree and returns the leaves that 
    are reached; in this variant no missing values are allowed.
    Usage: [cell_of_leaves, numCellsFilled] = fwd_theta_pis(var,cut,class,leftchildren,rightchildren,catsplit,Theta,X,Theta_rows,N,x_rows,n,thisnode,numCellsFilled,cell_of_leaves)

    We propagate Theta and X down the tree.
    If splitting on a parameter setting, Theta is split and X preserved,
	if splitting on an instance feature vice versa.

    Only the relevant parts of the regression tree are passed:
	var  Nx1 array of split variable
	cut  Nx1 array of cutoff value for continuous splits/indices for categorical splits
	nodepred   Nx1 array of node predictions
	leftchildren   Nx1 array of left children indices
	rightchildren   Nx1 array of right children indices
	catsplit  Nx2 cell array for categorical splits, left categories in column 1 and right categories in column 2.
   */
 
#include "mex.h"
#include <math.h>

#if !defined(ABS)
#define	ABS(A)	((A) > (0) ? (A) : (-A))
#endif

#if !defined(MAX)
#define	MAX(A, B)	((A) > (B) ? (A) : (B))
#endif

#if !defined(MIN)
#define	MIN(A, B)	((A) < (B) ? (A) : (B))
#endif

void printMatrixDouble(double* X,int nRows, int nCols)
{
    int i,j;
    
    for(i = 0; i < nRows; i++) {
        printf("< ");
        for(j = 0; j < nCols; j++) {
            printf("%lf ",X[i+nRows*j]);}
        printf(">\n");}
}

void printMatrixInt(int* X,int nRows, int nCols)
{
    int i,j;
    
    for(i = 0; i < nRows; i++) {
        printf("< ");
        for(j = 0; j < nCols; j++) {
            printf("%d ",X[i+nRows*j]);}
        printf(">\n");}
}

                                                                                                                                                                                                                                                                                         
void fwd_theta_pis(const int* var, const double* cut, const mxArray* ysub, const int* leftchildren, const int* rightchildren, const mxArray* catsplit, const double* Theta, const double* X, const int mrowsTheta, const int mrowsX, const int ncolsTheta, const int mrowsCatsplit, const int* Theta_rows, const int N, const int* x_rows, const int n, const int thisnode, const double* g_leaf, const double* m_leaf, const int* n_leaf, const double* leaf_mean, const double* leaf_var, mxArray* cell_of_leaves, int* pointerToNumCellsFilled)
{
	/* This is a recursive function. Starts at top node, then recurses over
    *  child nodes.  THISNODE is the current node at each step (Matlab indexing, so subtract -1 to index MEX arrays).
	*  Theta_rows are the idxs of Theta left at this node, x_rows the idxs of X.
	*  cell_of_leaves is passed from above, and we simply fill the leaves from
    *  this subtree into it starting at index numCellsFilled. 
    *
    * N and n are the size of Theta_rows and x_rows, respectively; they change per node.
    * mrowsTheta and mrowsX, on the other hand are fix, the number of rows in X and Theta, respectively.
    * 
    *  Matlab:
    *  function [cell_of_leaves, numCellsFilled] = fwd_theta_pis(var,cut,class,stddev,leftchildren,rightchildren,catsplit,Theta,X,Theta_rows,N,x_rows,n,thisnode,numCellsFilled,cell_of_leaves)
    */
	
    int splitvar, leftchild, rightchild, i, j, xLen, columnOffset, numLeft, numRight, nCatLeft, nCatRight;
    int *idx_goingleft, *idx_goingright, *sub_array_for_recursion;
    int *pointer_for_Theta_rows_result, *pointer_for_x_rows_result, *pointer_for_thisnode_result;
    int dims[2], n_ysub_here;
    mxArray *mxArray_for_Theta_rows_results, *mxArray_for_x_rows_results, *mxArray_for_thisnode_result, *mx_ysub_here, *mx_ysub_result, *mxArray_for_suff_stats;
    mxArray *cell_array_ptr;
    mxArray *mx_xleft, *mx_xright;
    int *xleft, *xright;
    double cutoff, *ysub_here, *ysub_res, *pointer_for_suff_stats;
    double *x;
    bool split_on_param, goleft, goright;
    
    splitvar    = var[thisnode-1];
    cutoff      = cut[thisnode-1];
    leftchild   = leftchildren[thisnode-1];
    rightchild  = rightchildren[thisnode-1];
    
/*    printf("THISNODE %d\n", thisnode); */

    /* Terminal case */
    if (splitvar==0){
/*        printf("Terminal thisnode=%d\n", thisnode);  */
        /* 
         * Matlab: cell_of_leaves{numCellsFilled} = {Theta_rows, x_rows, id, thisnode}; 
         * Here in MEX, this single statement takes up the next ~45 lines :-(
         */

        /* Create mxArray to hold Theta_rows, make a pointer to it and fill it. */
        dims[0] = 1;
        dims[1] = N;
        mxArray_for_Theta_rows_results = mxCreateNumericArray(2, dims, mxINT32_CLASS, mxREAL);
        pointer_for_Theta_rows_result = (int*) mxGetData(mxArray_for_Theta_rows_results);
        for(i=0; i<N; i++){
            pointer_for_Theta_rows_result[i] = Theta_rows[i] + 1; /* +1 for Matlab indexing */
        }

        /* Create mxArray to hold x_rows, make a pointer to it and fill it. */
        dims[0] = 1;
        dims[1] = n;
        mxArray_for_x_rows_results = mxCreateNumericArray(2, dims, mxINT32_CLASS, mxREAL);
        pointer_for_x_rows_result = (int*) mxGetData(mxArray_for_x_rows_results);
        for(i=0; i<n; i++){
            pointer_for_x_rows_result[i] = x_rows[i] + 1; /* +1 for Matlab indexing */
        }

        /* Use mxArray saved in this leaf. */
        mx_ysub_here = mxGetCell(ysub, thisnode-1);
		if (mxGetClassID(mx_ysub_here) != mxDOUBLE_CLASS){
			mexErrMsgTxt("Elements of ysub cell array have to be doubles");
		};

        n_ysub_here = mxGetN(mx_ysub_here);
        ysub_here = mxGetPr(mx_ysub_here);
        dims[0] = 1;
        dims[1] = n_ysub_here;
        mx_ysub_result = mxCreateNumericArray(2, dims, mxDOUBLE_CLASS, mxREAL);
        ysub_res = mxGetPr(mx_ysub_result);
        for(i=0; i<n_ysub_here; i++){
            ysub_res[i] = ysub_here[i];
        }

        /* Create mxArray to hold thisnode, make a pointer to it and fill it. */
        dims[0] = 1;
        dims[1] = 1;
        mxArray_for_thisnode_result = mxCreateNumericArray(2, dims, mxINT32_CLASS, mxREAL);
        pointer_for_thisnode_result = (int*) mxGetData(mxArray_for_thisnode_result);
        pointer_for_thisnode_result[0] = thisnode; /* already Matlab index */
        
        /* Create mxArray to hold sufficient stats, make a pointer to it and fill it. */
        dims[0] = 1;
        dims[1] = 5;
        mxArray_for_suff_stats = mxCreateNumericArray(2, dims, mxDOUBLE_CLASS, mxREAL);
        pointer_for_suff_stats = mxGetPr(mxArray_for_suff_stats);
        pointer_for_suff_stats[0] = g_leaf[thisnode-1];
        pointer_for_suff_stats[1] = m_leaf[thisnode-1];
        pointer_for_suff_stats[2] = (double) n_leaf[thisnode-1];
        pointer_for_suff_stats[3] = leaf_mean[thisnode-1];
        pointer_for_suff_stats[4] = leaf_var[thisnode-1];

        /* Create a cell array and fill it with the 5 pointers to mxArrays. */
        cell_array_ptr = mxCreateCellMatrix(1,5);
        mxSetCell(cell_array_ptr, 0, mxArray_for_Theta_rows_results);
        mxSetCell(cell_array_ptr, 1, mxArray_for_x_rows_results);
        mxSetCell(cell_array_ptr, 2, mx_ysub_result); /*mx_ysub_here);*/
        mxSetCell(cell_array_ptr, 3, mxArray_for_thisnode_result);
        mxSetCell(cell_array_ptr, 4, mxArray_for_suff_stats);
        
        /* Put this newly created cell array into the result cell array, at index numCellsFilled. */
        mxSetCell(cell_of_leaves, pointerToNumCellsFilled[0], cell_array_ptr);
        
        /* Increase the counter for numbers of cells filled in cell_of_leaves.  */     
        pointerToNumCellsFilled[0]++;
        return;
    }
   
    /* Now deal with non-terminal nodes */
    
    /* Allocate Memory for Auxiliary Arrays. 
     * (can't do that earlier b/c the terminal node just returns. */
    idx_goingleft  = mxCalloc(MAX(N,n),sizeof(int));
    idx_goingright = mxCalloc(MAX(N,n),sizeof(int));
    sub_array_for_recursion = mxCalloc(MAX(N,n),sizeof(int));
    
    x              = mxCalloc(MAX(N,n),sizeof(double));

    /* Determine whether splitting on a parameter or an instance feature. */
    if (ABS(splitvar) <= ncolsTheta){
        split_on_param = true;
        xLen = N;
        /* Matlab: x = Theta(Theta_rows,abs(splitvar)); */
        columnOffset = mrowsTheta*(ABS(splitvar)-1);
        for (i=0; i<xLen; i++){
            x[i] = Theta[Theta_rows[i] + columnOffset]; /* index: row + column*numRows */
        }
    } else {
        split_on_param = false;
        xLen = n;
        /* Matlab: x = X(x_rows,abs(splitvar)-size(Theta,2)); */
        columnOffset = mrowsX*(ABS(splitvar)-1-ncolsTheta);
        for (i=0; i<xLen; i++){
            x[i] = X[x_rows[i] + columnOffset]; /* index: row + column*numRows */
        }
    }

    numLeft = 0;
    numRight = 0;

    /* Determine if this point goes left, goes right, or stays here */
    if (splitvar>0){ /* continuous variable */
        for (i=0; i<xLen; i++){
            if (x[i] <= cutoff){
/*       			printf("i=%d, continuous %d left; %lf<%lf\n", i, splitvar, x[i], cutoff); */
                idx_goingleft[numLeft++] = i;
            } else {
/*       			printf("i=%d, continuous %d right; %lf>=%lf\n", i, splitvar, x[i], cutoff); */
                idx_goingright[numRight++] = i;
            }
        }
    } else { /* categorical variable */
        
        /*
         * Get the references to the appropriate entries in the catsplit cell array.
         */
        mx_xleft = mxGetCell(catsplit, cutoff-1);
		if (mxGetClassID(mx_xleft) != mxINT32_CLASS){
			mexErrMsgTxt("Elements of catsplit cell array have to be ints (cast them to int in Matlab)");
		};
		xleft = (int*) mxGetData(mx_xleft);
        nCatLeft = mxGetNumberOfElements(mx_xleft);

/*          printf("mrowsCatsplit=%d\n",mrowsCatsplit);*/
  

        mx_xright = mxGetCell(catsplit, cutoff-1+mrowsCatsplit);
		if (mxGetClassID(mx_xright) != mxINT32_CLASS){
			mexErrMsgTxt("Elements of catsplit cell array have to be ints (cast them to int in Matlab)");
		};
		xright = (int*) mxGetData(mx_xright);
        nCatRight = mxGetNumberOfElements(mx_xright);

        for (i=0; i<xLen; i++){
            goleft = false;
            for (j=0; j<nCatLeft; j++){  /*=== Does the value match any of the ones going left? */
/*      			printf("i=%d, categorical %d left; %d==%d?\n", i, -splitvar, (int) x[i], xleft[j]);  */
                if (xleft[j] == (int) x[i]){
                    goleft = true;
                    break;
                }
            }
            if (goleft){
                idx_goingleft[numLeft++] = i;
            } else { /* if we don't go left, maybe we go right ... */
                goright = false;
                for (j=0; j<nCatRight; j++){
/*        			printf("i=%d, categorical %d right; %d==%d?\n", i, -splitvar, (int) x[i], xright[j]); */
                    if (xright[j] == (int) x[i]){
                        goright = true;
                        break;
                    }
                }
                if (goright){
                    idx_goingright[numRight++] = i;
                } else {  
                    /* We go neither left nor right => missing in the subtree */
                    mexErrMsgTxt("Missing value -- not allowed in this implementation (randomly assign missing values during tree construction!).");
                }
            }
        }
    }

    /* Determine if this point goes left, goes right, or stays here */
/*    
    printf("idx_goingleft:\n");
    printMatrixInt(idx_goingleft, numLeft, 1);
    printf("\n\n");
    
    printf("idx_goingright:\n");
    printMatrixInt(idx_goingright, numRight, 1);
    printf("\n\n");

*/
    if (numLeft>0){ /* going left */
        if (split_on_param){
            for(i=0; i<numLeft; i++) sub_array_for_recursion[i] = Theta_rows[idx_goingleft[i]];
            fwd_theta_pis(var,cut,ysub,leftchildren,rightchildren,catsplit,Theta,X,mrowsTheta,mrowsX,ncolsTheta,mrowsCatsplit,sub_array_for_recursion,numLeft,x_rows,n,leftchild,g_leaf,m_leaf,n_leaf,leaf_mean,leaf_var,cell_of_leaves,pointerToNumCellsFilled);
        } else {
            for(i=0; i<numLeft; i++) sub_array_for_recursion[i] = x_rows[idx_goingleft[i]];
            fwd_theta_pis(var,cut,ysub,leftchildren,rightchildren,catsplit,Theta,X,mrowsTheta,mrowsX,ncolsTheta,mrowsCatsplit,Theta_rows,N,sub_array_for_recursion,numLeft,leftchild,g_leaf,m_leaf,n_leaf,leaf_mean,leaf_var,cell_of_leaves,pointerToNumCellsFilled);
        }
    }

    if (numRight>0){ /* going right */
        if (split_on_param){
            for(i=0; i<numRight; i++) sub_array_for_recursion[i] = Theta_rows[idx_goingright[i]];
            fwd_theta_pis(var,cut,ysub,leftchildren,rightchildren,catsplit,Theta,X,mrowsTheta,mrowsX,ncolsTheta,mrowsCatsplit,sub_array_for_recursion,numRight,x_rows,n,rightchild,g_leaf,m_leaf,n_leaf,leaf_mean,leaf_var,cell_of_leaves,pointerToNumCellsFilled);
        } else {
            for(i=0; i<numRight; i++) sub_array_for_recursion[i] = x_rows[idx_goingright[i]];
            fwd_theta_pis(var,cut,ysub,leftchildren,rightchildren,catsplit,Theta,X,mrowsTheta,mrowsX,ncolsTheta,mrowsCatsplit,Theta_rows,N,sub_array_for_recursion,numRight,rightchild,g_leaf,m_leaf,n_leaf,leaf_mean,leaf_var,cell_of_leaves,pointerToNumCellsFilled);
        }
    }
        
	/*=== Free Memory for Auxiliary Arrays. */
	mxFree(idx_goingleft);
	mxFree(idx_goingright);
    mxFree(sub_array_for_recursion);
    
    mxFree(x);
}


void mexFunction( int nlhs, mxArray *plhs[],
                  int nrhs, const mxArray *prhs[] )
{
  double *X, *Theta, *cut, *g_leaf, *m_leaf, *leaf_mean, *leaf_var;
  int    *var, *leftchildren, *rightchildren, *numCellsFilledPointer, *n_leaf;
  int i, mrowsX, ncolsX, mrows, ncols, mrowsCatsplit, mrowsT, mrowsTheta, ncolsTheta;
  int *Theta_rows, *x_rows;
  int dims[2]={1,1};
  
  /* Check for proper number of arguments. */
  if(nrhs!=13) {
    mexErrMsgTxt("Usage: [cell_of_leaves, numCellsFilled] = collect_big_leaves_theta_pis_nomissing(var, cut, ysub, leftchildren, rightchildren, catsplit, Theta, X, g_leaf, m_leaf, n_leaf).");
  } else if(nlhs>2) {
    mexErrMsgTxt("Too many output arguments");
  }

  /* Check each argument for proper form and dimensions. */
  mrowsT = mxGetM(prhs[0]);
  ncols = mxGetN(prhs[0]);
  if( !mxIsInt32(prhs[0]) || mxIsComplex(prhs[0]) || !(ncols==1)) {
    mexErrMsgTxt("var must be a noncomplex int Nx1 vector.");
  }
  var = (int*) mxGetPr(prhs[0]);

  mrows = mxGetM(prhs[1]);
  ncols = mxGetN(prhs[1]);
  if( !mxIsDouble(prhs[1]) || mxIsComplex(prhs[1]) || !(ncols==1) || !(mrows==mrowsT)) {
    mexErrMsgTxt("cut must be a noncomplex double Nx1 vector.");
  }
  cut = mxGetPr(prhs[1]);

  mrows = mxGetM(prhs[2]);
  ncols = mxGetN(prhs[2]);
  if( !mxIsCell(prhs[2]) || !(ncols==1) || !(mrows==mrowsT)) {
    mexErrMsgTxt("ysub must be a Nx1 cell array.");
  }

  
  mrows = mxGetM(prhs[3]);
  ncols = mxGetN(prhs[3]);
  if( !mxIsInt32(prhs[3]) || mxIsComplex(prhs[3]) || !(ncols==1) || !(mrows==mrowsT)) {
    mexErrMsgTxt("leftchildren must be a noncomplex int Nx1 vector.");
  }
  leftchildren = (int*) mxGetPr(prhs[3]);

  mrows = mxGetM(prhs[4]);
  ncols = mxGetN(prhs[4]);
  if( !mxIsInt32(prhs[4]) || mxIsComplex(prhs[4]) || !(ncols==1) || !(mrows==mrowsT)) {
    mexErrMsgTxt("rightchildren must be a noncomplex int Nx1 vector.");
  }
  rightchildren = (int*) mxGetPr(prhs[4]);
  
  mrowsCatsplit = mxGetM(prhs[5]);
  ncols = mxGetN(prhs[5]);
  if( !mxIsCell(prhs[5]) || !(ncols==2) ) {
    mexErrMsgTxt("catsplit must be a Mx2 cell array");
  }

  mrowsTheta = mxGetM(prhs[6]);
  ncolsTheta = mxGetN(prhs[6]);
  if( !mxIsDouble(prhs[6]) || mxIsComplex(prhs[6]) ) {
    mexErrMsgTxt("Theta must be a noncomplex double Nxp matrix.");
  }
  Theta = mxGetPr(prhs[6]);

  mrowsX = mxGetM(prhs[7]);
  ncolsX = mxGetN(prhs[7]);
  if( !mxIsDouble(prhs[7]) || mxIsComplex(prhs[7]) ) {
    mexErrMsgTxt("X must be a noncomplex double nxd matrix).");
  }
  X = mxGetPr(prhs[7]);

  mrows = mxGetM(prhs[8]);
  ncols = mxGetN(prhs[8]);
  if( !mxIsDouble(prhs[8]) || mxIsComplex(prhs[8]) || !(ncols==1) || !(mrows==mrowsT)) {
    mexErrMsgTxt("g_leaf must be a noncomplex double Nx1 vector.");
  }
  g_leaf = mxGetPr(prhs[8]);

  mrows = mxGetM(prhs[9]);
  ncols = mxGetN(prhs[9]);
  if( !mxIsDouble(prhs[9]) || mxIsComplex(prhs[9]) || !(ncols==1) || !(mrows==mrowsT)) {
    mexErrMsgTxt("m_leaf must be a noncomplex double Nx1 vector.");
  }
  m_leaf = mxGetPr(prhs[9]);
  
  mrows = mxGetM(prhs[10]);
  ncols = mxGetN(prhs[10]);
  if( !mxIsInt32(prhs[10]) || mxIsComplex(prhs[10]) || !(ncols==1) || !(mrows==mrowsT)) {
    mexErrMsgTxt("n_leaf must be a noncomplex int Nx1 vector.");
  }
  n_leaf = (int*) mxGetData(prhs[10]);
  
  mrows = mxGetM(prhs[11]);
  ncols = mxGetN(prhs[11]);
  if( !mxIsDouble(prhs[11]) || mxIsComplex(prhs[11]) || !(ncols==1) || !(mrows==mrowsT)) {
    mexErrMsgTxt("leaf_mean must be a noncomplex double Nx1 vector.");
  }
  leaf_mean = mxGetPr(prhs[11]);

  mrows = mxGetM(prhs[12]);
  ncols = mxGetN(prhs[12]);
  if( !mxIsDouble(prhs[12]) || mxIsComplex(prhs[12]) || !(ncols==1) || !(mrows==mrowsT)) {
    mexErrMsgTxt("leaf_var must be a noncomplex double Nx1 vector.");
  }
  leaf_var = mxGetPr(prhs[12]);
  
  
  /* Create cell array for the second return argument. */
  plhs[0] = mxCreateCellMatrix(1,mrowsT); /* Can't have more leaves than there are nodes in the tree*/

  /* Create matrix for the second return argument (a scalar) and assign pointer. */
  plhs[1] = mxCreateNumericArray(2, dims, mxINT32_CLASS, mxREAL);
  numCellsFilledPointer = (int*) mxGetData(plhs[1]); /* Simply fill numCellsFilledPointer[0] with the scalar. */
  
  /*=== Allocate Memory for Auxiliary Arrays. */
  Theta_rows = mxCalloc(mrowsTheta,sizeof(int));
  x_rows = mxCalloc(mrowsX,sizeof(int));
  
  for(i=0; i<mrowsTheta; i++){
      Theta_rows[i] = i;
  }
  for(i=0; i<mrowsX; i++){
      x_rows[i] = i;
  }
  
  /* Do the actual work. */
  fwd_theta_pis(var, cut, prhs[2], leftchildren, rightchildren, prhs[5], Theta, X, mrowsTheta, mrowsX, ncolsTheta, mrowsCatsplit, Theta_rows, mrowsTheta, x_rows, mrowsX, 1, g_leaf, m_leaf, n_leaf, leaf_mean, leaf_var, plhs[0], numCellsFilledPointer);

  /*=== Free Memory for Auxiliary Arrays. */
  mxFree(Theta_rows); 
  mxFree(x_rows);
}