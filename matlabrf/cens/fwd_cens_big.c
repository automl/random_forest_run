/* fwd.c
	Propagates data down a regression tree.
	Only the relevant parts of the regression tree are passed:
	var  Nx1 array of split variable
	cut  Nx1 array of cutoff value for continuous splits/indices for categorical splits
	nodepred   Nx1 array of node predictions
	children   Nx2 array of children indices
	catsplit  Nx2 cell array for categorical splits, left categories in column 1 and right categories in column 2
	
	This is a SLOW implementation: it propagates one input at a time, instead of splitting the data into 
	appropriate chunks going down to each of its children.
	It takes about 0.3 seconds to do 100 predictions of 1000 data points each.
	The Matlab version propagating one input at a time takes about 100 seconds,
	the normal Matlab version takes 6 seconds.
	But loops are slow in Matlab, so there is probably only potential for another speedup of a factor of 2-5 or so.
   */
 
#include "mex.h"

#if !defined(ABS)
#define	ABS(A)	((A) > (0) ? (A) : (-A))
#endif
void fwd_one_row(const double* X, const int* var, const double* cut, const mxArray* mx_y_sub_cell, const mxArray* mx_cens_sub_cell, const int* children, const mxArray* catsplit, const int row, const int numrows, const int numnodes, const int numRowsC, const int thisnode, mxArray* mx_y_res_cell, mxArray* mx_cens_res_cell)
{
	/*=== This is a recursive function for propagating a single row X.
	//=== Starts at top node, then recurses over child nodes. 
	//=== thisnode is the current node at each step. All other variables are the same at each node.*/
	
	int splitvar, left_kid, right_kid, nCatLeft, nCatRight, i, cutoff_int, *arrayOfVariables, numElements, numElements2;
	double cutoff, x;
    bool goleft=false, goright=false;
	mxArray    *cell_element_ptr;
    double *y_sub, *y_res;
    int *cens_sub, *cens_res, dims[2];
    mxArray *mx_y_res, *mx_cens_res;
/*printf("node %d, splitvar=%d\n", thisnode, var[thisnode]);*/
    
	splitvar = var[thisnode];
	cutoff = cut[thisnode];
	left_kid = children[thisnode+0]-1;
	right_kid = children[thisnode+numnodes]-1;
	
	/*=== Terminal case --- this is 1:1 the same as for the missing value case above.*/
	if (splitvar==0){
/*printf("Terminal case %d\n", thisnode); */
        
        /*=== Get the y_sub array from the cell array, copy it, and put it into the result cell array y_res. */
        cell_element_ptr = mxGetCell(mx_y_sub_cell, thisnode);
        if (cell_element_ptr == NULL){
            printf("Error in terminal case %d\n", thisnode);
			mexErrMsgTxt("Empty cell in mx_y_sub_cell\n");
		}        
        if( !mxIsDouble(cell_element_ptr) || mxIsComplex(cell_element_ptr) ) {
            mexErrMsgTxt("mx_y_sub_cell must have entries that are double arrays.");
        }
        numElements = mxGetNumberOfElements(cell_element_ptr);
        y_sub = mxGetPr(cell_element_ptr);
        dims[0] = 1;
        dims[1] = numElements;
        mx_y_res = mxCreateNumericArray(2, dims, mxDOUBLE_CLASS, mxREAL);
        y_res = mxGetPr(mx_y_res);
        for(i=0; i<numElements; i++){
            y_res[i] = y_sub[i];
        }
        mxSetCell(mx_y_res_cell, row, mx_y_res);
        
        /*=== Same thing for the cens_sub array */
        cell_element_ptr = mxGetCell(mx_cens_sub_cell, thisnode);
        if( !mxIsInt32(cell_element_ptr) || mxIsComplex(cell_element_ptr) ) {
            mexErrMsgTxt("mx_cens_sub_cell must have entries that are int arrays.");
        }
        numElements2 = mxGetNumberOfElements(cell_element_ptr);
        if (numElements != numElements2){
            mexErrMsgTxt("at each node, mx_ysub and mx_censsub must the have same number of elements!\n");
        }
        cens_sub = (int*) mxGetData(cell_element_ptr);
        mx_cens_res = mxCreateNumericArray(2, dims, mxINT32_CLASS, mxREAL);
        cens_res = (int*) mxGetData(mx_cens_res);
        for(i=0; i<numElements; i++){
            cens_res[i] = cens_sub[i];
        }
        mxSetCell(mx_cens_res_cell, row, mx_cens_res);
        return;
	}
	
	/*=== Now deal with non-terminal nodes.*/
/*printf("Non-terminal case %d\n", thisnode); */
	
	/* Determine if this point goes left, goes right, or stays here.*/
	x = X[row + (ABS(splitvar)-1)*numrows];
/*	printf("fwd_one_row, splitnode %d, left kid %d, right kid %d, x=%lf\n", thisnode+1, left_kid+1, right_kid+1, x);
//	printf("x[0] ... x[5] = %lf, %lf, %lf, %lf, %lf, %lf\n", X[row+0*numrows], X[row+1*numrows], X[row+2*numrows], X[row+3*numrows], X[row+4*numrows], X[row+5*numrows]);*/
		

	if (splitvar>0){    /* continuous variable*/
/*printf("continuous %d; x=%lf\n", splitvar,x);*/
	    if (x < cutoff){
	        fwd_one_row(X, var, cut, mx_y_sub_cell, mx_cens_sub_cell, children, catsplit, row, numrows, numnodes, numRowsC, left_kid, mx_y_res_cell, mx_cens_res_cell);
            return;
	    } else {
	        fwd_one_row(X, var, cut, mx_y_sub_cell, mx_cens_sub_cell, children, catsplit, row, numrows, numnodes, numRowsC, right_kid, mx_y_res_cell, mx_cens_res_cell);
            return;
		}
	} else {
		cutoff_int = (int) cutoff;
/*printf("categorical %d; x=%lf\n", splitvar, x);*/
		
/*printf("idx: %d\n", cutoff_int-1);*/
		if(!mxIsCell(catsplit)){
			mexErrMsgTxt("catsplit is not a cell array\n");
		}
		/*=== Does the value match any of the ones going left? */
		cell_element_ptr = mxGetCell(catsplit, cutoff_int-1);
		if (cell_element_ptr == NULL){
			mexErrMsgTxt("Empty Cell in catsplit\n");
		}
		if (mxGetClassID(cell_element_ptr) != mxINT32_CLASS){
			mexErrMsgTxt("Elements of catsplit cell array have to be ints (cast them to int in Matlab)");
		};
		arrayOfVariables = (int*) mxGetPr(cell_element_ptr);
		
/*		mrows = mxGetM(cell_element_ptr);
//		ncols = mxGetN(cell_element_ptr);
//		printMatrixInt(arrayOfVariables, mrows, ncols); */

		nCatLeft = mxGetNumberOfElements(cell_element_ptr);
		for (i=0; i<nCatLeft; i++){
/*			printf("categorical left; %d==%d?\n", (int) x, arrayOfVariables[i]); */
	        if (arrayOfVariables[i] == (int) x){
	            fwd_one_row(X, var, cut, mx_y_sub_cell, mx_cens_sub_cell, children, catsplit, row, numrows, numnodes, numRowsC, left_kid, mx_y_res_cell, mx_cens_res_cell);
                return;
	        }
	    }

/*		printf("idx2: %d, numRowsC =%d\n", cutoff_int-1+numRowsC, numRowsC); */
		/*=== Does the value match any of the ones going left? */
		if(!mxIsCell(catsplit)){
			mexErrMsgTxt("catsplit is not a cell array\n");
		}
		cell_element_ptr = mxGetCell(catsplit, cutoff_int-1+numRowsC);
		if (cell_element_ptr == NULL){
			mexErrMsgTxt("Empty Cell in catsplit\n");
		}
		if (mxGetClassID(cell_element_ptr) != mxINT32_CLASS){
			mexErrMsgTxt("Elements of catsplit cell array have to be ints (cast them to int in Matlab)");
		};
		arrayOfVariables = (int*) mxGetPr(cell_element_ptr);
		
/*		mrows = mxGetM(cell_element_ptr);
//		ncols = mxGetN(cell_element_ptr);
//		printMatrixInt(arrayOfVariables, mrows, ncols); */

		nCatRight = mxGetNumberOfElements(cell_element_ptr);
		for (i=0; i<nCatRight; i++){
/*			printf("categorical left; %d==%d?\n", (int) x, arrayOfVariables[i]); */
	        if (arrayOfVariables[i] == (int) x){
	            fwd_one_row(X, var, cut, mx_y_sub_cell, mx_cens_sub_cell, children, catsplit, row, numrows, numnodes, numRowsC, right_kid, mx_y_res_cell, mx_cens_res_cell);
                return;
	        }
	    }

        /* If we go neither left nor right => missing in the subtree, collect values here --- this is 1:1 the same as for the terminal case above. */
/*printf("Terminal interior node %d\n", thisnode);*/
        
        /*=== Get the y_sub array from the cell array, copy it, and put it into the result cell array y_res. */
        cell_element_ptr = mxGetCell(mx_y_sub_cell, thisnode);
        if (cell_element_ptr == NULL){
            printf("Error in terminal case %d\n", thisnode);
			mexErrMsgTxt("Empty cell in mx_y_sub_cell\n");
		}        
        if( !mxIsDouble(cell_element_ptr) || mxIsComplex(cell_element_ptr) ) {
            mexErrMsgTxt("mx_y_sub_cell must have entries that are double arrays.");
        }
        numElements = mxGetNumberOfElements(cell_element_ptr);
        y_sub = mxGetPr(cell_element_ptr);
        dims[0] = 1;
        dims[1] = numElements;
        mx_y_res = mxCreateNumericArray(2, dims, mxDOUBLE_CLASS, mxREAL);
        y_res = mxGetPr(mx_y_res);
        for(i=0; i<numElements; i++){
            y_res[i] = y_sub[i];
        }
        mxSetCell(mx_y_res_cell, row, mx_y_res);
        
        /*=== Same thing for the cens_sub array */
        cell_element_ptr = mxGetCell(mx_cens_sub_cell, thisnode);
        if( !mxIsInt32(cell_element_ptr) || mxIsComplex(cell_element_ptr) ) {
            mexErrMsgTxt("mx_cens_sub_cell must have entries that are int arrays.");
        }
        numElements2 = mxGetNumberOfElements(cell_element_ptr);
        if (numElements != numElements2){
            mexErrMsgTxt("at each node, mx_ysub and mx_censsub must the have same number of elements!\n");
        }
        cens_sub = (int*) mxGetData(cell_element_ptr);
        mx_cens_res = mxCreateNumericArray(2, dims, mxINT32_CLASS, mxREAL);
        cens_res = (int*) mxGetData(mx_cens_res);
        for(i=0; i<numElements; i++){
            cens_res[i] = cens_sub[i];
        }
        mxSetCell(mx_cens_res_cell, row, mx_cens_res);
        return;
	}
}

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


void mexFunction( int nlhs, mxArray *plhs[],
                  int nrhs, const mxArray *prhs[] )
{
  double *X,*cut, *minval, *maxval;
  int    *var, *children, mrowsX, ncolsX, mrows, ncols, mrowsT, i, mrowsC;
  int *tmp_int_ptr;
  int dims[2], numElements;
  mxArray* cell_element_ptr;
  
  /* Check for proper number of arguments. */
  if(nrhs!=7) {
    mexErrMsgTxt("Usage: fwd_cens(X, var, cut, ysub, cens, children, catsplit).");
  } else if(nlhs>2) {
    mexErrMsgTxt("Too many output arguments");
  }
  
  /* Check each argument for proper form and dimensions. */
  mrowsX = mxGetM(prhs[0]);
  ncolsX = mxGetN(prhs[0]);
  if( !mxIsDouble(prhs[0]) || mxIsComplex(prhs[0]) ) {
    mexErrMsgTxt("X must be a noncomplex double nxd matrix.");
  }
  X = mxGetPr(prhs[0]);

  mrowsT = mxGetM(prhs[1]);
  ncols = mxGetN(prhs[1]);
  if( !mxIsInt32(prhs[1]) || mxIsComplex(prhs[1]) || !(ncols==1)) {
    mexErrMsgTxt("var must be a noncomplex int Nx1 vector.");
  }
  var = (int*) mxGetPr(prhs[1]);

  mrows = mxGetM(prhs[2]);
  ncols = mxGetN(prhs[2]);
  if( !mxIsDouble(prhs[2]) || mxIsComplex(prhs[2]) || !(ncols==1) || !(mrows==mrowsT)) {
    mexErrMsgTxt("cut must be a noncomplex double Nx1 vector.");
  }
  cut = mxGetPr(prhs[2]);

  mrows = mxGetM(prhs[3]);
  ncols = mxGetN(prhs[3]);
  if( !mxIsCell(prhs[3]) || !(ncols==1) || !(mrows==mrowsT)) {
    mexErrMsgTxt("ysub must be a Nx1 cell array.");
  }
  
  mrows = mxGetM(prhs[4]);
  ncols = mxGetN(prhs[4]);
  if( !mxIsCell(prhs[4]) || !(ncols==1) || !(mrows==mrowsT)) {
    mexErrMsgTxt("cens must be a Nx1 cell array.");
  }
  
  mrows = mxGetM(prhs[5]);
  ncols = mxGetN(prhs[5]);
  if( !mxIsInt32(prhs[5]) || mxIsComplex(prhs[5]) || !(ncols==2) || !(mrows==mrowsT)) {
    mexErrMsgTxt("children must be a noncomplex int Nx2 matrix.");
  }
  children = (int*) mxGetPr(prhs[5]);
  
  mrowsC = mxGetM(prhs[6]);
/*  printf("mrowsC=%d\n",mrowsC);*/
  ncols = mxGetN(prhs[6]);
  if( !mxIsCell(prhs[6]) || !(ncols==2) ) {
    mexErrMsgTxt("catsplit must be a Mx2 cell array");
  }

  dims[0] = 1;
  dims[1] = mrowsX;
/*  printf("creating result cell arrays of size %d\n", mrowsX);*/
   
  /* Create cell arrays for the return arguments. */
  plhs[0] = mxCreateCellArray(2, dims); /*y_res*/
  plhs[1] = mxCreateCellArray(2, dims); /*cens_res*/
  
/*  printf("X[0] ... X[5] = %lf, %lf, %lf, %lf, %lf, %lf\n", X[0], X[1], X[2], X[3], X[4], X[5]);
//  printf("children[0] ... children[5] = %d, %d, %d, %d, %d, %d\n", children[0], children[1], children[2], children[3], children[4], children[5]); */
  
  /* Fill the result one entry at a time. */
  for (i=0; i<mrowsX; i++){
/*      printf("fwd for row %d\n", i);*/
    fwd_one_row(X, var, cut, prhs[3], prhs[4], children, prhs[6], i, mrowsX, mrowsT, mrowsC, 0, plhs[0], plhs[1]);
  }
}
