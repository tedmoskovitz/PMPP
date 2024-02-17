# Chapter 3: Multidimensional Grids and Data

- As you could guess from the title, this chapter is about multidimensional grids and data -- the main new ``thing'' from the previous chapter is launching grids/kernels which use multidimensional data
- For example, you might have a 2D grid of blocks, each containing a 3D array of threads:
![[figs/ch3_block-grid.png]]
- Basically, you use N dims depending on the dimensionality of your data, so you can easily compute columns, rows, etc. (because inside a kernel, you're using one thread, so you want the columns and rows, etc. of the block/grid of threads you launched to corresponds to columns/rows of the data you're operating on, e.g., pixels or matrix entries)
- you specify these using a special 3D vector datatype, `dim3`, e.g., 
```c
int N = 50; // say, a 50 x 50 matrix
dim3 blockDim(16, 16, 1); // use 16 x 16 blocks of threads
int nGridRows = ceil(N / (float)blockDim.y);
int nGridCols = ceil(N / (float)blockDim.x);
dim3 gridDim(nGridCols, nGridRows, 1);
```
- In memory, matrices are stored in row-major order (as in, you take all the rows of a matrix and lay them out in a line)