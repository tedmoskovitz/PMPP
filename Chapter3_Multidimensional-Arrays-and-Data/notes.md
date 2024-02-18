# Chapter 3: Multidimensional Grids and Data

- As you could guess from the title, this chapter is about multidimensional grids and data -- the main new ``thing'' from the previous chapter is launching grids/kernels which use multidimensional data
- For example, you might have a 2D grid of blocks, each containing a 3D array of threads:
![[figs/ch3_block-grid.png]]
- Basically, you use N dims depending on the dimensionality of your data, so you can easily compute columns, rows, etc. (because inside a kernel, you're using one thread, so you want the columns and rows, etc. of the block/grid of threads you launched to corresponds to columns/rows of the data you're operating on, e.g., pixels or matrix entries)
- you specify these using a special 3D vector datatype, `dim3`, e.g., for 2D data:
```c
int N = 50; // say, a 50 x 50 matrix
dim3 blockDim(16, 16, 1); // use 16 x 16 blocks of threads
//ceil(50/16) = 4 -> 64-50=14 extra threads
int nGridRows = ceil(N / (float)blockDim.y);
int nGridCols = ceil(N / (float)blockDim.x); // 
dim3 gridDim(nGridCols, nGridRows, 1);
```
- In memory, matrices are stored in row-major order (as in, you take all the rows of a matrix and lay them out in a line)
- We can now also implement a simple matmul kernel (see `matmul.cu`)
- matrix multiplication is a "Level 3" operation in the Basic Linear Algebra Subprogram (BLAS) standard for linear algebra operations (forming the basis for many more complicated lienar algebra operations); Level 1 = vector-vector operations of the form $a\mathbf{x} + y$, Level 2 = matrix-vector operations of the form $a\mathbf{Ax} + b\mathbf{y}$, and Level 3 = matrix-matrix operations of the form $a\mathbf{AB} + b\mathbf{C}$
- one important thing to note: the size of the matrices than can be multiplied is limited by the maximum threads per block and grid dimensions, so if you're multiplying bigger matrices, you need to spread the computation across multiple grids (something that's covered later)