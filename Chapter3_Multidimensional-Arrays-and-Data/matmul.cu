

#define N 3


#include <stdio.h>

void printMatrix(float* X) {
    for (int row = 0; row < N; ++row) {
        printf("[");
        for (int col = 0; col < N; ++col) {
            printf("%i ", (int)X[row * N + col]);
        }
        printf("]\n");
    }
}

__global__
void matmul(float* A, float* B, float* C, int n) {
    // multiply two n x n matrices
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    float sum = 0; 
    if (row < n && col < n) {
        for (int i = 0; i < n; ++i) {
            // C[row, col] = dot(A[row, :], B[:, col])
            sum = sum + A[row * n + i] * B[i * n + col];
        }
        C[row * n + col] = sum;
    }
}


int main() {

    float *A, *B, *C;
    int size = sizeof(float) * N * N;
    A = (float*)malloc(size);
    B = (float*)malloc(size);
    C = (float*)malloc(size); 

    for (int i = 0; i < N * N; ++i) {
        A[i] = (float)i;
        B[i] = (float)i;
    }

    // define device variables
    float *A_d, *B_d, *C_d;
    // allocate memory on device
    cudaMalloc((void**)&A_d, size);
    cudaMalloc((void**)&B_d, size);
    cudaMalloc((void**)&C_d, size);

    // copy data to device
    cudaMemcpy(A_d, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, A, size, cudaMemcpyHostToDevice);

    // call the multiplication kernel
    // use 16 x 16 thread blocks
    dim3 blockDim(16, 16, 1);  // 
    // (width (y == num. cols), height, depth)
    dim3 gridDim(ceil(N / (float)blockDim.y), ceil(N / (float)blockDim.x), 1);
    matmul<<<gridDim, blockDim>>>(A_d, B_d, C_d, N);

    // copy data from device back to host
    cudaMemcpy(C, C_d, size, cudaMemcpyDeviceToHost);

    // free the device memory
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);

    printf("A:\n");
    printMatrix(A);
    printf("B:\n");
    printMatrix(B);
    printf("C=AB:\n");
    printMatrix(C);
}
