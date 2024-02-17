

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


int main() {

    float *A, *B;
    int size = sizeof(float) * N * N;
    A = (float*)malloc(size);
    B = (float*)malloc(size);

    for (int i = 0; i < N * N; ++i) {
        A[i] = (float)i;
        B[i] = (float)i;
    }

    // define device variables
    float *A_d, *B_d;
    // allocate memory on device
    cudaMalloc((void**)&A_d, size);
    cudaMalloc((void**)&B_d, size);

    // copy data to device
    cudaMemcpy(A_d, A, size, cudaMemcpyHostToDevice);

    printMatrix(A);
}
