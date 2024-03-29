// compilation: nvcc vector_addition.cu -o vector_addition
// usage: ./vector_addition
#include <stdio.h>
#include <time.h>

#define N 100000000

void vecAddRegular(float* A_h, float* B_h, float* C_h, int n) {
    for (int i = 0; i < n; i++) {
        C_h[i] = A_h[i] + B_h[i];
    }
}

__global__
void addKernel(float* A, float* B, float* C, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        C[i] = A[i] + B[i];
    }
}

void vecAdd(float* A_h, float* B_h, float* C_h, int n) {
    int size = sizeof(float) * n; 
    // on-device versions
    float *A_d, *B_d, *C_d; 
    // allocate memory on device
    cudaMalloc((void**)&A_d, size); 
    cudaMalloc((void**)&B_d, size);
    cudaMalloc((void**)&C_d, size);
    // copy data from host to device
    cudaMemcpy(A_d, A_h, size, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B_h, size, cudaMemcpyHostToDevice);

    addKernel<<<ceil(n / 256.0), 256>>>(A_d, B_d, C_d, n);

    // copy result back to host
    cudaMemcpy(C_h, C_d, size, cudaMemcpyDeviceToHost);

    // free on-device memory
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
}

int main() {

    // Just do regular C here
    float *A, *B, *C;

    // allocate memory -- note regular C
    // malloc only takes size as input
    int size = sizeof(float) * N;
    A = (float*)malloc(size); 
    B = (float*)malloc(size);
    C = (float*)malloc(size);

    // initialize
    for (int i = 0; i < N; i++) {
        A[i] = 1.0f;
        B[i] = 2.0f;
    }

    // run regular addition
    clock_t start = clock();
    vecAddRegular(A, B, C, N); 
    // Record the end time
    clock_t end = clock();
    // Calculate the elapsed time in seconds
    double elapsed_time = (double)(end - start) / CLOCKS_PER_SEC;
    printf("Regular vector addition: %.3f seconds\n", elapsed_time);


    // now do the parallel / on-device version
    clock_t startp = clock();
    vecAdd(A, B, C, N); 
    // Record the end time
    clock_t endp = clock();
    // Calculate the elapsed time in seconds
    double elapsed_timep = (double)(endp - startp) / CLOCKS_PER_SEC;
    printf("Parallel vector addition: %.3f seconds\n", elapsed_timep);
    // this will actually probably take longer than the serial version;
    // there's too much overhead relative to the amount of computation 
    // happening (overhead wrt allocating/de-allocating device memory
    // and transferring data to and from device)

}


