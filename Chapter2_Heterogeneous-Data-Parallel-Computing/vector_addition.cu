// compilation: nvcc vector_addition.cu -o vector_addition
// usage: ./vector_addition

#define N 1000

void vecAddRegular(float* A_h, float* B_h, float* C_h, int n) {
    for (int i = 0; i < n; i++) {
        C_h[i] = A_h[i] + B_h[i];
    }
}

int main() {

    // Just do regular C here
    float *A, *B, *C;

    // allocate memory -- note regular C
    // malloc only takes size as input
    A = (float*)malloc(sizeof(float) * N); 
    B = (float*)malloc(sizeof(float) * N);
    C = (float*)malloc(sizeof(float) * N);

    // initialize
    for (int i = 0; i < N; i++) {
        A[i] = 1.0f;
        B[i] = 2.0f;
    }

    // run addition
    vecAddRegular(A, B, C, N); 

}


