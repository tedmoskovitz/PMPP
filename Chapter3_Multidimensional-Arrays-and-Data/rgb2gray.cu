#define N_CHANNELS 3


__global__
void colorToGrayscale(unsigned char * Im_in, unsigned char * Im_out, int height, int width) {
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    if (col < width && row < height) {
        int pixelOffset = row * width + col; 
        // think of the rgb image as haveing N_CHANNELS more columns than grayscale
        int rgbOffset = N_CHANNELS * pixelOffset; 
        unsigned char r = Im_in[rgbOffset];
        unsigned char g = Im_in[rgbOffset + 1];
        unsigned char b = Im_in[rgbOffset + 2];
        Im_out[pixelOffset] = 0.21 * r + 0.71 * g + 0.07 * b; 
    }
}



int main() {

    int height = 128;
    int width = 128;
    int n_pixels = height * width; 
    int size = sizeof(unsigned char) * n_pixels; 

    unsigned char *rgb_image, *gray_image; 
    // allocate host memory
    rgb_image = (unsigned char*)malloc(size * 3); 
    gray_image = (unsigned char*)malloc(size); 

    // initialize
    for (int i = 0; i < 3 * size; ++i) {
        rgb_image[i] = 76;
    }

    // allocate device memory
    unsigned char *rgb_image_d, *gray_image_d;
    cudaMalloc((void**)&rgb_image_d, size * 3);
    cudaMalloc((void**)&gray_image_d, size);

    // copy data to device
    cudaMemcpy(rgb_image_d, rgb_image, size * 3, cudaMemcpyHostToDevice);
    cudaMemcpy(gray_image_d, gray_image, size, cudaMemcpyHostToDevice);

    // call kernel
    int block_size = 256;
    int n_blocks = ceil(size / (float)block_size); 
    colorToGrayscale<<<n_blocks, block_size>>>(rgb_image_d, gray_image_d, height, width);

    // copy data back from device to host
    cudaMemcpy(gray_image, gray_image_d, size, cudaMemcpyDeviceToHost);

    // free device memory
    cudaFree(rgb_image_d);
    cudaFree(gray_image_d);

}


