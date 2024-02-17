// simple image transformations: RGB -> grayscale + uniform blurring
// compilation: nvcc image_transforms.cu -o image_transforms
// launch: ./image_transforms

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


__global__
void blurImage(unsigned char * Im, int height, int width, int blurRadius) {
    // blur an image via averaging the in-bounds pixels within a radius
    // equivalent to convolution with a kernel of all 1s
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int numPixels = 0;
    int pixelSum = 0;
    for (int i = row - blurRadius; i < row + blurRadius + 1; ++i) {
        for (int j = col - blurRadius; j < col + blurRadius + 1; ++j) {
            if (i >= 0 && j >= 0 && i < height && j < width) {
                pixelSum += Im[i * width + j];
                numPixels++; 
            }
        }
    }
    Im[row * width + col] = (unsigned char)(pixelSum / numPixels);
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

    // call RGB -> grayscale kernel
    dim3 dimBlock(16, 16, 1);
    // make enough 16 x 16 blocks of threads to cover the image dimensions
    dim3 dimGrid(ceil(width / (float)dimBlock.x), ceil(height / (float)dimBlock.y), 1);
    colorToGrayscale<<<dimGrid, dimBlock>>>(rgb_image_d, gray_image_d, height, width);

    // call blurring kernel
    int blurRadius = 1;
    blurImage<<<dimGrid, dimBlock>>>(gray_image_d, height, width, blurRadius);

    // copy data back from device to host
    cudaMemcpy(gray_image, gray_image_d, size, cudaMemcpyDeviceToHost);

    // free device memory
    cudaFree(rgb_image_d);
    cudaFree(gray_image_d);

}


