
**2.1 Data Parallelism**
- this is when you apply the same program/instruction to separate pieces of data independently 
- example: RGB -> grayscale pixel conversion (each pixel is independent of the others)
![](figs/ch2_rgb.png)
- *task parallelism*: another type of parallelism in which you apply instructions in parallel (shocker) --- e.g., you need to a vector addition and a matrix-vector product but one doesn't have to wait for the other to be completed first

**2.2 CUDA C Program Structure**
- CUDA C is just the base language with extra stuff tacked on (also starting to have more C++ features too)
- *host* = CPU, *device* = GPU
- regular C is just CUDA with only host code
- you can add device code with *kernels,* which place data on device and launch a bunch of parallel threads --- the threads launched by a kernel are called a *grid*
![](figs/ch2_host-device.png)
- above is example program, where things start on host, then a kernel launches a grid of threads, then the parallel part is over and control returns to host, etc. in more complicated programs, often have code running on host and device simultaneously
- in the RGB -> grayscale example, the no. of threads would be equal to the no. of pixels in the image