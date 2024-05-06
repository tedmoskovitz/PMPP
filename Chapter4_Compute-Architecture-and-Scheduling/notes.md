# Chapter 4: Computer Architecture and Scheduling

## 4.1 Architecture of a modern GPU
![](figs/ch4_arch.png)
- above is a high-level view of a CUDA-capable GPU
- it's organized into an array of highly threaded *streaming multiprocessors* (SMs), each of which contains a bunch of processing units called *CUDA cores* or just *cores* (the green blocks)
- all cores within an SM share control logic and memory resources
- e.g., an A100 has 108 SMs with 64 cores each
- In addition to local memory on the SM, the GPU has a global off-chip device memory with tight integration with the SMs---this is the DRAM, usually *high-bandwidth memory* (HBM) in the newest GPUs


## 4.2 Block scheduling

![](figs/ch4_block-assignment.png)

- when a kernel is called, the CUDA runtime system launches a grid of threads divided into blocks
- all threads within a block are always scheduled simultaneously on the same SM
- often the grid contains more blocks than can simultaneously execute across SMs, so the runtime system tracks which are running and which still need to be assigned
- because threads within a block are launched together and share hardware resources, it's relatively easy to facilitate their interaction via *synchronization* (4.3) and/or *shared memory* (ch. 5) 
