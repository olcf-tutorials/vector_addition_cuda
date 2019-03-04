#Vector Addition (CUDA)
In this tutorial, we will look at a simple vector addition program, which is often used as the "Hello, World!" of GPU computing. We will assume an understanding of basic CUDA concepts, such as kernel functions and thread blocks. If you are not already familiar with such concepts, there are links at the bottom of this page that will show you where to begin.

Let's walk through the following CUDA C vector addition program:

```c
#include <stdio.h>

// Size of array
#define N 1048576

// Kernel
__global__ void add_vectors(double *a, double *b, double *c)
{
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    if(id < N) c[id] = a[id] + b[id];
}

// Main program
int main()
{
    // Number of bytes to allocate for N doubles
    size_t bytes = N*sizeof(double);

    // Allocate memory for arrays A, B, and C on host
    double *A = (double*)malloc(bytes);
    double *B = (double*)malloc(bytes);
    double *C = (double*)malloc(bytes);

    // Allocate memory for arrays d_A, d_B, and d_C on device
    double *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    // Fill host arrays A and B
    for(int i=0; i<N; i++)
    {
        A[i] = 1.0;
        B[i] = 2.0;
    }

    // Copy data from host arrays A and B to device arrays d_A and d_B
    cudaMemcpy(d_A, A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, bytes, cudaMemcpyHostToDevice);

    // Set execution configuration parameters
    //      thr_per_blk: number of CUDA threads per grid block
    //      blk_in_grid: number of blocks in grid
    int thr_per_blk = 256;
    int blk_in_grid = ceil( float(N) / thr_per_blk );

    // Launch kernel
    add_vectors<<< blk_in_grid, thr_per_blk >>>(d_A, d_B, d_C);

    // Copy data from device array d_C to host array C
    cudaMemcpy(C, d_C, bytes, cudaMemcpyDeviceToHost);

    // Verify results
    for(int i=0; i<N; i++)
    {
        if(C[i] != 3.0)
        {
            printf("\nError: value of C[%d] = %d instead of 3.0\n\n", i, C[i]);
            exit(-1);
        }
    }

    // Free CPU memory
    free(A);
    free(B);
    free(C);

    // Free GPU memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    printf("\n---------------------------\n");
    printf("__SUCCESS__\n");
    printf("---------------------------\n");
    printf("N                 = %d\n", N);
    printf("Threads Per Block = %d\n", thr_per_blk);
    printf("Blocks In Grid    = %d\n", blk_in_grid);
    printf("---------------------------\n\n");

    return 0;
}
```

We'll ignore the `add_vectors` "kernel" function for the moment and jump down to the main function. Here, we first determine the number of bytes of memory required for an array with N double precision elements:

```c
    // Number of bytes to allocate for N doubles
    size_t bytes = N*sizeof(double);
```

Now we allocate memory on the CPU for arrays A, B and C:

```c
    // Allocate memory for arrays A, B, and C on host
    double *A = (double*)malloc(bytes);
    double *B = (double*)malloc(bytes);
    double *C = (double*)malloc(bytes);
```

Now we allocate memory on the GPU for arrays d\_A, d\_B, and d\_C (these variables are prefixed with "d_" to distinguish them as "device variables", but they can be named differently):

```c
    // Allocate memory for arrays d_A, d_B, and d_C on device
    double *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);
```

Here, we fill the arrays A and B on the CPU:

```c
    // Fill host arrays A and B
    for(int i=0; i<N; i++)
    {
        A[i] = 1.0;
        B[i] = 2.0;
    }
```

Now we set the execution configuration parameters in preparation for launching the kernel function on the GPU. The `ceil` funciton is needed to ensure there are enough blocks in the grid to cover all N elements of the arrays.

```c
    // Set execution configuration parameters
    //      thr_per_blk: number of CUDA threads per grid block
    //      blk_in_grid: number of blocks in grid
    int thr_per_blk = 256;
    int blk_in_grid = ceil( float(N) / thr_per_blk );
```

Now we launch the kernel function on the GPU:

```c
    // Launch kernel
    add_vectors<<< blk_in_grid, thr_per_blk >>>(d_A, d_B, d_C);
```

This causes execution to jump up to the `add_vectors` kernel function (defined before main). The variable `id` is used to define a unique thread ID among all threads in the grid. The `if` statement ensures that we do not perform an element-wise addition on an out-of-bounds array element. In this program, `blk_in_grid` equals 4096, but if `thr_per_blk` did not divide evenly into `N`, the `ceil` function would increase `blk_in_grid` by 1. For example, if `N` had 1 extra element, `blk_in_grid` would be 4097, which would mean a total of 4097 * 256 = 1048832 threads. So without the `if` statement, element-wise additions would be calculated for elements that we have not allocated memory for.

```c
// Kernel
__global__ void add_vectors(double *a, double *b, double *c)
{
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    if(id < N) c[id] = a[id] + b[id];
}
```

Now that the results of the vection addition have been stored in array d\_C on the GPU, we can copy the results back to the CPU copy of C:

```c
    // Copy data from device array d_C to host array C
    cudaMemcpy(C, d_C, bytes, cudaMemcpyDeviceToHost);
```

Now we check to make sure we received the correct results from the GPU:

```c
    // Verify results
    for(int i=0; i<N; i++)
    {
        if(C[i] != 3.0)
        {
            printf("\nError: value of C[%d] = %d instead of 3.0\n\n", i, C[i]);
            exit(-1);
        }
    }
```

And finally, we can free memory on the CPU:

```c
    // Free CPU memory
    free(A);
    free(B);
    free(C);
```

And free the memory on the GPU:

```c
    // Free GPU memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
```

##Helpful Links

CUDA C Programming Model: <a href="https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#programming-model">https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#programming-model</a>

##Problems?
If you see a problem with the code or have suggestions to improve it, feel free to open an issue.