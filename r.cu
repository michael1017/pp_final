#include <math.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define OFFSET(x, y, m) (((x)*(m)) + (y))
__host__ void initialize(double * A, int m, int n);
__global__ void calcNext(double * A, double * Anew, double *device_error, int m, int n);
__global__ void swap(double * A, double * Anew, int m, int n);
__host__ void deallocate(double * A, double * Anew);
int main(int argc, char** argv)
{
    // n = height, m = width
    // width for y direction
    // height for x direction
    const int n = 4096;
    const int m = 4096;
    const int iter_max = 1000;
    
    const double tol = 1.0e-6;
    double error = 0.0;
    
    const int block_width = 32;
    const int block_height = 32;
    const int grid_width = m/block_width;
    const int grid_height = n/block_height;

    dim3 grid(grid_height, grid_width);
    dim3 block(block_height, block_width);

    double * host_A    = (double*)malloc(sizeof(double)*n*m);
    double * A, * Anew, *device_error;
    cudaMalloc(&A, sizeof(double)*n*m);
    cudaMalloc(&Anew, sizeof(double)*n*m);
    cudaMalloc(&device_error, sizeof(double));

    initialize(host_A, m, n);
    
    cudaMemcpy(A, host_A, sizeof(double)*n*m, cudaMemcpyHostToDevice);
    cudaMemcpy(Anew, host_A, sizeof(double)*n*m, cudaMemcpyHostToDevice);
    cudaMemcpy(device_error, &error, sizeof(double), cudaMemcpyHostToDevice);
    
    printf("Jacobi relaxation Calculation: %d x %d mesh\n", n, m);
    
    double st = omp_get_wtime();
    int iter = 0;
    error = 1.0;
    while ( error > tol && iter < iter_max )
    {
        //DONE fix error return value
        //DONE fix swap function 
        calcNext<<<grid, block>>>(A, Anew, device_error, m, n);
        swap<<<grid, block>>>(A, Anew, m, n);
        cudaMemcpy(&error, device_error, sizeof(double), cudaMemcpyDeviceToHost);
        if(iter % 100 == 0) 
            printf("%5d, %0.6f\n", iter, error);
        iter++;
    }

    double runtime = omp_get_wtime() - st;
    printf(" total: %f s\n", runtime);
    deallocate(A, Anew);
    return 0;
}

// must do it on cpu
__host__ inline void initialize(double * A, int m, int n)
{
    memset(A, 0, n * m * sizeof(double));
    int i;
    for(i = 0; i < m; i++){
        A[i] = 1.0;
    }
}

// main time comsumer
// TODO: Find max error in all threads
// Done: Handle memory error problem
__global__ void calcNext(double * A, double * Anew, double *device_error, int m, int n)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int s;
    int local_i = i + 1;
    int local_j = j + 1;
    int local_m = blockDim.y + 2;
    int tid = OFFSET(local_j, local_i, local_m);
    __shared__ double sm[34*34];
    __shared__ double outcome[34*34];
    //__shared__ double local_max;

    //device_error[0] = 3;
    
    if (1 <= j && j < n-1 && 1 <= i && i < m-1) {
        sm[OFFSET(local_j-1, local_i-1, local_m)] = A[OFFSET(j-1, i-1, m)];
        sm[OFFSET(local_j-1, local_i+1, local_m)] = A[OFFSET(j-1, i+1, m)];
        sm[OFFSET(local_j+1, local_i-1, local_m)] = A[OFFSET(j+1, i-1, m)];
        sm[OFFSET(local_j+1, local_i+1, local_m)] = A[OFFSET(j+1, i+1, m)];
        __syncthreads();

        outcome[OFFSET(local_j, local_i, local_m)] = 0.25 * ( sm[OFFSET(local_j, local_i-1, local_m)] + sm[OFFSET(local_j, local_i+1, local_m)] + sm[OFFSET(local_j-1, local_i, local_m)] + sm[OFFSET(local_j+1, local_i, local_m)]);
        Anew[OFFSET(j, i, m)] = outcome[OFFSET(local_j, local_i, local_m)];
        //use same variable to save memory usage
        outcome[OFFSET(local_j, local_i, local_m)] = fabs(outcome[OFFSET(local_j, local_i, local_m)] - sm[OFFSET(local_j, local_i, local_m)]);
        
        for (s=blockDim.x*blockDim.y/2; s>0; s>>=1) 
        {
            if (tid < s)
            outcome[tid] = max(outcome[tid], outcome[tid + s]);  // 2
            __syncthreads();
        }
        // what to do now?
        // option 1: save block result and launch another kernel
        if (tid == 0)        
            device_error[0] = 3;
            //device_error[0] = fmax(device_error[0], outcome[tid]); // 3

    }
    //device_error[0] = fmax(device_error[0], device_error[0]);
    device_error[0] = 3;
}

// Not efficiency here        
__global__ void swap(double * A, double * Anew, int m, int n)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (1 <= j && j < n-1 && 1 <= i && i < m-1) {
        A[OFFSET(j, i, m)] = Anew[OFFSET(j, i, m)];
    }
}

__host__ void deallocate(double * A, double * Anew)
{
    cudaFree(A);
    cudaFree(Anew);
}
