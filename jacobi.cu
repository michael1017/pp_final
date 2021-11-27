#include <math.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <cuda.h>
#include <cuda_device_runtime_api.h>
#define OFFSET(x, y, m) (((x)*(m)) + (y))
__host__ void initialize(double * A, int m, int n);
__global__ void calcNext(double * A, double *device_error, int m, int n);
__host__ void deallocate(double * A, double * device_error);
__global__ void find_max(double *device_error, double *max_error);
int main(int argc, char** argv)
{
    // n = height, m = width
    // width for y direction
    // height for x direction
    const int n = 4096;
    const int m = 4096;
    const int iter_max = 1000;
    
    const double tol = 1.0e-6;
    double error = 1.0;
    
    const int block_width = 32;
    const int block_height = 32;
    const int grid_width = m/block_width;
    const int grid_height = n/block_height;

    dim3 grid(grid_height, grid_width);
    dim3 block(block_height, block_width);

    double * host_A    = (double*)malloc(sizeof(double)*n*m);
    double * host_err  = (double*)malloc(sizeof(double));
    double * A, *device_error, *max_error;
    cudaMalloc(&A, sizeof(double)*n*m);
    cudaMalloc(&device_error, sizeof(double)*grid_height*grid_width);
    cudaMalloc(&max_error, sizeof(double));
    initialize(host_A, m, n);
    
    printf("Jacobi relaxation Calculation: %d x %d mesh\n", n, m);
    
    double st = omp_get_wtime();
    cudaMemcpy(A, host_A, sizeof(double)*n*m, cudaMemcpyHostToDevice);
    int iter = 0;
    while ( error > tol && iter < iter_max )
    {
        calcNext<<<grid, block>>>(A, device_error, m, n);
        
        if(iter % 100 == 0){
            find_max<<<1, block>>>(device_error, max_error);
            cudaMemcpy(host_err, max_error, sizeof(double), cudaMemcpyDeviceToHost);
            printf("%5d, %0.6f\n", iter, *host_err);
        }
            
        iter++;
    }

    double runtime = omp_get_wtime() - st;
    printf(" total: %f s\n", runtime);
    deallocate(A, device_error);
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
__global__ void calcNext(double * A, double *device_error, int m, int n)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int s;
    int local_i = threadIdx.y + 1;
    int local_j = threadIdx.x + 1;
    int local_m = blockDim.y + 2;
    int tid = OFFSET(local_j-1, local_i-1, 32);
    double outcome;
    __shared__ double sm[34*34];
    __shared__ double error[32*32];

    //device_error[0] = 3;
    error[tid] = 0.0;
    if (1 <= j && j < n-1 && 1 <= i && i < m-1) {
        sm[OFFSET(local_j-1, local_i-1, local_m)] = A[OFFSET(j-1, i-1, m)];
        sm[OFFSET(local_j-1, local_i+1, local_m)] = A[OFFSET(j-1, i+1, m)];
        sm[OFFSET(local_j+1, local_i-1, local_m)] = A[OFFSET(j+1, i-1, m)];
        sm[OFFSET(local_j+1, local_i+1, local_m)] = A[OFFSET(j+1, i+1, m)];
    }
    __syncthreads();
    if (1 <= j && j < n-1 && 1 <= i && i < m-1) {
        outcome = 0.25 * (sm[OFFSET(local_j, local_i-1, local_m)] + sm[OFFSET(local_j, local_i+1, local_m)] + sm[OFFSET(local_j-1, local_i, local_m)] + sm[OFFSET(local_j+1, local_i, local_m)]);
        A[OFFSET(j, i, m)] = outcome;
        error[tid] = fabs(outcome - sm[OFFSET(local_j, local_i, local_m)]);
    }
    __syncthreads();
    for (s=((blockDim.x*blockDim.y)*0.5); s>0; s>>=1) 
    {
        if (tid < s)
            error[tid] = fmax(error[tid], error[tid + s]);  // 2
        __syncthreads();
    }
    if (tid == 0){
        device_error[blockIdx.x*gridDim.y + blockIdx.y] = error[tid];
    }
}
__host__ void deallocate(double * A, double * device_error)
{
    cudaFree(A);
    cudaFree(device_error);
}
__global__ void find_max(double *device_error, double *max_error) {
    __shared__ double sm[64*64];
    __shared__ double sm2[32*32];
    int s, x, y;
    x = threadIdx.x; y = threadIdx.y;
    sm[OFFSET(x, y, 64)] = fmax(fmax(device_error[OFFSET(x, y, 128)], device_error[OFFSET(x, y+64, 128)]),
        fmax(device_error[OFFSET(x+64, y, 128)], device_error[OFFSET(x+64, y+64, 128)]));
    
    x = threadIdx.x; y = threadIdx.y+32;
    sm[OFFSET(x, y, 64)] = fmax(fmax(device_error[OFFSET(x, y, 128)], device_error[OFFSET(x, y+64, 128)]),
        fmax(device_error[OFFSET(x+64, y, 128)], device_error[OFFSET(x+64, y+64, 128)]));

    x = threadIdx.x+32; y = threadIdx.y;
    sm[OFFSET(x, y, 64)] = fmax(fmax(device_error[OFFSET(x, y, 128)], device_error[OFFSET(x, y+64, 128)]),
        fmax(device_error[OFFSET(x+64, y, 128)], device_error[OFFSET(x+64, y+64, 128)]));

    x = threadIdx.x+32; y = threadIdx.y+32;    
    sm[OFFSET(x, y, 64)] = fmax(fmax(device_error[OFFSET(x, y, 128)], device_error[OFFSET(x, y+64, 128)]),
        fmax(device_error[OFFSET(x+64, y, 128)], device_error[OFFSET(x+64, y+64, 128)]));
    __syncthreads();
    sm2[OFFSET(threadIdx.x, threadIdx.y, 32)] = fmax(fmax(sm[OFFSET(threadIdx.x, threadIdx.y, 64)], sm[OFFSET(threadIdx.x, threadIdx.y+32, 64)])
        , fmax(sm[OFFSET(threadIdx.x+32, threadIdx.y, 64)], sm[OFFSET(threadIdx.x+32, threadIdx.y+32, 64)]));
    __syncthreads();
    int tid = OFFSET(threadIdx.x, threadIdx.y, 32);
    for (s=512; s>0; s>>=1) 
    {
        if (tid < s)
            sm2[tid] = fmax(sm2[tid], sm2[tid + s]);
        __syncthreads();
    }
    if (tid == 0){
        *max_error = sm2[tid];
    }
}