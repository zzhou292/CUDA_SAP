#include <Eigen/Dense>
#include <iostream>
#include <chrono>
#include <cuda_runtime.h>

// helper function to check CUDA errors
#define CUDA_CHECK_RETURN(value)                                            \
    {                                                                       \
        cudaError_t _m_cudaStat = value;                                    \
        if (_m_cudaStat != cudaSuccess)                                     \
        {                                                                   \
            std::cerr << "Error " << cudaGetErrorString(_m_cudaStat)        \
                      << " at line " << __LINE__ << " in file " << __FILE__ \
                      << std::endl;                                         \
            exit(1);                                                        \
        }                                                                   \
    }

// Util Func 1 - matrix multiplication using global memory
__global__ void matrixMultiplyKernel(double *A, double *B, double *C, int M, int N, int K)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < K)
    {
        double sum = 0.0;
        for (int i = 0; i < N; ++i)
        {
            sum += A[row + i * M] * B[i + col * N];
        }
        C[row + col * M] = sum;
    }
}

void matrixMultiply(const Eigen::MatrixXd &A, const Eigen::MatrixXd &B, Eigen::MatrixXd &C)
{
    int M = A.rows();
    int N = A.cols();
    int K = B.cols();

    double *d_A;
    double *d_B;
    double *d_C;

    size_t size_A = M * N * sizeof(double);
    size_t size_B = N * K * sizeof(double);
    size_t size_C = M * K * sizeof(double);

    // Allocate device memory
    CUDA_CHECK_RETURN(cudaMalloc((void **)&d_A, size_A));
    CUDA_CHECK_RETURN(cudaMalloc((void **)&d_B, size_B));
    CUDA_CHECK_RETURN(cudaMalloc((void **)&d_C, size_C));

    // Copy data to device
    CUDA_CHECK_RETURN(cudaMemcpy(d_A, A.data(), size_A, cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(d_B, B.data(), size_B, cudaMemcpyHostToDevice));

    // Define block and grid sizes
    dim3 blockSize(16, 16);
    dim3 gridSize((K + blockSize.x - 1) / blockSize.x, (M + blockSize.y - 1) / blockSize.y);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    // Launch kernel
    matrixMultiplyKernel<<<gridSize, blockSize>>>(d_A, d_B, d_C, M, N, K);
    cudaDeviceSynchronize();

    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    float milliseconds = 0.0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    std::cout << "Elapsed time for matrixMultiplyKernel: " << milliseconds << " ms\n";

    // Copy result back to host
    CUDA_CHECK_RETURN(cudaMemcpy(C.data(), d_C, size_C, cudaMemcpyDeviceToHost));

    // Free device memory
    CUDA_CHECK_RETURN(cudaFree(d_A));
    CUDA_CHECK_RETURN(cudaFree(d_B));
    CUDA_CHECK_RETURN(cudaFree(d_C));
}

// Util Func 2 - tiled matrix multiplication using shared memory

#define TILE_WIDTH 16

__global__ void TiledMatrixMulKernel(const double *M, const double *N, double *P, const int M_rows, const int M_cols, const int N_cols)
{
    __shared__ double Mds[TILE_WIDTH][TILE_WIDTH];
    __shared__ double Nds[TILE_WIDTH][TILE_WIDTH];

    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int row = by * TILE_WIDTH + ty;
    const int column = bx * TILE_WIDTH + tx;

    double Pvalue = 0.0;
    for (int ph = 0; ph < (M_cols + TILE_WIDTH - 1) / TILE_WIDTH; ++ph)
    {
        if (row < M_rows && ph * TILE_WIDTH + tx < M_cols)
        {
            Mds[ty][tx] = M[row + (ph * TILE_WIDTH + tx) * M_rows];
        }
        else
        {
            Mds[ty][tx] = 0.0;
        }
        if (ph * TILE_WIDTH + ty < M_cols && column < N_cols)
        {
            Nds[ty][tx] = N[(ph * TILE_WIDTH + ty) + column * M_cols];
        }
        else
        {
            Nds[ty][tx] = 0.0;
        }
        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; ++k)
        {
            Pvalue += Mds[ty][k] * Nds[k][tx];
        }
        __syncthreads();
    }
    if (row < M_rows && column < N_cols)
    {
        P[row + column * M_rows] = Pvalue;
    }
}

void tiledMatrixMultiply(const Eigen::MatrixXd &A, const Eigen::MatrixXd &B, Eigen::MatrixXd &C)
{
    int M_rows = A.rows();
    int M_cols = A.cols();
    int N_cols = B.cols();
    assert(M_cols == B.rows());

    size_t size_A = M_rows * M_cols * sizeof(double);
    size_t size_B = M_cols * N_cols * sizeof(double);
    size_t size_C = M_rows * N_cols * sizeof(double);

    double *A_d, *B_d, *C_d;

    // Allocate device memory
    cudaMalloc((void **)&A_d, size_A);
    cudaMalloc((void **)&B_d, size_B);
    cudaMalloc((void **)&C_d, size_C);

    // Copy data to device
    cudaMemcpy(A_d, A.data(), size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B.data(), size_B, cudaMemcpyHostToDevice);

    const dim3 dimGrid((N_cols + TILE_WIDTH - 1) / TILE_WIDTH, (M_rows + TILE_WIDTH - 1) / TILE_WIDTH);
    const dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    TiledMatrixMulKernel<<<dimGrid, dimBlock>>>(A_d, B_d, C_d, M_rows, M_cols, N_cols);
    cudaDeviceSynchronize();
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    float milliseconds = 0.0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    std::cout << "Elapsed time for TiledMatrixMulKernel: " << milliseconds << " ms\n";

    // Copy result back to host
    cudaMemcpy(C.data(), C_d, size_C, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

// Util Func 3 - choleskyDecomposition

__global__ void choleskyKernel(double *A, double *L, int n, int offset)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x + offset;
    if (j >= n)
        return;

    for (int i = 0; i <= j; ++i)
    {
        __syncthreads();

        if (i == j)
        {
            double sum = 0.0;
            for (int k = 0; k < i; ++k)
            {
                sum += L[i + k * n] * L[i + k * n];
            }
            L[i + i * n] = sqrt(A[i + i * n] - sum);
        }
        __syncthreads();

        if (j > i)
        {
            double sum = 0.0;
            for (int k = 0; k < i; ++k)
            {
                sum += L[j + k * n] * L[i + k * n];
            }
            L[j + i * n] = (A[j + i * n] - sum) / L[i + i * n];
        }
    }
}

void choleskyDecomposition(Eigen::MatrixXd &A, Eigen::MatrixXd &L)
{
    int n = A.rows();
    if (n != A.cols())
    {
        std::cerr << "Matrix must be square!" << std::endl;
        exit(1);
    }

    double *d_L;
    double *d_A;
    size_t size = n * n * sizeof(double);

    // Allocate device memory
    CUDA_CHECK_RETURN(cudaMalloc((void **)&d_L, size));
    CUDA_CHECK_RETURN(cudaMalloc((void **)&d_A, size));

    // Copy data to device
    CUDA_CHECK_RETURN(cudaMemcpy(d_L, L.data(), size, cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(d_A, A.data(), size, cudaMemcpyHostToDevice));

    // Define block and grid sizes
    int blockSize = 64;
    int gridSize = (n + blockSize - 1) / blockSize;

    int offset = 0;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    while (offset <= n)
    {
        // Launch kernel
        choleskyKernel<<<gridSize, blockSize>>>(d_A, d_L, n, offset);
        CUDA_CHECK_RETURN(cudaDeviceSynchronize());
        offset += blockSize;
    }

    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    float milliseconds = 0.0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    std::cout << "Elapsed time for cholesky size " << n << " elapsed: " << milliseconds << " ms\n";

    // Copy result back to host
    CUDA_CHECK_RETURN(cudaMemcpy(L.data(), d_L, size, cudaMemcpyDeviceToHost));

    // Free device memory
    CUDA_CHECK_RETURN(cudaFree(d_L));
    CUDA_CHECK_RETURN(cudaFree(d_A));
}

int main()
{
    // Benchcase 1
    std::cout << "===================== BENCH CASE 1 ========================" << std::endl;
    // Initialize Eigen matrices
    Eigen::MatrixXd A = Eigen::MatrixXd::Random(1024, 500);
    Eigen::MatrixXd B = Eigen::MatrixXd::Random(500, 300);
    Eigen::MatrixXd C_t1(1024, 300); // matrix multiplication res
    Eigen::MatrixXd C_t2(1024, 300); // tiled matrix multiplication res
    Eigen::MatrixXd C_t3(1024, 300); // validation res

    // Time the Eigen matrix multiplication for C_t3 in milliseconds
    auto start = std::chrono::high_resolution_clock::now();
    C_t3 = A * B;
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;

    std::cout << "Elapsed time for Eigen mat mul CPU: " << elapsed.count() << " ms\n";

    // Perform matrix multiplication
    matrixMultiply(A, B, C_t1);
    tiledMatrixMultiply(A, B, C_t2);

    std::cout << "err 1: " << (C_t3 - C_t1).norm() << std::endl;
    std::cout << "err 2: " << (C_t3 - C_t2).norm() << std::endl;

    // Benchcase 2
    std::cout << "===================== BENCH CASE 2 ========================" << std::endl;
    // Initialize a positive-definite matrix
    Eigen::MatrixXd A_2_1 = Eigen::MatrixXd::Random(128, 128);
    A_2_1 = A_2_1.transpose() * A_2_1; // Make it symmetric positive-definite

    Eigen::MatrixXd A_2_2 = Eigen::MatrixXd::Random(256, 256);
    A_2_2 = A_2_2.transpose() * A_2_2; // Make it symmetric positive-definite

    Eigen::MatrixXd A_2_3 = Eigen::MatrixXd::Random(512, 512);
    A_2_3 = A_2_3.transpose() * A_2_3; // Make it symmetric positive-definite

    Eigen::MatrixXd A_2_4 = Eigen::MatrixXd::Random(1024, 1024);
    A_2_4 = A_2_4.transpose() * A_2_4; // Make it symmetric positive-definite

    // Output matrix
    Eigen::MatrixXd L_1(128, 128);
    L_1.setZero();

    Eigen::MatrixXd L_2(256, 256);
    L_2.setZero();

    Eigen::MatrixXd L_3(512, 512);
    L_3.setZero();

    Eigen::MatrixXd L_4(1024, 1024);
    L_4.setZero();

    // Perform Cholesky decomposition
    choleskyDecomposition(A_2_1, L_1);
    choleskyDecomposition(A_2_2, L_2);
    choleskyDecomposition(A_2_3, L_3);
    choleskyDecomposition(A_2_4, L_4);

    std::cout << "err size 128: " << (A_2_1 - L_1 * L_1.transpose()).norm() << std::endl;
    std::cout << "err size 256: " << (A_2_2 - L_2 * L_2.transpose()).norm() << std::endl;
    std::cout << "err size 512: " << (A_2_3 - L_3 * L_3.transpose()).norm() << std::endl;
    std::cout << "err size 1024: " << (A_2_4 - L_4 * L_4.transpose()).norm() << std::endl;

    return 0;
}