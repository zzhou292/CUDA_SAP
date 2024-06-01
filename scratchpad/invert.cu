#include <Eigen/Dense>
#include <iostream>

__global__ void gaussJordanKernel(double *d_mat, double *d_inv, int n)
{
    int col = threadIdx.x + blockDim.x * blockIdx.x;
    if (col >= n)
        return;

    for (int pivot = 0; pivot < n; ++pivot)
    {
        if (col == pivot)
        {
            double pivot_val = d_mat[pivot + pivot * n];
            for (int j = 0; j < n; ++j)
            {
                d_mat[pivot + j * n] /= pivot_val;
                d_inv[pivot + j * n] /= pivot_val;
            }
        }
        __syncthreads();

        if (col != pivot)
        {
            double factor = d_mat[col + pivot * n];
            for (int j = 0; j < n; ++j)
            {
                d_mat[col + j * n] -= factor * d_mat[pivot + j * n];
                d_inv[col + j * n] -= factor * d_inv[pivot + j * n];
            }
        }
        __syncthreads();
    }
}

void invertMatrix(double *h_mat, double *h_inv, int n)
{
    double *d_mat;
    double *d_inv;
    size_t size = n * n * sizeof(double);
    cudaMalloc((void **)&d_mat, size);
    cudaMalloc((void **)&d_inv, size);

    cudaMemcpy(d_mat, h_mat, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_inv, h_inv, size, cudaMemcpyHostToDevice);

    // cudaEvent_t start, stop;
    // cudaEventCreate(&start);
    // cudaEventCreate(&stop);

    // cudaEventRecord(start);

    int threadsPerBlock = 128;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    gaussJordanKernel<<<blocksPerGrid, threadsPerBlock>>>(d_mat, d_inv, n);

    cudaDeviceSynchronize();

    // cudaEventRecord(stop);

    // cudaEventSynchronize(stop);
    // float milliseconds = 0.0;
    // cudaEventElapsedTime(&milliseconds, start, stop);

    // std::cout << "Elapsed time for Gausse Jordan invert : " << milliseconds << " ms\n";

    cudaMemcpy(h_inv, d_inv, size, cudaMemcpyDeviceToHost);

    cudaFree(d_mat);
    cudaFree(d_inv);
}

int main()
{

    const int n = 256;
    Eigen::MatrixXd A = Eigen::MatrixXd::Random(n, n);
    A = A.transpose() * A; // Create a symmetric positive-definite matrix

    // Convert Eigen matrix to column-major double array
    double *h_mat = A.data();
    Eigen::MatrixXd I = Eigen::MatrixXd::Identity(n, n);
    double *h_inv = I.data();

    // Perform matrix inversion using CUDA
    invertMatrix(h_mat, h_inv, n);

    // Print the inverted matrix
    Eigen::Map<Eigen::MatrixXd> inv(h_inv, n, n);
    std::cout << A.inverse() << std::endl;
    std::cout << "=========================" << std::endl;
    std::cout << inv << std::endl;
    std::cout << "inverse err: " << (A.inverse() - inv).norm() << std::endl;

    return 0;
}
