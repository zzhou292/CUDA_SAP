#include <cuda_runtime.h>
#include <iostream>

// Define the structures for AABB and Sphere
struct AABB
{
    float3 min;
    float3 max;
};

struct Sphere
{
    float3 center;
    float radius;
};

// Device function to check AABB-AABB collision
__device__ bool checkAABBCollision(const AABB &a, const AABB &b)
{
    return (a.min.x <= b.max.x && a.max.x >= b.min.x) &&
           (a.min.y <= b.max.y && a.max.y >= b.min.y) &&
           (a.min.z <= b.max.z && a.max.z >= b.min.z);
}

// Device function to check Sphere-Sphere collision
__device__ bool checkSphereCollision(const Sphere &a, const Sphere &b)
{
    float3 dist = make_float3(a.center.x - b.center.x,
                              a.center.y - b.center.y,
                              a.center.z - b.center.z);
    float distSquared = dist.x * dist.x + dist.y * dist.y + dist.z * dist.z;
    float radiusSum = a.radius + b.radius;
    return distSquared <= (radiusSum * radiusSum);
}

// Device function to check AABB-Sphere collision
__device__ bool checkAABBSphereCollision(const AABB &box, const Sphere &sphere)
{
    float3 clamped;
    clamped.x = max(box.min.x, min(sphere.center.x, box.max.x));
    clamped.y = max(box.min.y, min(sphere.center.y, box.max.y));
    clamped.z = max(box.min.z, min(sphere.center.z, box.max.z));

    float3 dist = make_float3(clamped.x - sphere.center.x,
                              clamped.y - sphere.center.y,
                              clamped.z - sphere.center.z);

    float distSquared = dist.x * dist.x + dist.y * dist.y + dist.z * dist.z;
    return distSquared <= (sphere.radius * sphere.radius);
}

// Kernel to detect collisions between AABBs
__global__ void detectAABBCollisions(const AABB *aabbs, int numAABBs, bool *collisionMatrix)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numAABBs)
    {
        for (int j = 0; j < numAABBs; j++)
        {
            if (idx != j)
            {
                collisionMatrix[idx * numAABBs + j] = checkAABBCollision(aabbs[idx], aabbs[j]);
            }
        }
    }
}

// Kernel to detect collisions between Spheres
__global__ void detectSphereCollisions(const Sphere *spheres, int numSpheres, bool *collisionMatrix)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numSpheres)
    {
        for (int j = 0; j < numSpheres; j++)
        {
            if (idx != j)
            {
                collisionMatrix[idx * numSpheres + j] = checkSphereCollision(spheres[idx], spheres[j]);
            }
        }
    }
}

// Kernel to detect collisions between AABBs and Spheres
__global__ void detectAABBSphereCollisions(const AABB *aabbs, int numAABBs, const Sphere *spheres, int numSpheres, bool *collisionMatrix)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numAABBs)
    {
        for (int j = 0; j < numSpheres; j++)
        {
            collisionMatrix[idx * numSpheres + j] = checkAABBSphereCollision(aabbs[idx], spheres[j]);
        }
    }
}

int main()
{
    const int numAABBs = 2;
    const int numSpheres = 2;

    // Define some AABBs and Spheres
    AABB h_aabbs[numAABBs] = {{{0, 0, 0}, {1, 1, 1}}, {{1.5, 1.5, 1.5}, {2.5, 2.5, 2.5}}};
    Sphere h_spheres[numSpheres] = {{{0.5, 0.5, 0.5}, 0.5}, {{2, 2, 2}, 0.5}};

    // Device memory allocations
    AABB *d_aabbs;
    Sphere *d_spheres;
    bool *d_collisionMatrixAABB;
    bool *d_collisionMatrixSpheres;
    bool *d_collisionMatrixAABBSpheres;

    cudaMalloc((void **)&d_aabbs, numAABBs * sizeof(AABB));
    cudaMalloc((void **)&d_spheres, numSpheres * sizeof(Sphere));
    cudaMalloc((void **)&d_collisionMatrixAABB, numAABBs * numAABBs * sizeof(bool));
    cudaMalloc((void **)&d_collisionMatrixSpheres, numSpheres * numSpheres * sizeof(bool));
    cudaMalloc((void **)&d_collisionMatrixAABBSpheres, numAABBs * numSpheres * sizeof(bool));

    // Copy data to device
    cudaMemcpy(d_aabbs, h_aabbs, numAABBs * sizeof(AABB), cudaMemcpyHostToDevice);
    cudaMemcpy(d_spheres, h_spheres, numSpheres * sizeof(Sphere), cudaMemcpyHostToDevice);

    // Kernel launches
    int threadsPerBlock = 256;
    int blocksPerGridAABB = (numAABBs + threadsPerBlock - 1) / threadsPerBlock;
    int blocksPerGridSpheres = (numSpheres + threadsPerBlock - 1) / threadsPerBlock;

    detectAABBCollisions<<<blocksPerGridAABB, threadsPerBlock>>>(d_aabbs, numAABBs, d_collisionMatrixAABB);
    detectSphereCollisions<<<blocksPerGridSpheres, threadsPerBlock>>>(d_spheres, numSpheres, d_collisionMatrixSpheres);
    detectAABBSphereCollisions<<<blocksPerGridAABB, threadsPerBlock>>>(d_aabbs, numAABBs, d_spheres, numSpheres, d_collisionMatrixAABBSpheres);

    cudaDeviceSynchronize();

    // Allocate memory for results on host
    bool h_collisionMatrixAABB[numAABBs * numAABBs];
    bool h_collisionMatrixSpheres[numSpheres * numSpheres];
    bool h_collisionMatrixAABBSpheres[numAABBs * numSpheres];

    // Copy results back to host
    cudaMemcpy(h_collisionMatrixAABB, d_collisionMatrixAABB, numAABBs * numAABBs * sizeof(bool), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_collisionMatrixSpheres, d_collisionMatrixSpheres, numSpheres * numSpheres * sizeof(bool), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_collisionMatrixAABBSpheres, d_collisionMatrixAABBSpheres, numAABBs * numSpheres * sizeof(bool), cudaMemcpyDeviceToHost);

    // Print results
    std::cout << "AABB-AABB Collisions:" << std::endl;
    for (int i = 0; i < numAABBs; ++i)
    {
        for (int j = 0; j < numAABBs; ++j)
        {
            std::cout << h_collisionMatrixAABB[i * numAABBs + j] << " ";
        }
        std::cout << std::endl;
    }

    std::cout << "Sphere-Sphere Collisions:" << std::endl;
    for (int i = 0; i < numSpheres; ++i)
    {
        for (int j = 0; j < numSpheres; ++j)
        {
            std::cout << h_collisionMatrixSpheres[i * numSpheres + j] << " ";
        }
        std::cout << std::endl;
    }

    std::cout << "AABB-Sphere Collisions:" << std::endl;
    for (int i = 0; i < numAABBs; ++i)
    {
        for (int j = 0; j < numSpheres; ++j)
        {
            std::cout << h_collisionMatrixAABBSpheres[i * numSpheres + j] << " ";
        }
        std::cout << std::endl;
    }

    // Free device memory
    cudaFree(d_aabbs);
    cudaFree(d_spheres);
    cudaFree(d_collisionMatrixAABB);
    cudaFree(d_collisionMatrixSpheres);
    cudaFree(d_collisionMatrixAABBSpheres);

    return 0;
}
