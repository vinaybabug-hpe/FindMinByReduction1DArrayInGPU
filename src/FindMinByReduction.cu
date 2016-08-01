/*
 ============================================================================
 Name        : FindMinByReduction.cu
 Author      : Vinay B Gavirangaswamy
 Version     :
 Copyright   : Put copyright notice
 Description : CUDA compute reciprocals
 ============================================================================
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <math.h>
#include <unistd.h>
#include <sys/time.h>

#if __DEVICE_EMULATION__
#define DEBUG_SYNC __syncthreads();
#else
#define DEBUG_SYNC
#endif

#ifndef MIN
#define MIN(x,y) ((x < y) ? x : y)
#endif

#ifndef MIN_IDX
#define MIN_IDX(x,y, idx_x, idx_y) ((x < y) ? idx_x : idx_y)
#endif

#if (__CUDA_ARCH__ < 200)
#define int_mult(x,y)	__mul24(x,y)
#else
#define int_mult(x,y)	x*y
#endif

#define inf 0x7f800000

bool isPow2(unsigned int x)
{
    return ((x&(x-1))==0);
}

unsigned int nextPow2(unsigned int x)
{
    --x;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    return ++x;
}

// Utility class used to avoid linker errors with extern
// unsized shared memory arrays with templated type
template<class T>
struct SharedMemory {
	__device__ inline operator T *() {
		extern __shared__ int __smem[];
		return (T *) __smem;
	}

	__device__ inline operator const T *() const {
		extern __shared__ int __smem[];
		return (T *) __smem;
	}
};

// specialize for double to avoid unaligned memory
// access compile errors
template<>
struct SharedMemory<double> {
	__device__ inline operator double *() {
		extern __shared__ double __smem_d[];
		return (double *) __smem_d;
	}

	__device__ inline operator const double *() const {
		extern __shared__ double __smem_d[];
		return (double *) __smem_d;
	}
};



/*
 This version finds minimum and index at which it was found in multiple elements per thread sequentially.  This reduces the overall
 cost of the algorithm while keeping the work complexity O(n) and the step complexity O(log n).
 (Brent's Theorem optimization)

 Note, this kernel needs a minimum of 64*sizeof(T) bytes of shared memory.
 In other words if blockSize <= 32, allocate 64*sizeof(T) bytes.
 If blockSize > 32, allocate blockSize*sizeof(T) bytes.
 */
template<class T, unsigned int blockSize, bool nIsPow2>
__global__ void reduceMin6(T *g_idata, int *g_idxs, T *g_odata, int *g_oIdxs, unsigned int n) {

	T *sdata = SharedMemory<T>();
	int *sdataIdx = ((int *)sdata) + blockSize;



	// perform first level of reduction,
	// reading from global memory, writing to shared memory
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x * blockSize * 2 + threadIdx.x;
	unsigned int gridSize = blockSize * 2 * gridDim.x;


	T myMin = 99999;
	int myMinIdx = -1;
	// we reduce multiple elements per thread.  The number is determined by the
	// number of active thread blocks (via gridDim).  More blocks will result
	// in a larger gridSize and therefore fewer elements per thread
	while (i < n) {
		myMinIdx  = MIN_IDX(g_idata[i], myMin, g_idxs[i], myMinIdx);
		myMin = MIN(g_idata[i], myMin);



		// ensure we don't read out of bounds -- this is optimized away for powerOf2 sized arrays
		if (nIsPow2 || i + blockSize < n){
			//myMin += g_idata[i + blockSize];
			myMinIdx  = MIN_IDX(g_idata[i + blockSize], myMin, g_idxs[i + blockSize], myMinIdx);
			myMin = MIN(g_idata[i + blockSize], myMin);
		}

		i += gridSize;
	}


	// each thread puts its local sum into shared memory
	sdata[tid] = myMin;
	sdataIdx[tid] = myMinIdx;

	__syncthreads();

	// do reduction in shared mem
	if ((blockSize >= 512) && (tid < 256)) {

		sdataIdx[tid] = myMinIdx = MIN_IDX(sdata[tid + 256], myMin, sdataIdx[tid + 256], myMinIdx);
		sdata[tid] = myMin = MIN(sdata[tid + 256], myMin);

	}

	__syncthreads();

	if ((blockSize >= 256) && (tid < 128)) {

		sdataIdx[tid] = myMinIdx = MIN_IDX(sdata[tid + 128], myMin, sdataIdx[tid + 128], myMinIdx);
		sdata[tid] = myMin = MIN(sdata[tid + 128], myMin);


	}

	__syncthreads();

	if ((blockSize >= 128) && (tid < 64)) {

		sdataIdx[tid] = myMinIdx = MIN_IDX(sdata[tid + 64], myMin, sdataIdx[tid + 64], myMinIdx);
		sdata[tid] = myMin = MIN(sdata[tid + 64], myMin);
	}

	__syncthreads();

#if (__CUDA_ARCH__ >= 300 )
	if (tid < 32) {
		// Fetch final intermediate sum from 2nd warp
		if (blockSize >= 64){

			myMinIdx = MIN_IDX(sdata[tid + 32], myMin, sdataIdx[tid + 32], myMinIdx);
			myMin = MIN(sdata[tid + 32], myMin);
		}
		// Reduce final warp using shuffle
		for (int offset = warpSize / 2; offset > 0; offset /= 2) {

			int tempMyMinIdx = __shfl_down(myMinIdx, offset);
			float tempMyMin = __shfl_down(myMin, offset);

			myMinIdx = MIN_IDX(tempMyMin, myMin, tempMyMinIdx , myMinIdx);
			myMin = MIN(tempMyMin, myMin);

		}

	}
#else
	// fully unroll reduction within a single warp
	if ((blockSize >= 64) && (tid < 32))
	{
		//sdata[tid] = myMin = myMin + sdata[tid + 32];
		sdataIdx[tid] = myMinIdx = MIN_IDX(sdata[tid + 32], myMin, sdataIdx[tid + 32], myMinIdx);
		sdata[tid] = myMin = MIN(sdata[tid + 32], myMin);
	}

	__syncthreads();

	if ((blockSize >= 32) && (tid < 16))
	{


		sdataIdx[tid] = myMinIdx = MIN_IDX(sdata[tid + 16], myMin, sdataIdx[tid + 16], myMinIdx);
		sdata[tid] = myMin = MIN(sdata[tid + 16], myMin);
	}

	__syncthreads();

	if ((blockSize >= 16) && (tid < 8))
	{

		sdataIdx[tid] = myMinIdx = MIN_IDX(sdata[tid + 8], myMin, sdataIdx[tid + 8], myMinIdx);
		sdata[tid] = myMin = MIN(sdata[tid + 8], myMin);
	}

	__syncthreads();

	if ((blockSize >= 8) && (tid < 4))
	{

		sdataIdx[tid] = myMinIdx = MIN_IDX(sdata[tid + 4], myMin, sdataIdx[tid + 4], myMinIdx);
		sdata[tid] = myMin = MIN(sdata[tid + 4], myMin);
	}

	__syncthreads();

	if ((blockSize >= 4) && (tid < 2))
	{

		sdataIdx[tid] = myMinIdx = MIN_IDX(sdata[tid + 2], myMin, sdataIdx[tid + 2], myMinIdx);
		sdata[tid] = myMin = MIN(sdata[tid + 2], myMin);
	}

	__syncthreads();

	if ((blockSize >= 2) && ( tid < 1))
	{

		sdataIdx[tid] = myMinIdx = MIN_IDX(sdata[tid + 1], myMin, sdataIdx[tid + 1], myMinIdx);
		sdata[tid] = myMin = MIN(sdata[tid + 1], myMin);
	}

	__syncthreads();
#endif

	__syncthreads();
	// write result for this block to global mem
	if (tid == 0){
		g_odata[blockIdx.x] = myMin;
		g_oIdxs[blockIdx.x] = myMinIdx;
	}
}

////////////////////////////////////////////////////////////////////////////////
// Compute the number of threads and blocks to use for the given reduction kernel
// For the kernels >= 3, we set threads / block to the minimum of maxThreads and
// n/2. For kernels < 3, we set to the minimum of maxThreads and n.  For kernel
// 6, we observe the maximum specified number of blocks, because each thread in
// that kernel can process a variable number of elements.
////////////////////////////////////////////////////////////////////////////////
void getNumBlocksAndThreads(int whichKernel, int n, int maxBlocks,
		int maxThreads, int &blocks, int &threads) {

	//get device capability, to avoid block/grid size exceed the upper bound
	cudaDeviceProp prop;
	int device;
	cudaGetDevice(&device);
	cudaGetDeviceProperties(&prop, device);

	if (whichKernel < 3) {
		threads = (n < maxThreads) ? nextPow2(n) : maxThreads;
		blocks = (n + threads - 1) / threads;
	} else {
		threads = (n < maxThreads * 2) ? nextPow2((n + 1) / 2) : maxThreads;
		blocks = (n + (threads * 2 - 1)) / (threads * 2);
	}

	if ((float) threads * blocks
			> (float) prop.maxGridSize[0] * prop.maxThreadsPerBlock) {
		printf("n is too large, please choose a smaller number!\n");
	}

	if (blocks > prop.maxGridSize[0]) {
		printf(
				"Grid size <%d> exceeds the device capability <%d>, set block size as %d (original %d)\n",
				blocks, prop.maxGridSize[0], threads * 2, threads);

		blocks /= 2;
		threads *= 2;
	}

	if (whichKernel == 6) {
		blocks = MIN(maxBlocks, blocks);
	}
}

////////////////////////////////////////////////////////////////////////////////
// Wrapper function for kernel launch
////////////////////////////////////////////////////////////////////////////////
template<class T>
void reduceMin(int size, int threads, int blocks, int whichKernel, T *d_idata,
		T *d_odata, int *idxs, int *oIdxs) {
	dim3 dimBlock(threads, 1, 1);
	dim3 dimGrid(blocks, 1, 1);

	// when there is only one warp per block, we need to allocate two warps
	// worth of shared memory so that we don't index shared memory out of bounds
	int smemSize =
	        (threads <= 32) ? 2 * threads * sizeof(T) : threads * sizeof(T);
	smemSize += threads*sizeof(int);

	if (isPow2(size)) {
		switch (threads) {
		case 512:
			reduceMin6<T, 512, true> <<<dimGrid, dimBlock, smemSize>>>(d_idata, idxs,
					d_odata, oIdxs, size);
			break;

		case 256:
			reduceMin6<T, 256, true> <<<dimGrid, dimBlock, smemSize>>>(d_idata, idxs,
					d_odata, oIdxs, size);
			break;

		case 128:
			reduceMin6<T, 128, true> <<<dimGrid, dimBlock, smemSize>>>(d_idata, idxs,
					d_odata, oIdxs, size);
			break;

		case 64:
			reduceMin6<T, 64, true> <<<dimGrid, dimBlock, smemSize>>>(d_idata, idxs,
					d_odata, oIdxs, size);
			break;

		case 32:
			reduceMin6<T, 32, true> <<<dimGrid, dimBlock, smemSize>>>(d_idata, idxs,
					d_odata, oIdxs, size);
			break;

		case 16:
			reduceMin6<T, 16, true> <<<dimGrid, dimBlock, smemSize>>>(d_idata, idxs,
					d_odata, oIdxs, size);
			break;

		case 8:
			reduceMin6<T, 8, true> <<<dimGrid, dimBlock, smemSize>>>(d_idata, idxs,
					d_odata, oIdxs, size);
			break;

		case 4:
			reduceMin6<T, 4, true> <<<dimGrid, dimBlock, smemSize>>>(d_idata, idxs,
					d_odata, oIdxs, size);
			break;

		case 2:
			reduceMin6<T, 2, true> <<<dimGrid, dimBlock, smemSize>>>(d_idata, idxs,
					d_odata, oIdxs, size);
			break;

		case 1:
			reduceMin6<T, 1, true> <<<dimGrid, dimBlock, smemSize>>>(d_idata, idxs,
					d_odata, oIdxs, size);
			break;
		}
	} else {
		switch (threads) {
		case 512:
			reduceMin6<T, 512, false> <<<dimGrid, dimBlock, smemSize>>>(d_idata, idxs,
					d_odata, oIdxs, size);
			break;

		case 256:
			reduceMin6<T, 256, false> <<<dimGrid, dimBlock, smemSize>>>(d_idata, idxs,
					d_odata, oIdxs, size);
			break;

		case 128:
			reduceMin6<T, 128, false> <<<dimGrid, dimBlock, smemSize>>>(d_idata, idxs,
					d_odata, oIdxs, size);
			break;

		case 64:
			reduceMin6<T, 64, false> <<<dimGrid, dimBlock, smemSize>>>(d_idata, idxs,
					d_odata, oIdxs, size);
			break;

		case 32:
			reduceMin6<T, 32, false> <<<dimGrid, dimBlock, smemSize>>>(d_idata, idxs,
					d_odata, oIdxs, size);
			break;

		case 16:
			reduceMin6<T, 16, false> <<<dimGrid, dimBlock, smemSize>>>(d_idata, idxs,
					d_odata, oIdxs, size);
			break;

		case 8:
			reduceMin6<T, 8, false> <<<dimGrid, dimBlock, smemSize>>>(d_idata, idxs,
					d_odata, oIdxs, size);
			break;

		case 4:
			reduceMin6<T, 4, false> <<<dimGrid, dimBlock, smemSize>>>(d_idata, idxs,
					d_odata, oIdxs, size);
			break;

		case 2:
			reduceMin6<T, 2, false> <<<dimGrid, dimBlock, smemSize>>>(d_idata, idxs,
					d_odata, oIdxs, size);
			break;

		case 1:
			reduceMin6<T, 1, false> <<<dimGrid, dimBlock, smemSize>>>(d_idata, idxs,
					d_odata, oIdxs, size);
			break;
		}
	}

}

////////////////////////////////////////////////////////////////////////////////
//! Compute minimum reduction on CPU
//! We use Kahan summation for an accurate sum of large arrays.
//! http://en.wikipedia.org/wiki/Kahan_summation_algorithm
//!
//! @param data       pointer to input data
//! @param size       number of input data elements
//! @param min        minimum value (out)
//! @param min        minimum value index (out)
////////////////////////////////////////////////////////////////////////////////
template<class T>
void reduceMINCPU(T *data, int size, T *min, int *idx)
{
    *min = data[0];
    int min_idx = 0;
    T c = (T)0.0;

    for (int i = 1; i < size; i++)
    {
        T y = data[i];
        T t = MIN(*min, y);
        min_idx = MIN_IDX(*min, y, min_idx, i);
        (*min) = t;
    }

    *idx = min_idx;

    return;
}


// Instantiate the reduction function for 3 types
template void
reduceMin<int>(int size, int threads, int blocks, int whichKernel, int *d_idata,
		int *d_odata, int *idxs, int *oIdxs);

template void
reduceMin<float>(int size, int threads, int blocks, int whichKernel, float *d_idata,
		float *d_odata, int *idxs, int *oIdxs);

template void
reduceMin<double>(int size, int threads, int blocks, int whichKernel, double *d_idata,
		double *d_odata, int *idxs, int *oIdxs);

unsigned long long int minimizationViaReduction(int num_els) {


	unsigned long long int delta;

	int maxThreads = 256;  // number of threads per block
	int whichKernel = 6;
	int maxBlocks = 64;

	float* d_in = NULL;
	float* d_out = NULL;
	int *d_idxs = NULL;
	int *d_oIdxs = NULL;

	printf("%d elements\n", num_els);
	printf("%d threads (max)\n", maxThreads);

	int numBlocks = 0;
	int numThreads = 0;
	getNumBlocksAndThreads(whichKernel, num_els, maxBlocks, maxThreads, numBlocks,
			numThreads);


	cudaMalloc((void **) &d_in, num_els * sizeof(float));
	cudaMalloc((void **) &d_idxs, num_els * sizeof(int));
	cudaMalloc((void **) &d_out, numBlocks * sizeof(float));
	cudaMalloc((void **) &d_oIdxs, numBlocks * sizeof(int));

	float* in = (float*) malloc(num_els * sizeof(float));
	int *idxs = (int*) malloc(num_els * sizeof(int));
	float* out = (float*) malloc(numBlocks * sizeof(float));
	int* oIdxs = (int*) malloc(numBlocks * sizeof(int));

	for (int i = 0; i < num_els; i++) {
		in[i] = (double) rand() / (double) RAND_MAX;
		idxs[i] = i;
	}


	// copy data directly to device memory
	cudaMemcpy(d_in, in, num_els * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_idxs, idxs, num_els * sizeof(int),cudaMemcpyHostToDevice);
	cudaMemcpy(d_out, out, numBlocks * sizeof(float),cudaMemcpyHostToDevice);
	cudaMemcpy(d_oIdxs, oIdxs, numBlocks * sizeof(int),cudaMemcpyHostToDevice);

	reduceMin<float>(num_els, numThreads, numBlocks, whichKernel, d_in, d_out, d_idxs, d_oIdxs);

	cudaMemcpy(out, d_out, numBlocks * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(oIdxs, d_oIdxs, numBlocks * sizeof(int), cudaMemcpyDeviceToHost);

	int min_idx = -1;
	float min = 999999;

	for(int i=0; i< numBlocks; i++){

		printf("\n Reduce MIN \ BLOCK GPU idx: %d  value: %f", oIdxs[i], out[i]);
		min_idx = MIN_IDX(out[i], min, oIdxs[i], min_idx);
		min = MIN(out[i], min);

	}


	printf("\n\n Reduce MIN GPU idx: %d  value: %f\n", min_idx, min);


	reduceMINCPU<float>(in, num_els, &min, &min_idx);


	printf("\n\n Reduce MIN CPU idx: %d  value: %f", min_idx, min);

	cudaFree(d_in);
	cudaFree(d_out);
	cudaFree(d_idxs);

	free(in);
	free(out);

	//system("pause");

	return delta;

}

int main(int argc, char* argv[]) {

	minimizationViaReduction(1024);

	return 0;
}
