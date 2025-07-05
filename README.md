# CUDA Introduction: Parallel Vector Addition

This is a simple CUDA program that demonstrates how to run computations on the GPU using **NVIDIA CUDA C++**. It performs **element-wise addition** of two large float arrays (`x` and `y`) in parallel using a custom GPU kernel.

## 🔧 What This Program Does

- Allocates two large float arrays (1 million elements each).
- Initializes them with values on the CPU.
- Launches a CUDA kernel to perform `y[i] = x[i] + y[i]` in parallel.
- Synchronizes and checks the result for accuracy.
- Prints the maximum error between computed and expected values.

---

## 🚀 CUDA Concepts Covered

### ✅ Kernel Function

```cpp
__global__ void add(int n, float* x, float* y);
```

This is the GPU function executed by many threads in parallel.

### ✅ Thread Indexing Logic

```cpp
int index = blockIdx.x * blockDim.x + threadIdx.x;
int stride = blockDim.x * gridDim.x;
for (int i = index; i < n; i += stride) {
    y[i] = x[i] + y[i];
}
```

Each thread handles a portion of the array using a **grid-stride loop** to ensure full coverage of data across blocks and threads.

### ✅ Unified Memory

```cpp
cudaMallocManaged(&x, N * sizeof(float));
```

Uses CUDA **unified memory** so both CPU and GPU can access data without manual memory copies.

### ✅ Kernel Launch

```cpp
add<<<numBlocks, blockSize>>>(N, x, y);
```

Launches the `add` kernel with calculated grid and block sizes.

### ✅ Synchronization

```cpp
cudaDeviceSynchronize();
```

Ensures the GPU work completes before accessing data on the CPU.

---

## 📦 Build Instructions

Make sure you have **NVIDIA CUDA Toolkit** installed (e.g., CUDA 12+).

Compile with `nvcc`:

```bash
nvcc -o cuda_add cuda_add.cu
```

Run:

```bash
./cuda_add
```

---

## 🧪 Example Output

```
threadIdx.x = 0, threadIdx.y = 0, threadIdx.z = 0
...
Max error: 0
```

Note: The `printf` inside the kernel prints per-thread information (mainly for educational purposes).

---

## 🧠 Key Takeaways

- CUDA enables massive parallelism by distributing work across thousands of threads.
- Grid-stride loops help efficiently scale kernels to any array size.
- Unified memory simplifies memory management for small to mid-size projects.
- Synchronization is crucial before reading GPU results on CPU.

---

## 🖥 Requirements

- CUDA-capable GPU
- CUDA Toolkit (e.g., 12.0 or later)
- C++ compiler (g++ or MSVC)

---

## 📚 References

- [CUDA C++ Programming Guide – NVIDIA](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html)
- [CUDA Toolkit Documentation](https://developer.nvidia.com/cuda-toolkit)

---

Happy parallel computing! 🚀
