WITH cuda_entry AS (
  SELECT $body$
CUDA is NVIDIA's parallel computing platform and programming model for running general-purpose work on NVIDIA GPUs. It lets software move suitable parts of a computation from the CPU to thousands of GPU threads, making operations such as matrix multiplication, simulation, rendering, video processing, and AI training or inference much faster when the workload has enough parallel work.

CUDA was introduced publicly in 2007. The name is often expanded as **Compute Unified Device Architecture**, although current NVIDIA documentation generally treats CUDA as the platform's name. CUDA is not one programming language, one library, or the physical “CUDA cores” on a GPU. It is an ecosystem that includes programming APIs, compilers, runtimes, optimized libraries, debugging and profiling tools, and driver interfaces.

## How CUDA works

A CUDA application is heterogeneous. **Host code** runs on the CPU and uses CUDA APIs to allocate memory, move data, launch GPU work, and synchronize results. **Device code** runs on the GPU. A function launched on the GPU is called a **kernel**.

One kernel launch creates many lightweight threads. CUDA organizes them into a hierarchy:

- A **thread** handles one logical piece of work, such as one array element.
- A **warp** is a hardware execution group of 32 threads using NVIDIA's SIMT model.
- A **thread block** groups threads that can synchronize and share fast on-chip shared memory.
- A **grid** contains all blocks launched for one kernel.
- A **stream** is an ordered queue of kernels and memory operations. Multiple streams can expose overlap and concurrency when the hardware and dependencies allow it.

Blocks are scheduled onto streaming multiprocessors, or SMs. The programmer describes the grid and block dimensions, while the GPU schedules blocks according to available resources. Blocks generally cannot assume a particular execution order, which is what allows the same kernel grid to scale across small and large GPUs.

## CUDA platform vs CUDA Toolkit vs driver

| Term | What it is | Why it matters |
| --- | --- | --- |
| CUDA platform | The overall NVIDIA GPU-computing architecture, APIs, software stack, and programming model | The umbrella term people usually mean by “CUDA” |
| CUDA Toolkit | Developer package containing the compiler toolchain, runtime, headers, libraries, and tools | Needed to compile CUDA applications or custom extensions locally |
| NVIDIA driver | System software that controls the GPU and exposes the CUDA driver API | Required to run CUDA workloads |
| CUDA runtime | Higher-level API, commonly called cudart, used by applications to launch kernels and manage devices | Often bundled with an application or framework build |
| CUDA driver API | Lower-level interface exposed by the installed NVIDIA driver | Used directly by runtimes, JIT systems, and advanced applications |
| CUDA-enabled GPU | NVIDIA hardware with a supported compute capability | The device that executes the GPU code |

This distinction explains a common surprise: a machine can run a CUDA-enabled PyTorch wheel without having the full CUDA Toolkit installed. The wheel may include the CUDA runtime and required libraries, while the machine supplies a compatible NVIDIA driver. The Toolkit is still needed when compiling CUDA C++ or building a custom CUDA extension from source.

## What is included in the CUDA ecosystem?

| Component | Role |
| --- | --- |
| `nvcc` | Compiler driver for CUDA C++ and PTX that coordinates host and device compilation |
| CUDA Runtime and Driver APIs | Device management, memory, kernel launches, streams, events, and synchronization |
| cuBLAS and cuBLASLt | Dense linear algebra and matrix multiplication |
| cuDNN | Optimized deep-learning primitives including attention, convolution, matmul, normalization, softmax, and pooling |
| NCCL | Collective communication for multi-GPU and multi-node workloads |
| cuFFT, cuSPARSE, and cuSOLVER | Fourier transforms, sparse operations, and numerical solvers |
| Thrust, CUB, and libcu++ | Parallel C++ algorithms, primitives, and standard-library support |
| Nsight Systems and Nsight Compute | System-level tracing and detailed GPU-kernel profiling |
| PTX | NVIDIA's virtual GPU instruction-set representation used between source code and hardware-specific machine code |

Frameworks such as PyTorch, TensorFlow, and JAX call these lower-level components rather than implementing every GPU kernel themselves. Higher-level compilers may also generate CUDA code, PTX, or library calls. A user can benefit from CUDA without writing a kernel directly.

## Kernels, warps, and SIMT

CUDA uses **Single Instruction, Multiple Threads** execution. Threads in a warp execute the same kernel but can operate on different data. If threads in one warp take different branches, the GPU may need to execute the paths separately. This **warp divergence** can reduce efficiency, although correctness is preserved.

Performance also depends on how warps access memory. Adjacent threads that access adjacent global-memory locations can produce coalesced transfers. Threads in a block can reuse data through shared memory, but shared-memory bank conflicts and unnecessary synchronization can remove the benefit. More threads are not automatically faster if registers, shared memory, memory bandwidth, or dependencies limit occupancy and throughput.

## CUDA memory hierarchy

| Memory area | Scope and typical use | Main tradeoff |
| --- | --- | --- |
| Registers | Private values for one thread | Fastest, but limited and excessive use can reduce active warps |
| Shared memory | On-chip storage shared by a thread block | Fast reuse and cooperation, but manually managed and capacity-limited |
| Global memory | Large device memory visible to all GPU threads | High bandwidth but much higher latency than on-chip memory |
| Constant memory | Read-only values shared across a launch | Efficient when threads read the same locations |
| Local memory | Per-thread address space that can spill into device memory | Convenient abstraction, but spills are much slower than registers |
| Unified or managed memory | One managed address space accessible by CPUs and GPUs | Easier programming, but page movement and placement still affect performance |

CUDA streams and asynchronous copies can overlap CPU work, GPU computation, and data transfers. Actual overlap depends on the device, memory type, operation dependencies, and available engines. A kernel launch being asynchronous to the CPU does not guarantee that two GPU operations execute simultaneously.

## Compute capability, PTX, and GPU compatibility

Every CUDA-capable NVIDIA GPU has a **compute capability** such as 8.6, 9.0, or 10.0. It describes supported GPU features and hardware limits. Compiler target names commonly use `compute_XY` for a virtual PTX target and `sm_XY` for hardware-specific code.

`nvcc` can place several forms of device code into a **fat binary**:

- **Cubin or SASS code** is compiled for a particular SM target and can start without JIT compilation on compatible hardware.
- **PTX** is a virtual instruction representation that a sufficiently new driver can JIT-compile for compatible current or future GPUs.

Shipping only one hardware target can make a binary fail on another GPU generation. Shipping PTX improves forward reach but depends on the installed driver understanding that PTX version and may add first-run compilation time. Production packages often include several cubins plus PTX as a fallback.

Compute capability is a GPU property. CUDA Toolkit version is a software release. They are related through compiler support, but **CUDA 13.x does not mean compute capability 13.x**.

## CUDA version and driver compatibility

CUDA compatibility involves at least four versions: the GPU's compute capability, the installed NVIDIA driver, the Toolkit used to build an application, and the runtime or libraries shipped with that application.

| Compatibility case | General rule |
| --- | --- |
| Newer driver running an application built with an older Toolkit | Normally supported through backward compatibility |
| Older driver with a newer Toolkit runtime in the same CUDA major family | May work through minor-version compatibility, subject to minimum-driver and feature restrictions |
| Older driver across CUDA major families | Requires a supported forward-compatibility package on eligible systems |
| Application contains PTX generated by a newer Toolkit | The driver must be new enough to JIT-compile that PTX version |
| Library or framework build targets an unsupported compute capability | The package may need another build or the GPU may be too old for that release |

The “CUDA Version” shown by `nvidia-smi` comes from driver capability. It should not be treated as proof that the matching CUDA Toolkit is installed. Check `nvcc --version` for a locally installed compiler toolkit, inspect the framework's build version for bundled runtime libraries, and test device access inside the actual environment.

NVIDIA's archive listed CUDA Toolkit **13.3.1** as the latest production release on August 31, 2026. CUDA 13.4.0 was listed separately as a developer preview. Version-specific compatibility and known issues should always be checked in the release notes because the current version changes faster than the CUDA programming model.

## Why CUDA matters for AI

Modern AI workloads are dominated by tensor operations that expose large amounts of parallelism. CUDA gives frameworks access to NVIDIA GPUs and to optimized libraries for matrix multiplication, attention, convolution, collectives, and memory movement. Tensor Cores accelerate selected low-precision and mixed-precision operations, while CUDA provides the software path used to schedule and feed them.

CUDA's advantage is the full stack: compiler support, tuned libraries, framework integration, profilers, deployment images, documentation, and years of optimized kernels. This ecosystem creates real switching costs. It does not mean every model operation is automatically fast, every NVIDIA GPU supports every datatype, or CUDA code is portable to non-NVIDIA hardware unchanged.

## CUDA vs ROCm vs oneAPI and SYCL

| Platform | Primary hardware ecosystem | Programming approach | Practical distinction |
| --- | --- | --- | --- |
| CUDA | NVIDIA GPUs | CUDA C++, CUDA Python, libraries, generated kernels, and framework bindings | Deepest first-party NVIDIA integration and widest assumption in AI software |
| ROCm and HIP | Primarily AMD GPUs | HIP and ROCm libraries, with tools for porting some CUDA-style code | Main open AMD GPU-compute stack, but hardware and library support must be checked |
| oneAPI and SYCL | Intel-focused with a cross-vendor programming model | Standard C++-based SYCL plus oneAPI libraries and compilers | Designed for heterogeneous portability, with backend support varying by implementation |

These ecosystems can solve similar problems, but matching API names or source syntax does not guarantee identical kernels, numerical behavior, library coverage, performance, or operational support.

## Common CUDA errors

- **“CUDA unavailable”** can mean no NVIDIA GPU is visible, the driver is missing, the process lacks device access, or the installed framework is a CPU-only build.
- **Driver too old** means the runtime, library, or PTX code needs capabilities the installed driver does not provide.
- **No kernel image is available** often means the binary was not compiled for the GPU's compute capability and contains no usable PTX fallback.
- **Out of memory** concerns available GPU memory and allocations, not whether the GPU supports CUDA.
- **Illegal memory access** usually indicates a kernel indexing, lifetime, synchronization, or pointer error. The failure may surface at a later synchronization call because launches are asynchronous.
- **Slow GPU utilization** can come from small workloads, CPU or input bottlenecks, excessive transfers, synchronization, poor memory access, divergence, or unsupported optimized kernels.

The reliable debugging sequence is to identify the GPU and compute capability, check the NVIDIA driver, inspect the application or framework's CUDA build, verify device visibility inside its container or environment, reproduce with a minimal device query, and then profile rather than guessing from utilization alone.

## Bottom line

CUDA is the software platform that turns an NVIDIA GPU into a programmable parallel computer. The GPU supplies the hardware, the driver exposes it, the Toolkit builds applications, the runtime launches work, and optimized libraries make common workloads practical. Understanding those layers is the key to both CUDA performance and CUDA compatibility.
$body$::text AS body
)
UPDATE content_items AS item
SET
  title = 'CUDA',
  summary = 'NVIDIA''s parallel computing platform and programming model for accelerating general-purpose, scientific, graphics, and AI workloads on NVIDIA GPUs.',
  body = cuda_entry.body,
  blocks = jsonb_build_array(
    jsonb_build_object('id', 'markdown-1', 'type', 'markdown', 'content', cuda_entry.body)
  ),
  sources = jsonb_build_array(
    jsonb_build_object('title', 'NVIDIA CUDA Programming Guide', 'url', 'https://docs.nvidia.com/cuda/cuda-programming-guide/index.html'),
    jsonb_build_object('title', 'NVIDIA CUDA Platform, Compute Capability, PTX, and Fat Binaries', 'url', 'https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/cuda-platform.html'),
    jsonb_build_object('title', 'NVIDIA CUDA Compatibility Guide', 'url', 'https://docs.nvidia.com/deploy/cuda-compatibility/latest/'),
    jsonb_build_object('title', 'NVIDIA CUDA Toolkit Archive', 'url', 'https://developer.nvidia.com/cuda-toolkit-archive'),
    jsonb_build_object('title', 'NVIDIA cuDNN Documentation', 'url', 'https://docs.nvidia.com/deeplearning/cudnn/latest/')
  ),
  metadata = COALESCE(item.metadata, '{}'::jsonb) || jsonb_build_object(
    'category', 'Systems, Tools & Safety',
    'relatedTerms', jsonb_build_array('gpu', 'deep-learning', 'inference', 'large-language-model', 'pytorch', 'quantization'),
    'analogy', 'CUDA is like a workshop where the CPU is the foreman and the GPU supplies thousands of specialized workers, plus the tools and rules needed to coordinate them.',
    'seoDescription', 'CUDA is NVIDIA''s GPU computing platform. Learn how kernels, memory, toolkits, drivers, compute capability, AI libraries, and compatibility work.',
    'seoKeywords', jsonb_build_array('what is CUDA', 'CUDA explained', 'CUDA programming', 'CUDA Toolkit vs driver', 'CUDA cores vs Tensor Cores', 'CUDA compute capability', 'CUDA version compatibility', 'CUDA for AI', 'CUDA vs ROCm', 'CUDA kernel threads blocks grids', 'CUDA memory hierarchy', 'nvidia-smi CUDA version', 'NVCC PTX SASS')
  ),
  published_at = DATE '2026-08-31',
  updated_at = NOW()
FROM cuda_entry
WHERE item.kind = 'glossary'
  AND item.slug = 'cuda'
  AND item.parent_slug = '';
