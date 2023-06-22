# `Concurrent Kernels` Sample

The `Concurrent Kernels` sample demonstrates the use of SYCL queues for concurrent execution of several kernels on GPU device. It is implemented using SYCL by migrating code from original CUDA source code for offloading computations to a GPU or CPU and further demonstrates how to optimize and improve processing time.

| Area                   | Description
|:---                    |:---
| What you will learn    | How to begin migrating CUDA to SYCL
| Time to complete       | 15 minutes

## Purpose

The `Concurrent Kernels` sample shows the execution of multiple kernels on the device at the same time.

> **Note**: We use Intel® open-sources SYCLomatic tool which assists developers in porting CUDA code automatically to SYCL code. To finish the process, developers complete the rest of the coding manually and then tune to the desired level of performance for the target architecture. User's can also use SYCLomatic Tool which comes along with the Intel® oneAPI Base Toolkit.

This sample contains two versions of the code in the following folders:

| Folder Name                   | Description
|:---                           |:---
| `01_dpct_output`              | Contains output of SYCLomatic Tool used to migrate SYCL-compliant code from CUDA code. This SYCL code has some unmigrated code that has to be manually fixed to get full functionality. (The code does not functionally work as supplied.)
| `02_sycl_migrated`            | Contains manually migrated SYCL code from CUDA code.

### Workflow For CUDA to SYCL migration

Refer [Workflow](https://www.intel.com/content/www/us/en/developer/tools/oneapi/training/cuda-sycl-migration-workflow.html#gs.s2njvh) for details.

### CUDA source code evaluation

This code demonstrates the use of CUDA streams for concurrent execution of multiple kernels. It creates multiple streams, each associated with a separate kernel, and uses cudaStreamWaitEvent to introduce dependencies between the streams. The code measures the elapsed time for both serial and concurrent execution of the kernels. The kernel has a loop which iterates for specific number of times without performing any actual work. Finally, the code verifies if the concurrent execution of kernels was faster than the serial execution based on the measured times.

This sample is migrated from NVIDIA CUDA sample. See the [concurrentKernels](https://github.com/NVIDIA/cuda-samples/tree/master/Samples/0_Introduction/concurrentKernels) sample in the NVIDIA/cuda-samples GitHub.

## Prerequisites

| Optimized for              | Description
|:---                        |:---
| OS                         | Ubuntu* 20.04
| Hardware                   | Intel® Gen9, Gen11 and Intel® Xeon(R) Gold 6128 CPU
| Software                   | SYCLomatic version 2023.1, Intel oneAPI Base Toolkit version 2023.1

For more information on how to install SYCLomatic Tool, visit [Migrate from CUDA* to C++ with SYCL*](https://www.intel.com/content/www/us/en/developer/tools/oneapi/training/migrate-from-cuda-to-cpp-with-sycl.html#gs.v3584e).

## Key Implementation Details

This sample demonstrates the migration of the following prominent CUDA feature: 
- Streams and Event Management
- Reduction

concurrentKernels involves a kernel that does no real work but runs at least for a specified number of iterations.

## Build the `Concurrent Kernels` Sample for CPU and GPU

> **Note**: If you have not already done so, set up your CLI
> environment by sourcing  the `setvars` script in the root of your oneAPI installation.
>
> Linux*:
> - For system wide installations: `. /opt/intel/oneapi/setvars.sh`
> - For private installations: ` . ~/intel/oneapi/setvars.sh`
> - For non-POSIX shells, like csh, use the following command: `bash -c 'source <install-dir>/setvars.sh ; exec csh'`
>
> For more information on configuring environment variables, see [Use the setvars Script with Linux* or macOS*](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-the-setvars-script-with-linux-or-macos.html).

## Tool assisted migration – SYCLomatic 

For this sample, the SYCLomatic Tool automatically migrates 100% of the CUDA code to SYCL. Follow these steps to generate the SYCL code using the compatibility tool:

1. git clone https://github.com/NVIDIA/cuda-samples.git
2. cd cuda-samples/Samples/0_Introduction/concurrentKernels/
3. Generate a compilation database with intercept-build
   ```
   intercept-build make
   ```
4. The above step creates a JSON file named compile_commands.json with all the compiler invocations and stores the names of the input files and the compiler options.
5. Pass the JSON file as input to the Intel® SYCLomatic Compatibility Tool. The result is written to a folder named dpct_output. The --in-root specifies path to the root of the source tree to be migrated.
   ```
   c2s -p compile_commands.json --in-root ../../.. --use-custom-helper=api
   ```

### On Linux*

1. Change to the sample directory.
2. Build the program.
   ```
   $ mkdir build
   $ cd build
   $ cmake ..
   $ make
   ```

   By default, this command sequence will build the `01_dpct_output`, `02_sycl_migrated` version of the program.

3. Run the program.
   
   Run `01_dpct_output` on GPU.
   ```
   make run
   ```  
   Run `01_dpct_output` on CPU.
   ```
   export ONEAPI_DEVICE_SELECTOR=cpu
   make run
   unset ONEAPI_DEVICE_SELECTOR 
   ```
 4. Run the program.
   
   Run `02_sycl_migrated` on GPU.
   ```
   make run_sm
   ```  
   Run `02_sycl_migrated` on CPU.
   ```
   export ONEAPI_DEVICE_SELECTOR=cpu
   make run_sm
   unset ONEAPI_DEVICE_SELECTOR 
   ```
   
#### Troubleshooting

If an error occurs, you can get more details by running `make` with
the `VERBOSE=1` argument:
```
make VERBOSE=1
```
If you receive an error message, troubleshoot the problem using the **Diagnostics Utility for Intel® oneAPI Toolkits**. The diagnostic utility provides configuration and system checks to help find missing dependencies, permissions errors, and other issues. See the [Diagnostics Utility for Intel® oneAPI Toolkits User Guide](https://www.intel.com/content/www/us/en/develop/documentation/diagnostic-utility-user-guide/top.html) for more information on using the utility.


### Example Output

The following example is for `02_sycl_migrated` for GPU on **Intel(R) UHD Graphics [0x9a60]**.
```
[./a.out] - Starting...
Device: Intel(R) UHD Graphics P630 [0x3e96]
> Detected Compute SM 3.0 hardware with 24 multi-processors
Expected time for serial execution of 8 kernels = 0.080s
Expected time for concurrent execution of 8 kernels = 0.010s
Measured time for sample = 0.08594s
Test passed
```
## License
Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third party program licenses are at [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt).
