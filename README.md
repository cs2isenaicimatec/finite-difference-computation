# STENCIL CODE

The source of the simples codes in CUDA and DPC++ are inside,respectively, the folders cuda_refenrence_stencil_computation, dpct_migrated_stencil_computation and dpct_migrated_stencil_computation_with_buffers.

## COMPILE

To these simple codes there is no need to compile a library.

### CUDA

There is a `Makefile` inside the folder that can used to compile. To use the make command you need to be inside the folder that contains `Makefile`. An executble file is created inside the current folder.

### DPC++

There is a `Makefile` inside the folder that can used to compile. To use the make command you need to be inside the folder that contains `Makefile`. An executble file is created inside the current folder.

# RTM CODE

The folders of RTM codes in CUDA and DPC++ has four subdirectories: library (lib), velocity models (models), output and source (src).

## COMPILE

To compile and run the RTM code needs fist compile a library that has some auxilary functions. There is a `build.sh` file in lib/src that can executed or you can use make in the same path, you need to be inside the folder that contains the file. If the compilation has finished without errors and produced an libsource.a in lib folder.

### CUDA

There is a `build.sh` file in src that can executed or you can use make in the same path. To use `build.sh` file or the make command you need to be inside the folder that contains `Makefile`. An executble file is created inside the current folder.

### DPC++

There is a `build.sh` file in src that can executed or you can use make in the same path. To use `build.sh` file or the make command you need to be inside the folder that contains `Makefile`. An executble file is created inside the current folder.

> You may notice that some flags are missing in the compilation command in DPC ++ , only one flag match was found: `--fmad = false`. This flag was inserted in the main function of the source code in DPC++:  `_MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);`
