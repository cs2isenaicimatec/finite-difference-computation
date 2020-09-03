################################################################################
# Automatically-generated file. Do not edit!
################################################################################
THRDDIR = ../../lib
CWPROOT = $(THRDDIR)/cwp
CLROOT = $(THRDDIR)/opencl
IDIR = $(CWPROOT)/include
LDIR = $(CWPROOT)/lib

C_SRCS =
OBJS =

C_SRCS += \
./src/fd.c \
./src/fdlap.c \
./src/clmisc.c  \
./src/fdmisc.c  \
./src/mod_main.c 

OBJ += \
./src/fd \
./src/fdlap \
./src/fdmisc  \
./src/clmisc  \
./src/mod_main 

DFLAGS = -D$(USE_FPGA)=1 -D$(UPDATE_PP)=1
CFLAGS = -w -I"$(IDIR)" -I"../include" -I"./include" -I"$(CLROOT)/include" -O3 -ggdb -Wall -Wextra -fmessage-length=0 -w
LDFLAGS = -L"$(LDIR)" -L"../lib" -L"$(CLROOT)/lib" 
LIBS = -lsu -lpar -lcwp -lm -lrtm-base -lOpenCL

RM := rm -rf

# Add inputs and outputs from these tool invocations to the build variables 

# All Target

all: rtm_opencl

# Tool invocations
rtm_opencl: $(USER_OBJS)
	@echo 'Building file: $<'
	@echo 'Invoking: GCC C Linker'

	# remember: libs must be included AFTER .c 
	gcc -o $@ $(CFLAGS) $(DFLAGS) $(LDFLAGS) $(C_SRCS) $(LIBS)
	@echo 'Finished building target: $@'
	@echo ' '

# Other Targets
clean:
	-$(RM) $(ARCHIVES) $(OBJS) $(C_DEPS) rtm_opencl
	-@echo ' '

.PHONY: all clean dependents
