################################################################################
# Automatically-generated file. Do not edit!
################################################################################
#THRDDIR = /scratch/alu
THRDDIR = ../..
CWPROOT = $(THRDDIR)/lib/cwp
CLROOT = $(THRDDIR)/lib/opencl
IDIR = $(CWPROOT)/include
LDIR = $(CWPROOT)/lib

C_SRCS =
OBJS =


C_SRCS += \
./src/fd.c \
./src/rtmcl.c \
./src/misc.c \
./src/ptsrc.c \
./src/taper.c  \
./src/rtm.c  \
./src/main.c 

OBJ += \
./src/fd \
./src/rtmcl \
./src/misc \
./src/ptsrc \
./src/taper  \
./src/rtm  \
./src/main 

DFLAGS = -D$(USE_FPGA)=1 -DOPENCLRTM=1
CFLAGS = -I"$(IDIR)" -I"$(CLROOT)/include" -I"./include" -I"/usr/lib/openmpi/include/" -O3 -ggdb -Wall -Wextra -fmessage-length=0 -w
LDFLAGS = -L"$(LDIR)" -L"../lib" -L"$(CLROOT)/lib" 
LIBS = -lsu -lpar -lcwp -lm -lOpenCL

RM := rm -rf

# Add inputs and outputs from these tool invocations to the build variables 

# All Target

all: rtm_modmig

# Tool invocations
rtm_modmig: $(USER_OBJS)
	@echo 'Building file: $<'
	@echo 'Invoking: GCC C Linker'

	# remember: libs must be included AFTER .c 
ifeq ($(CC),mpicc)
		${CC} -o $@ $(CFLAGS) $(DFLAGS) $(LDFLAGS) $(C_SRCS) $(LIBS) -DUSE_MPI=1
else
		${CC} -o $@ $(CFLAGS) $(DFLAGS) $(LDFLAGS) $(C_SRCS) $(LIBS)
endif
	
	@echo 'Finished building target: $@'
	@echo ' '

# Other Targets
clean:
	-$(RM) $(ARCHIVES) $(OBJS) $(C_DEPS) rtm_modmig
	-@echo ' '

.PHONY: all clean dependents
