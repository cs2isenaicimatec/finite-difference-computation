################################################################################
# Automatically-generated file. Do not edit!
################################################################################

CWPROOT = ../../lib/cwp
CWP_INC = -I$(CWPROOT)/include -I$(CWPROOT)/src/Complex/include
IDIR = $(CWPROOT)/include
LDIR = $(CWPROOT)/lib

C_SRCS =
OBJS =

C_SRCS += \
./src/fd.cu \
./src/mod_main.cu 


OBJ += \
./src/fd \
./src/mod_main 


CFLAGS = -I"$(IDIR)" "$(CWP_INC)" -I"../include" -I"./include" -O3 -w --ftz=false --prec-div=true --fmad=false -arch=sm_60
#CFLAGS = -I"$(IDIR)" -I"../include" -I"./include" -O3 -Wall -Wextra -fmessage-length=0
LDFLAGS = -L"$(LDIR)" -L"../lib" 
LIBS = -lsu -lpar -lcwp -lm -lrtm-base

RM := rm -rf

# Add inputs and outputs from these tool invocations to the build variables 

# All Target

all: rtm_cuda

# Tool invocations
rtm_cuda: $(USER_OBJS)
	@echo 'Building file: $<'
	@echo 'Invoking: GCC C Linker'

	# remember: libs must be included AFTER .c files
	nvcc -o $@ $(CFLAGS) $(LDFLAGS) $(C_SRCS) $(LIBS)
	@echo 'Finished building target: $@'
	@echo ' '

# Other Targets
clean:
	-$(RM) $(ARCHIVES) $(OBJS) $(C_DEPS) rtm_cuda
	-@echo ' '

.PHONY: all clean dependents
