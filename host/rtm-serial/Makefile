################################################################################
# Automatically-generated file. Do not edit!
################################################################################

CWPROOT = ../../lib/cwp
IDIR = $(CWPROOT)/include
LDIR = $(CWPROOT)/lib

C_SRCS =
OBJS =

C_SRCS += \
./src/fd.c \
./src/mod_main.c 


OBJ += \
./src/fd \
./src/mod_main 


CFLAGS = -I"$(IDIR)" -I"../include" -I"./include" -O3 -ggdb -Wall -Wextra -fmessage-length=0
#CFLAGS = -I"$(IDIR)" -I"../include" -I"./include" -O3 -Wall -Wextra -fmessage-length=0
LDFLAGS = -L"$(LDIR)" -L"../lib" 
LIBS = -lsu -lpar -lcwp -lm -lrtm-base

RM := rm -rf

# Add inputs and outputs from these tool invocations to the build variables 

# All Target

all: rtm_serial

# Tool invocations
rtm_serial: $(USER_OBJS)
	@echo 'Building file: $<'
	@echo 'Invoking: GCC C Linker'

	# remember: libs must be included AFTER .c files
	gcc -o $@ $(CFLAGS) $(LDFLAGS) $(C_SRCS) $(LIBS)
	@echo 'Finished building target: $@'
	@echo ' '

# Other Targets
clean:
	-$(RM) $(ARCHIVES) $(OBJS) $(C_DEPS) rtm_serial
	-@echo ' '

.PHONY: all clean dependents
