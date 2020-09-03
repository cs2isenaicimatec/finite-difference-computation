################################################################################
# Automatically-generated file. Do not edit!
################################################################################

CWPROOT = ../../lib/cwp
IDIR = $(CWPROOT)/include
LDIR = $(CWPROOT)/lib

C_SRCS += \
 ./src/ptsrc.c \
 ./src/taper.c 
OBJS += \
 ./src/ptsrc.o \
 ./src/taper.o 
C_DEPS += \
 ./src/ptsrc.d \
 ./src/taper.d 

CFLAGS = -I$(IDIR) -I../include -L$(LDIR) -Llib/ -O3 -g3 -Wall -Wextra -c -fmessage-length=0 -pedantic
LIBS = -lsu -lpar -lcwp -lm

RM := rm -rf

# All Target
all: librtm-base.a


# Tool invocations
librtm-base.a: $(OBJS) $(USER_OBJS)
	@echo 'Building file: $<'
	@echo 'Invoking: GCC C Compiler'
	gcc -I"./include" $(CFLAGS)  -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

	@echo 'Building target: $@'
	@echo 'Invoking: GCC Archiver'
	ar -r  "../lib/librtm-base.a"  $(OBJS) $(USER_OBJS)
	-$(RM) $(ARCHIVES)$(OBJS)$(C_DEPS)
	@echo 'Finished building target: ../lib/$@'
	@echo ' '

# Other Targets
clean:
	-$(RM) $(ARCHIVES)$(OBJS)$(C_DEPS) librtm-base.a
	-@echo ' '

.PHONY: all clean dependents
