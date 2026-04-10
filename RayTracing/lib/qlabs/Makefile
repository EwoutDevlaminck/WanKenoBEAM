
# Python wrapper for the qlabs EC absorption module.
# Builds a shared library callable from Python via f2py.
#
# Build order:
#   1. ecdisp  — compile Farina's warmdisp library (already in ../ecdisp/)
#   2. qlabs   — compile qlabs_bessel.f90, qlabs_edf.f90, qlabs.f90
#               into libqlabs.a
#   3. wrap    — generate the f2py Python extension qlabsECabsorption.so
#
# Usage:
#   make build    (runs all three steps)
#   make wrap     (f2py wrapper only; assumes libqlabs.a and libecdisp.a exist)
#   make clean    (remove build artefacts)

include $(CURDIR)/../../../config.mk

FFLAGS = -O3 -fdefault-real-8 -fPIC
ECDISP_DIR = $(CURDIR)/../ecdisp

build: ecdisp qlabs wrap

# Step 1: ensure ecdisp library is up-to-date
ecdisp:
	@echo 'Building ecdisp library ...'
	$(MAKE) -C $(ECDISP_DIR) ecdisp
	@echo 'ecdisp ready.'

# Step 2: compile the three qlabs Fortran source files
qlabs:
	@echo 'Building qlabs modules ...'
	gfortran -c qlabs_bessel.f90 $(FFLAGS)
	gfortran -c qlabs_edf.f90    $(FFLAGS)
	gfortran -c qlabs.f90        $(FFLAGS) -I$(ECDISP_DIR)
	rm -f libqlabs.a
	ar rcv libqlabs.a *.o
	rm -f *.o
	@echo 'qlabs library built.'

# Step 3: f2py wrapper
#
# f2py is given qlabs_wrap.f90 (standalone subroutines, not inside a module).
# These have simple linker symbols qlabs_init_ / qlabs_absorption_ that f2py
# can call directly.  The compiled qlabs module code is provided via -lqlabs.
#
# The Fortran compiler needs qlabs.mod (generated in step 2) to compile
# qlabs_wrap.f90; -I$(CURDIR) and -I$(ECDISP_DIR) point to the .mod files.
wrap:
	@echo 'Running f2py ...'
	$(FTOPY) -m qlabsECabsorption -c qlabs_wrap.f90 \
	         -L$(CURDIR) -lqlabs \
	         -I$(ECDISP_DIR) -L$(ECDISP_DIR) -lecdisp \
	         --f90flags="-fdefault-real-8 -I$(CURDIR) -I$(ECDISP_DIR)"
	@echo 'f2py wrapper complete.'

clean:
	rm -f *.o *.mod *.a *.so qlabs_wrap.f90.bak
