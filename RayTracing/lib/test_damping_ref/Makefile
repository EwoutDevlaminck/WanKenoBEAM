
# Builds damping_refECabsorption.so — a Python/f2py extension that exposes
# damping_ref_ec(), a subroutine-version of Shkarofsky & Shoucri (2011) damping.f
# with NAG Bessel replaced by qlabs_bessel.
#
# Usage:
#   make build    — compile and wrap
#   make clean    — remove build artefacts

include $(CURDIR)/../../../config.mk   # provides PYINT, FTOPY

FFLAGS  = -O3 -fdefault-real-8 -fPIC
QLABS   = $(CURDIR)/../qlabs
BESSEL_SRC = $(QLABS)/qlabs_bessel.f90

build:
	@echo "Compiling qlabs_bessel ..."
	gfortran -c $(BESSEL_SRC) $(FFLAGS)
	@echo "Compiling damping_ref ..."
	gfortran -c damping_ref.f90 $(FFLAGS)
	rm -f libdamping_ref.a
	ar rcv libdamping_ref.a *.o
	rm -f *.o
	@echo "Running f2py ..."
	$(FTOPY) -m damping_refECabsorption -c damping_ref.f90 \
	         -L$(CURDIR) -ldamping_ref \
	         --f90flags="-fdefault-real-8 -I$(CURDIR)"
	@echo "Build complete."

clean:
	rm -f *.o *.mod *.a *.so
