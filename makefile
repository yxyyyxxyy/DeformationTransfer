CXX=icc
LIB_OPENMESH=/usr/local/lib/libOpenMeshCore.a /usr/local/lib/libOpenMeshTools.a 

dt: deformation_transfer.cpp $(LIB_OPENMESH)
	$(CXX) -DOM_STATIC_BUILD -o $@ $^ -I${MKLROOT}/include -Wl,--start-group ${MKLROOT}/lib/intel64/libmkl_intel_lp64.a ${MKLROOT}/lib/intel64/libmkl_intel_thread.a ${MKLROOT}/lib/intel64/libmkl_core.a ${MKLROOT}/lib/intel64/libmkl_blacs_intelmpi_lp64.a -Wl,--end-group -liomp5 -lpthread -lm -ldl -fopenmp -O3
