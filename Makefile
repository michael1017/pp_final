NVFLAGS  := -std=c++11 -O3 -Xptxas="-v" -arch=sm_70 -Xcompiler -fopenmp 
LDFLAGS  := -lm
EXES     := jacobi seq

alls: $(EXES)

clean:
	rm -f $(EXES)


seq :seq.cc
	nvcc $(NVFLAGS) $(LDFLAGS) -o seq seq.cc
jacobi: jacobi.cu 
	nvcc $(NVFLAGS) $(LDFLAGS) -o jacobi jacobi.cu 
