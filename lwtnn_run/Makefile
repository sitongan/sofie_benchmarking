CXX = g++
CPPFLAGS = -std=c++14 -g

lwtnn_infer: lwtnn_infer.cxx
	${CXX} -o lwtnn_infer $^ ${CPPFLAGS} -I/afs/cern.ch/user/s/sitong/sofie_benchmark/lwtnn/include -L/afs/cern.ch/user/s/sitong/sofie_benchmark/lwtnn/build/lib -l:liblwtnn.so


.phony: clean
clean:
	-rm *.d
	-rm *.o
