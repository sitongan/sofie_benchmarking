CXX = g++
CPPFLAGS = -std=c++14 -MMD -MP -g
PROTOBUF = `pkg-config --cflags protobuf`
PROTOBUFL = `pkg-config --libs protobuf`
ROOTCONFIG =
ROOTCONFIG2 = `root-config --cflags --glibs`
BLASDIR = /afs/cern.ch/user/s/sitong/sofie_benchmark/BLAS-3.8.0
BLASFLAG = -I${BLASDIR} -lblas_LINUX
SRC = ${wildcard *.cxx}
SOFIEOBEJCT =
SOFIEHEADER =
SOFIE = $(SOFIEOBEJCT) $(SOFIEHEADER)

testinfer: test.cpp
	${CXX} -o testinfer test.cpp -std=c++14 -g -O3 -I ./Eigen/

-include $(SRC:%.cxx=%.d)

%.o: %.cxx
	${CXX} ${CPPFLAGS} -c $< `root-config --cflags` $(PROTOBUF)

.phony: clean
clean:
	-rm *.d
	-rm *.o
