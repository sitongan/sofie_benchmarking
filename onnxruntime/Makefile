CXX = g++
CPPFLAGS = -std=c++14 -g
ONNXR_I = /cvmfs/sft.cern.ch/lcg/releases/onnxruntime/1.6.0-520ec/x86_64-centos8-gcc10-opt/include/core/session
ONNXR_L = /cvmfs/sft.cern.ch/lcg/releases/onnxruntime/1.6.0-520ec/x86_64-centos8-gcc10-opt/lib

run: run.cxx
	${CXX} ${CPPFLAGS} -I${ONNXR_I} -L${ONNXR_L}  -o run run.cxx -l:libonnxruntime.so

%.o: %.cxx
	${CXX} ${CPPFLAGS} -c $< -I${ONNXR_I} -L${ONNXR_L}

.phony: clean
clean:
	-rm *.d
	-rm *.o

#for macos
#export DYLD_LIBRARY_PATH=/Users/sitongan/rootdev/onnx_runtime_cxx/onnxruntime/build/MacOS/Release/
#before building
