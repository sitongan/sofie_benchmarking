onnxruntime benchmark example for tmva SOFIE

to build onnxruntime:
git clone --recursive https://github.com/Microsoft/onnxruntime
run <path>\build.sh --config Release --build_shared_lib --parallel
change the <path> in the Makefile
