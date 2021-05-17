#include <onnxruntime_cxx_api.h>
#include <iostream>
#include <chrono>
#include <stdlib.h>
#include <time.h>
#include <cmath>
#include <numeric>

int main(){

   const int batchsize = 1;
   const int n = 1000;

   Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test");
   Ort::SessionOptions session_options;
   session_options.SetIntraOpNumThreads(1);
   session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

   const char* model_path = "Linear_event.onnx";

   printf("Using Onnxruntime C++ API\n");
   Ort::AllocatorWithDefaultOptions allocator;
   Ort::Session session(env, model_path, session_options);
   std::vector<const char*> input_node_names(1);
   std::vector<const char*> output_node_names(1);
   input_node_names[0] = session.GetInputName(0, allocator);
   output_node_names[0] = session.GetOutputName(0, allocator);
   Ort::TypeInfo type_info = session.GetInputTypeInfo(0);
   auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
   std::vector<int64_t> input_node_dims = tensor_info.GetShape();
   std::cout << "input size: ";
   for (auto &i: input_node_dims){
      std::cout << i << ",";
   }
   std::cout <<std::endl;



   size_t input_tensor_size = batchsize*100;
   std::vector<std::vector<float>> input_tensor_values;
   input_tensor_values.resize(n, std::vector<float>(input_tensor_size));

   for (int k = 0; k < n; k++){
      //k = 0;
      for (unsigned int i = 0; i < input_tensor_size; i++){
         srand(time(0));
         input_tensor_values[k][i] = rand();
      }
   }


   auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
   std::vector<Ort::Value> input_tensor;
   for (int k = 0; k < n; k++){
      input_tensor.push_back(Ort::Value::CreateTensor<float>(memory_info, input_tensor_values[k].data(), input_tensor_size, input_node_dims.data(), input_node_dims.size()));

   }
   //std::vector<Ort::Value> output_tensor;
   std::vector<float> total_time;

   bool flash_cache = true;
   const size_t bigger_than_cachesize = 10 * 1024 * 1024;
   //Ort::Value output_tensor;
   for (int k =0; k < n; k++){

//#define FLASH_CACHE

#ifdef FLASH_CACHE
      std::vector<float> tmp(bigger_than_cachesize);
      // When you want to "flush" cache.
      for(int i = 0; i < bigger_than_cachesize; i++)
      {
         tmp[i] = rand();
      }
#endif

      auto t1 = std::chrono::high_resolution_clock::now();
      auto output_tensor = session.Run(Ort::RunOptions{nullptr}, input_node_names.data(), &(input_tensor[k]), 1, output_node_names.data(), 1);
      auto t2 = std::chrono::high_resolution_clock::now();
      total_time.push_back(float(std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count()));
   }

   //float* floatarr = output_tensors.front().GetTensorMutableData<float>();
   //for (int i = 0; i < 10; i++)
     //printf("%f\t", i, floatarr[i]);

  float sum = std::accumulate(total_time.begin(), total_time.end(), 0.0);
  float mean = sum / total_time.size();

  float sq_sum = std::inner_product(total_time.begin(), total_time.end(), total_time.begin(), 0.0);
  float std = std::sqrt(sq_sum / total_time.size() - mean * mean);


  std::cout << std::endl << mean / batchsize << std::endl;
  std::cout << std / batchsize << std::endl;

}
