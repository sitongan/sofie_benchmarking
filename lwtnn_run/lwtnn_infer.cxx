#include "lwtnn/LightweightGraph.hh"
#include "lwtnn/parse_json.hh"
#include <fstream>
#include <string>
#include <iostream>
#include <chrono>
#include <stdlib.h>
#include <time.h>
#include <cmath>
#include <numeric>


int main(){

int n = 1000;

using namespace lwt;

std::ifstream input("linear_event_nn.json");
LightweightGraph graph(parse_json_graph(input));
std::vector<std::map<std::string, std::map<std::string, double> >> inputs(n);

for (int k = 0; k < n; ++k) {
   for (int i = 0; i < 100; i++)
   {
      float x = std::rand() / (RAND_MAX + 1u);
      inputs[k]["node_0"].insert({"variable_" + std::to_string(i), x});
   }
}
std::vector<float> total_time;
bool flash_cache = true;
const size_t bigger_than_cachesize = 10 * 1024 * 1024;

std::vector<std::map<std::string,double>> outputs(n);
//#define FLASH_CACHE
for (int i = 0; i < n; ++i ) {


#ifdef FLASH_CACHE
      std::vector<float> tmp(bigger_than_cachesize);
      // When you want to "flush" cache.
      for(int i = 0; i < bigger_than_cachesize; i++)
      {
         tmp[i] = rand();
      }
#endif
   auto t1 = std::chrono::high_resolution_clock::now();
   outputs[i] = graph.compute(inputs[i]);
   auto t2 = std::chrono::high_resolution_clock::now();

   total_time.push_back(float(std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count()));
}

float sum = std::accumulate(total_time.begin(), total_time.end(), 0.0);
float mean = sum / total_time.size();

float sq_sum = std::inner_product(total_time.begin(), total_time.end(), total_time.begin(), 0.0);
float std = std::sqrt(sq_sum / total_time.size() - mean * mean);


std::cout << std::endl << mean  << std::endl;
std::cout << std  << std::endl;


for (auto& i: outputs){
   //std::cout << i.first << "," <<i.second << "\t";
}
//std::cout << std::endl << duration << std::endl;

return 0;
}
