// Copyright (c) zouy68@gmail.com(ZY)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <cmath>
#include <iostream>
#include <string>

#include <torch/script.h>
#include <torch/torch.h>

int main(int argc, char *argv[]) {
  torch::DeviceType device_type;
  if (torch::cuda::is_available()) {
    std::cout << "CUDA available! Predicting on GPU." << std::endl;
    device_type = torch::kCUDA;
  } else {
    std::cout << "Predicting on CPU." << std::endl;
    device_type = torch::kCPU;
  }
  torch::Device device(device_type);
  torch::Tensor tensor = torch::eye(3);
  tensor = tensor.to(device_type);
  std::cout << "hello torch" << std::endl;
  std::cout << tensor << std::endl;

  std::string model_path = "";
  torch::jit::script::Module model = torch::jit::load(model_path);
  torch::NoGradGuard no_grad;
  model.eval();
  torch::Tensor feats = torch::zeros({1, 100, 24}, torch::kFloat32);
  torch::Tensor outputs = model.forward({feats}).toTensor();
  auto accessor = outputs.accessor<float, 3>();
  std::vector<float> embedding;
  for (int i = 0; i < outputs.size(2); ++i) {
    embedding.push_back(accessor[0][0][i]);
    std::cout << embedding.size() << " " << accessor[0][0][i] << std::endl;
  }

  return 0;
}