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

#include "gflags/gflags.h"
#include "glog/logging.h"
#include "torch/script.h"
#include "torch/torch.h"

#include "frontend/feature_pipeline.h"
#include "frontend/wav.h"
#include "voiceprint/features.h"

DEFINE_string(model, "../model/tdnn.pt", "Path to voiceprint model.");
DEFINE_string(enroll_wav, "../test_data/BAC009S0749W0480.wav",
              "First wav as enroll wav.");
DEFINE_string(test_wav, "../test_data/BAC009S0749W0489.wav",
              "Second wav as test wav.");
DEFINE_uint32(sample_rate, 16000, "Wav sample rate supported.");
DEFINE_uint32(feats_dims, 24, "Dims for input features.");

using TorchModule = torch::jit::script::Module;

void FeedForward(std::vector<std::vector<float>>* chunk_feats,
                 const std::string& model_path, std::vector<float>* embedding) {
  // Convert std::vector features to torch::Tensor as the scripted model need
  // Tensor as input.
  TorchModule model = torch::jit::load(model_path);
  torch::Tensor feats =
      torch::zeros({1, static_cast<int>(chunk_feats->size()), FLAGS_feats_dims},
                   torch::kFloat32);
  for (size_t i = 0; i < chunk_feats->size(); ++i) {
    torch::Tensor row = torch::from_blob(chunk_feats->at(i).data(),
                                         {FLAGS_feats_dims}, torch::kFloat32)
                            .clone();
    feats[0][i] = std::move(row);
  }
  torch::NoGradGuard no_grad;
  model.eval();
  torch::Tensor outputs = model.forward({feats}).toTensor();
  auto accessor = outputs.accessor<float, 3>();

  for (int i = 0; i < outputs.size(2); ++i) {
    embedding->push_back(accessor[0][0][i]);
  }
}

float CosineSimilarity(const std::vector<float>& embedding1,
                       const std::vector<float>& embedding2) {
  CHECK_EQ(embedding1.size(), embedding2.size());
  float dot = 0.f;
  float emb1_sum = 0.f;
  float emb2_sum = 0.f;
  for (size_t i = 0; i < embedding1.size(); ++i) {
    dot += embedding1[i] * embedding2[i];
    emb1_sum += embedding1[i] * embedding1[i];
    emb2_sum += embedding2[i] * embedding2[i];
  }
  dot /= std::max(std::sqrt(emb1_sum * emb2_sum),
                  std::numeric_limits<float>::epsilon());
  return dot;
}

int main(int argc, char* argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, false);
  google::InitGoogleLogging(argv[0]);

  // Load and init model
  std::string model_path = FLAGS_model;
  unsigned int sample_rate = FLAGS_sample_rate;
  unsigned int feats_dims = FLAGS_feats_dims;

  std::string wav_path = FLAGS_enroll_wav;
  std::vector<std::vector<float>> chunk_feats;
  OpenSpeaker::Features features(sample_rate, feats_dims);
  features.ExtractFeatures(wav_path, &chunk_feats);

  std::vector<float> enroll_embedding;
  FeedForward(&chunk_feats, model_path, &enroll_embedding);
  LOG(INFO) << "Enroll wav embedding:" << enroll_embedding;

  wav_path = FLAGS_test_wav;
  chunk_feats.clear();
  features.ExtractFeatures(wav_path, &chunk_feats);
  std::vector<float> test_embedding;
  FeedForward(&chunk_feats, model_path, &test_embedding);
  LOG(INFO) << "Test wav embedding:" << test_embedding;

  // Compute cosine similarity
  float similarity = CosineSimilarity(enroll_embedding, test_embedding);
  LOG(INFO) << "Cosine similarity: " << similarity;
  return 0;
}