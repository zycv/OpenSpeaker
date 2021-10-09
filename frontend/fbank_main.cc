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

#include "glog/logging.h"

#include "frontend/feature_pipeline.h"
#include "frontend/wav.h"

int main(int argc, char *argv[]) {
  google::InitGoogleLogging(argv[0]);
  LOG(INFO) << "Hello";
  std::string audio_wav = "../test_data/BAC009S0749W0489.wav";
  wenet::WavReader wav_reader(audio_wav);
  wenet::FeaturePipelineConfig config(80, 16000);
  wenet::FeaturePipeline feature_pipeline(config);
  feature_pipeline.Reset();
  feature_pipeline.AcceptWaveform(std::vector<float>(std::vector<float>(
      wav_reader.data(), wav_reader.data() + wav_reader.num_sample())));
  feature_pipeline.set_input_finished();
  std::vector<std::vector<float>> chunk_feats;
  feature_pipeline.Read(std::numeric_limits<int>::max(), &chunk_feats);
  for (const auto &feats : chunk_feats) {
    for (auto f : feats) {
      LOG(INFO) << f;
    }
    break;
  }
  return 0;
}