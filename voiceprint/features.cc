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

#include "voiceprint/features.h"

#include <algorithm>

namespace openspeaker {

void Features::ApplyMean(std::vector<std::vector<float>>* feats) {
  std::vector<float> mean(feats_dims_, 0);
  for (auto& i : *feats) {
    std::transform(i.begin(), i.end(), mean.begin(), mean.begin(),
                   std::plus<>{});
  }
  std::transform(mean.begin(), mean.end(), mean.begin(),
                 [&](const float d) { return d / feats->size(); });
  for (auto& i : *feats) {
    std::transform(i.begin(), i.end(), mean.begin(), i.begin(), std::minus<>{});
  }
}

void Features::ExtractFeatures(const std::string& wav_path,
                               std::vector<std::vector<float>>* chunk_feats) {
  // Read wav and extract features.
  wenet::WavReader wav_reader(wav_path);
  wenet::FeaturePipelineConfig config(feats_dims_, sample_rate_);
  wenet::FeaturePipeline feature_pipeline(config);
  feature_pipeline.Reset();
  feature_pipeline.AcceptWaveform(std::vector<float>(std::vector<float>(
      wav_reader.data(), wav_reader.data() + wav_reader.num_sample())));
  feature_pipeline.set_input_finished();
  feature_pipeline.Read(std::numeric_limits<int>::max(), chunk_feats);
  ApplyMean(chunk_feats);
}

}  // namespace openspeaker
