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

#ifndef VOICEPRINT_FEATURES_H_
#define VOICEPRINT_FEATURES_H_

#include <string>
#include <vector>

#include "frontend/feature_pipeline.h"
#include "frontend/wav.h"

namespace openspeaker {
class Features {
 public:
  Features(unsigned int sample_rate, unsigned int feats_dims)
      : sample_rate_(sample_rate), feats_dims_(feats_dims) {}
  ~Features() = default;

  void ExtractFeatures(const std::string& wav_path,
                       std::vector<std::vector<float>>* chunk_feats);

 private:
  unsigned int sample_rate_;
  unsigned int feats_dims_;

  void ApplyMean(std::vector<std::vector<float>>* feats);

  // Disallow copy and assign
  Features(const Features&) = delete;
  Features& operator=(const Features&) = delete;
};
}  // namespace openspeaker

#endif  // VOICEPRINT_FEATURES_H_
