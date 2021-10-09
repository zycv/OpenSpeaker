# OpenSpeaker

OpenSpeaker is a completely independent and open source speaker recognition project. It provides the entire process of speaker recognition including data preparation, model training, multi-platform deployment and model optimization.

# Build

- Clone the repo

``` sh
git clone https://github.com/zycv/OpenSpeaker.git
```

- Build 

``` sh
cd OpenSpeaker && mkdir build && cd build && cmake .. && cmake --build . -j8
```
- Run

``` sh
GLOG_logtostderr=1 ./voiceprint_main 
```

- Output

Then you will see output like this:

> I1010 00:04:44.024065  5598 voiceprint_main.cc:107] `Enroll wav embedding`:-0.000527721 -42.1776 0.383387 -4.88499 -35.6465 22.2982 28.4302 0.335538 -7.77056 21.861 0.852521 38.6987 -33.6933 8.14212 2.1728 5.32039 -31.5929 26.2519 1.12099 17.0451 -34.2525 28.0443 2.62836 13.5042 24.5696 8.05435 1.86737 14.1633 15.315 15.0323 -0.243613 22.7958 23.3888 -3.52539 -4.50719 2.26 22.6081 16.9342 -3.73238 -2.30486 34.241 -4.34527 -8.53935 13.4037 -5.2506 31.0014 0.477698 19.1969 -0.0354049 8.56949 -0.00334034 -0.557936 -18.4449 11.7907 -28.2117 9.47196 -17.3517 36.2643 -1.56259 21.1091 -32.0706 -31.3819 1.85436 -0.00354689 4.80609 21.506 -5.79249 0.25165 -55.7074 -4.32717e-05 18.4236 -16.799 -30.733 1.3678 8.01844 -25.2722 20.525 -1.54043 11.9003 -2.97013 4.9329 5.92645 27.0518 -2.54181 -0.00735733 4.37697 -0.149234 15.4831 15.8355 0.597157 -12.4455 -23.3576 12.2849 -0.0110179 -18.9306 -0.0170257 -4.67177e-06 -22.8124 14.2268 -34.0223 ...


> I1010 00:04:44.024065  5598 voiceprint_main.cc:107] `Test wav embedding`:-0.000527721 -42.1776 0.383387 -4.88499 -35.6465 22.2982 28.4302 0.335538 -7.77056 21.861 0.852521 38.6987 -33.6933 8.14212 2.1728 5.32039 -31.5929 26.2519 1.12099 17.0451 -34.2525 28.0443 2.62836 13.5042 24.5696 8.05435 1.86737 14.1633 15.315 15.0323 -0.243613 22.7958 23.3888 -3.52539 -4.50719 2.26 22.6081 16.9342 -3.73238 -2.30486 34.241 -4.34527 -8.53935 13.4037 -5.2506 31.0014 0.477698 19.1969 -0.0354049 8.56949 -0.00334034 -0.557936 -18.4449 11.7907 -28.2117 9.47196 -17.3517 36.2643 -1.56259 21.1091 -32.0706 -31.3819 1.85436 -0.00354689 4.80609 21.506 -5.79249 0.25165 -55.7074 -4.32717e-05 18.4236 -16.799 -30.733 1.3678 8.01844 -25.2722 20.525 -1.54043 11.9003 -2.97013 4.9329 5.92645 27.0518 -2.54181 -0.00735733 4.37697 -0.149234 15.4831 15.8355 0.597157 -12.4455 -23.3576 12.2849 -0.0110179 -18.9306 -0.0170257 -4.67177e-06 -22.8124 14.2268 -34.0223 ...


>I1010 00:04:44.358219  5598 voiceprint_main.cc:118] `Cosine similarity`: 0.975539

The `Cosine similarity` in the last line indicates the similarity of the current two speakers.

- Optional

If you want to test other audio or run with another model, you can run:

``` sh
GLOG_logtostderr=1 ./voiceprint_main --help
```

Then you will see the help information as follows:

``` sh
-enroll_wav (First wav as enroll wav.) type: string
    default: "../test_data/BAC009S0749W0480.wav"

-feats_dims (Dims for input features.) type: uint32 default: 24

-model (Path to voiceprint model.) type: string default: "../model/tdnn.pt"

-sample_rate (Wav sample rate supported.) type: uint32 default: 16000

-test_wav (Second wav as test wav.) type: string
    default: "../test_data/BAC009S0749W0489.wav"

```