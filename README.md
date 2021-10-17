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

> I1010 00:04:44.024065  5598 voiceprint_main.cc:107] `Enroll wav embedding`:-0.000429691 -43.0401 4.34895 -5.0006 -37.4051 29.0244 25.5545 0.310511 -7.27563 22.3967 0.809033 38.1637 -35.2139 10.3848 2.01294 10.3652 -33.9101 27.9782 4.02987 11.04 -35.7037 29.3087 5.22847 10.328 25.5269 12.7381 3.3835 14.3239 8.97036 19.8363 -0.410791 21.3439 21.0263 -3.17396 -4.10953 8.02122 19.2493 22.9025 -4.43258 -2.36184 35.7907 -3.819 -8.07737 16.0546 -4.59325 29.6196 2.62565 22.2861 -0.0329346 9.01507 -0.003556 -0.44748 -23.2599 8.75573 -22.7237 5.64375 -15.553 27.0606 -1.29388 21.1305 -29.6681 -19.7094 1.61875 -0.00355155 8.97268 20.0471 -3.87054 1.85049 -48.2718 -3.76629e-05 16.503 -14.0614 -27.3728 6.75437 7.99673 -24.249 19.2107 -1.53355 15.0827 0.488096 8.42653 3.9248 22.3116 -2.30652 0.241701 7.35245 -0.222364 19.3998 13.2346 3.91667 -14.5992 -26.711 9.30076 -0.00933527 -20.3823 -0.017337 -4.87047e-06 -20.5478 12.8254 -31.0729 ...


> I1010 00:04:44.024065  5598 voiceprint_main.cc:107] `Test wav embedding`:-0.000517501 -47.1283 7.99776 -5.38489 -36.4123 29.9109 21.3303 0.420559 -8.42374 21.639 1.02079 33.6529 -36.7717 10.2539 2.00366 3.71844 -35.5568 22.5469 8.54491 6.42058 -31.3658 27.0976 3.81696 14.9781 29.1551 11.4731 3.96022 16.805 12.1343 20.7819 -0.375162 35.5286 14.7085 -3.60218 -4.69037 3.30785 16.4984 18.3481 -4.71795 -2.57306 22.5738 -4.06993 -8.78727 15.8166 -4.6445 28.6336 4.23745 21.8884 -0.0407903 9.78283 -0.00302488 -0.515661 -16.6951 10.9236 -24.7703 6.26954 -14.359 34.8278 -1.7668 20.6768 -30.5551 -23.4073 2.07097 -0.002863 6.87817 20.5055 3.0923 1.02305 -51.536 -4.44536e-05 15.3016 -9.54486 -25.2829 1.60107 7.55688 -20.8518 22.0509 -1.64083 14.93 2.70506 13.9315 10.9669 28.7917 -2.71148 1.96971 0.723389 -0.176561 12.9038 12.7331 5.80038 -14.0332 -24.1235 10.7129 -0.0108158 -21.1903 -0.0183596 -5.28466e-06 -21.5339 11.7961 -29.2118 ...

>I1010 00:04:44.358219  5598 voiceprint_main.cc:118] `Cosine similarity`: 0.975489

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

# Acknowledge

1. We borrowed a lot of code from [WeNet](https://github.com/wenet-e2e/wenet) for wav read and features extract.