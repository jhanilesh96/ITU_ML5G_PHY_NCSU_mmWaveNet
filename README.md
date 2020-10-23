# This repository contains the solution to ITU Artificial Intelligence/Machine Learning in 5G Challenge
## Problem Statement: Site-Specific Channel Estimation with Hybrid MIMO Architectures (https://research.ece.ncsu.edu/ai5gchallenge/)
### Team Name: mmWaveNet
### Organization: The Hong Kong University of Science and Technology (HKUST)
Members:
1) Neel Kanth Kundu (nkkundu@connect.ust.hk)
2) Nilesh Kumar Jha (nkjha@connect.ust.hk)
3) Amartansh Dubey (adubey@connect.ust.hk)

### Report
https://github.com/jhanilesh96/ITU_ML5G_PHY_NCSU_mmWaveNet/blob/main/AI_5G_Challenge_mmWave_Report_v3.pdf

### How to Use:
1. Download the Datasets: https://research.ece.ncsu.edu/ai5gchallenge/#datasets
1. Download gen_channel_ray_tracing_rev.mat from https://research.ece.ncsu.edu/ai5gchallenge/#datasets
1. For Training: Use line search to find $L^{*}$ for training SNR -15dB, -10dB and 0dB by running Compare_SOMP_sparsity.mat

### For Test: 
  1. First download the test data pilots from https://research.ece.ncsu.edu/ai5gchallenge/#datasets
  1. Run test.mat by changing the variables Dataset_pilots = 20/40/80  and Dataset_snr = 1/2/3 to get the 9 files containing the estimated channels

