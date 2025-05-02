Model configuration naming conventions:
Dataset random sampling seed: d1 = 42, d2 = 87
Bootstrapped random sampling (i.e. with replacement or not): b0 = False, b1 = True
NN initialization (i.e. model trained separately but on same data): init1, init2
HMC initialization (i.e. HMC sampling done twice on same data with same model): hmc1, hmc2

Train fractions of interest:
0.0001 0.001 0.01 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0

I trained the below 3 model configurations on the below train fractions of interest:
d1_b1_init1
d1_b1_init2
d2_b1_init1

I trained the below 2 model configurations on the full (non-bootstrapped) train set only:
d1_b0_init1
d1_b0_init2
