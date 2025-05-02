v2.0 is MSE NN, v3.0 is MAE NN. There are networks for pos and neg polarity.

Under model_size_investigation, models for 100 sizes between 0.0001 and 1.0. networks each.

Under model_size_investigation_bootstrap, models versions v1, v2, and v1.1 trained on bootstrapped samples of the data. v1 and v2 have different sampling seeds, v1.1 has same sampling seed as v1 but different initialization.

Under model_size_investigation_shuffled_trainset, move naming of experiments to the following:
- data subsample (d1, d2, etc). NN initialization (init1, init2, etc). bootstrap is b1 if bootstrapping, b0 if not. 