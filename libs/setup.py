# Number of samples to include (randomly selected). None means all samples.
sample_size = 7331
# train samples
train_samples = 5533
# validation samples
val_samples = 2111
# batch of images used in each back propagation step
batch_size = 16
# Number of concurrent processes using to prepare data (0 means data will be loaded in
# the main process)
num_workers = 4
# Number of epochs to train the model
num_epochs = 80
# Base learning rate for the optimizer
base_learning_rate = 0.001
# Learning rate for the optimizer
learning_rate = 0.000115
# Wight decay for the optimizer
weight_decay = 0.05
# Layer decay for the optimizer
layer_decay = 0.75
# Smoothing factor for the optimizer
smoothing_factor = 0.1
# Early stoping patience
patience = 20
