# TensorTonic Solutions

Welcome to my TensorTonic solutions repository!

Here you'll find my solutions to various machine learning and deep learning problems from [TensorTonic](https://tensortonic.com).

## What is TensorTonic?

TensorTonic is a platform where you can implement core algorithms of Machine Learning from scratch.

This repository contains my personal solutions to these problems, automatically synchronized from the platform.

<!-- tensortonic:start -->
# Naiwen Xie's TensorTonic Solutions

Verified machine learning implementations completed on [TensorTonic](https://www.tensortonic.com).

<p align="center">
  <img src="https://www.tensortonic.com/api/badge/naiwen.svg" alt="TensorTonic Verified Solutions" width="100%" />
</p>

| Problem | Description | Link |
|---|---|---|
| Implement Adam Optimizer Step | Implement one vectorized Adam optimizer step in NumPy with first and second moments, bias correction, and elementwise parameter updates. | https://www.tensortonic.com/problems/adam-optimizer |
| Angle Between 3D Vectors | Compute the angle between two 3D vectors in NumPy with clamped cosine values and safe handling of zero norms. | https://www.tensortonic.com/problems/angle-between-3d |
| Bag-of-Words Vector | Build a NumPy bag-of-words count vector from an ordered vocabulary while ignoring out-of-vocabulary tokens. | https://www.tensortonic.com/problems/bag-of-words |
| Batch Normalization (Forward) | Implement the batch-normalization forward pass in NumPy using feature-wise statistics, scale, shift, and numerical stability. | https://www.tensortonic.com/problems/batch-normalization |
| Implement Causal Masking for Attention | Create a causal attention mask that blocks each token from attending to future positions in a sequence. | https://www.tensortonic.com/problems/causal-masking |
| Color to Grayscale | Convert an RGB image to grayscale using weighted color channels while preserving its spatial dimensions. | https://www.tensortonic.com/problems/color-to-grayscale |
| Implement Cross-Entropy Loss | Compute multiclass cross-entropy loss from class probabilities and integer labels with stable logarithms. | https://www.tensortonic.com/problems/cross-entropy-loss |
| Implement Dot Product | Implement the dot product of equal-length numeric vectors by summing element-wise products without library shortcuts. | https://www.tensortonic.com/problems/dot-product |
| Implement Global Average Pooling | Apply global average pooling to spatial feature maps by averaging each channel across its height and width. | https://www.tensortonic.com/problems/global-avg-pooling |
| He Initialization | Scale raw weights into the He uniform range using a bound derived from the layer fan-in. | https://www.tensortonic.com/problems/he-initialization |
| Jaccard Similarity | Compute Jaccard similarity between two collections as intersection size divided by union size. | https://www.tensortonic.com/problems/jaccard-similarity |
| Linear Layer Forward | Implement a dense linear layer forward pass by multiplying inputs by weights and adding a bias vector. | https://www.tensortonic.com/problems/linear-layer-forward |
| Max Pooling Forward | Apply 2D max pooling to a numeric matrix using a configurable square window and stride. | https://www.tensortonic.com/problems/maxpool-forward |
| Mean Squared Error (MSE) | Compute mean squared error between predictions and targets by averaging their squared element-wise differences. | https://www.tensortonic.com/problems/mean-squared-error |
| One-Hot Encoding (Multi-class) | Convert multiclass integer labels into a NumPy one-hot matrix with one active column per sample. | https://www.tensortonic.com/problems/one-hot-encoding |
| Perplexity Computation | Compute language-model perplexity from token probability distributions and the observed token indices. | https://www.tensortonic.com/problems/perplexity-computation |
| Implement Positional Encoding (sin/cos) | Generate sinusoidal Transformer positional encodings across sequence positions and embedding dimensions. | https://www.tensortonic.com/problems/positional-encoding |
| Ridge Regression | Fit ridge regression with L2 regularization using the closed-form solution required by the problem. | https://www.tensortonic.com/problems/ridge-regression |
| RNN Step Forward (Tanh Cell) | Implement one vanilla RNN timestep with affine input and recurrent transforms followed by tanh activation. | https://www.tensortonic.com/problems/rnn-step-forward |
| Implement Sigmoid in NumPy | Implement a vectorized sigmoid activation in NumPy for scalars, lists, vectors, and matrices, including large positive and negative inputs. | https://www.tensortonic.com/problems/sigmoid-numpy |
| Implement a Simple CNN Layer (NumPy) | Implement a NumPy CNN layer forward pass with batched valid convolution across channels and bias addition. | https://www.tensortonic.com/problems/simple-cnn-layer |
| Implement Softmax Function | Implement numerically stable softmax by shifting logits before exponentiation and normalizing probabilities. | https://www.tensortonic.com/problems/softmax-function |
| Implement Swish Activation | Apply the Swish activation element-wise by multiplying each input by its sigmoid value. | https://www.tensortonic.com/problems/swish-activation |
| Implement Tanh Activation | Implement the hyperbolic tangent activation element-wise with outputs bounded between minus one and one. | https://www.tensortonic.com/problems/tanh-activation |
| Xavier Initialization | Scale raw weights into the Xavier uniform range using a bound derived from fan-in and fan-out. | https://www.tensortonic.com/problems/xavier-initialization |
| AlexNet Convolution Layer | Implement an AlexNet convolutional layer with learned filters, bias, stride, padding, and multi-channel outputs. | https://www.tensortonic.com/research/alexnet/alexnet-conv-layers |
| Dropout Regularization | Implement inverted dropout for AlexNet with seeded masks and training-versus-inference behavior. | https://www.tensortonic.com/research/alexnet/alexnet-dropout |
| ReLU Activation Function | Implement AlexNet's elementwise ReLU activation, preserving positive values while setting negative values to zero. | https://www.tensortonic.com/research/alexnet/alexnet-relu |
| Forward Diffusion Process | Implement the DDPM forward diffusion process by mixing clean samples with Gaussian noise at a selected timestep. | https://www.tensortonic.com/research/ddpm/ddpm-forward |
| DDPM Training Loss | Compute the DDPM training objective as mean squared error between sampled noise and the model's noise prediction. | https://www.tensortonic.com/research/ddpm/ddpm-loss |
| DDPM Sampling | Implement iterative DDPM sampling from Gaussian noise by applying the learned reverse process over all timesteps. | https://www.tensortonic.com/research/ddpm/ddpm-sampling |
| Noise Schedule | Generate a DDPM noise schedule with beta values and cumulative alpha products used across diffusion timesteps. | https://www.tensortonic.com/research/ddpm/ddpm-schedule |
| GAN Discriminator | Implement a GAN discriminator that maps input samples through dense layers to real-versus-fake probabilities. | https://www.tensortonic.com/research/gan/gan-discriminator |
| Complete GAN System | Assemble a complete GAN system that generates samples, scores them with the discriminator, and returns both outputs. | https://www.tensortonic.com/research/gan/gan-full-network |
| GAN Generator | Implement a GAN generator that transforms latent noise through learned dense layers into generated samples. | https://www.tensortonic.com/research/gan/gan-generator |
| GAN Loss Functions | Compute numerically stable binary cross-entropy losses for the GAN generator and discriminator objectives. | https://www.tensortonic.com/research/gan/gan-loss |
| Mode Collapse Detection | Detect GAN mode collapse by measuring diversity across generated samples and flagging low-variance outputs. | https://www.tensortonic.com/research/gan/gan-mode-collapse |
| Complete LSTM Cell | Build a complete LSTM cell with forget, input, candidate, cell-state, output, and hidden-state calculations. | https://www.tensortonic.com/research/lstm/lstm-cell |
| Cell State Update | Implement the LSTM cell-state update by combining retained memory with input-gated candidate information. | https://www.tensortonic.com/research/lstm/lstm-cell-state |
| Forget Gate | Implement an LSTM forget gate by combining the previous hidden state and current input with a sigmoid projection. | https://www.tensortonic.com/research/lstm/lstm-forget-gate |
| Complete LSTM Network | Assemble an LSTM sequence forward pass that carries hidden and cell states across every time step. | https://www.tensortonic.com/research/lstm/lstm-full-network |
| Input Gate | Implement the LSTM input gate and candidate activation that control new information written to the cell state. | https://www.tensortonic.com/research/lstm/lstm-input-gate |
| Output Gate | Implement the LSTM output gate and expose the current hidden state from the updated cell memory. | https://www.tensortonic.com/research/lstm/lstm-output-gate |
| BatchNorm in ResNet | Implement ResNet batch normalization with channel statistics, learned scale and bias, and training or inference behavior. | https://www.tensortonic.com/research/resnet/resnet-batch-norm |
| Bottleneck Block | Build a ResNet bottleneck block using 1x1 channel reduction, 3x3 convolution, and 1x1 channel expansion. | https://www.tensortonic.com/research/resnet/resnet-bottleneck |
| Convolutional Block | Implement a ResNet convolutional block with a projected shortcut that matches changed spatial and channel dimensions. | https://www.tensortonic.com/research/resnet/resnet-conv-block |
| Full ResNet Assembly | Assemble a ResNet forward pass from the stem, residual stages, global average pooling, and the classification head. | https://www.tensortonic.com/research/resnet/resnet-full-network |
| Identity Block | Implement a ResNet identity block with a three-layer bottleneck branch, batch normalization, ReLU, and an unchanged skip path. | https://www.tensortonic.com/research/resnet/resnet-identity-block |
| Backpropagation Through Time | Implement one backpropagation-through-time step using the tanh derivative and hidden-to-hidden weight gradients. | https://www.tensortonic.com/research/rnn/rnn-bptt |
| Forward Through Sequence | Implement a vanilla RNN forward pass that updates and returns hidden states across every sequence time step. | https://www.tensortonic.com/research/rnn/rnn-forward-sequence |
| Complete Vanilla RNN | Assemble a vanilla RNN that processes sequences into recurrent hidden states and per-time-step output logits. | https://www.tensortonic.com/research/rnn/rnn-full-network |
| Hidden State | Initialize a vanilla RNN hidden state as a floating-point zero matrix for the requested batch and hidden dimensions. | https://www.tensortonic.com/research/rnn/rnn-hidden-state |
| Vanishing Gradients | Simulate vanishing or exploding RNN gradients by repeatedly applying the hidden matrix's spectral norm. | https://www.tensortonic.com/research/rnn/rnn-vanishing-gradients |
| Scaled Dot-Product Attention | Implement scaled dot-product attention in PyTorch using query-key scores, softmax weights, and value aggregation. | https://www.tensortonic.com/research/transformer/transformers-attention |
| Embedding Layer | Create PyTorch token embeddings and scale each lookup by the square root of the Transformer model dimension. | https://www.tensortonic.com/research/transformer/transformers-embedding |
| Encoder Block | Assemble a Transformer encoder block with multi-head attention, residual paths, layer normalization, and a feed-forward network. | https://www.tensortonic.com/research/transformer/transformers-encoder-block |
| Feed-Forward Network | Implement the Transformer's position-wise feed-forward network with two linear projections and a ReLU activation. | https://www.tensortonic.com/research/transformer/transformers-feed-forward |
| Layer Normalization | Implement Transformer layer normalization in NumPy using per-token mean, variance, scale, and bias. | https://www.tensortonic.com/research/transformer/transformers-layer-normalization |
| Multi-Head Attention | Build NumPy multi-head attention with learned projections, per-head scaled attention, concatenation, and output projection. | https://www.tensortonic.com/research/transformer/transformers-multi-head-attention |
| Positional Encoding | Implement sinusoidal Transformer positional encodings in NumPy with alternating sine and cosine dimensions. | https://www.tensortonic.com/research/transformer/transformers-positional-encoding |
| Tokenization | Build a word-level Transformer tokenizer with fixed special-token IDs, sorted vocabulary entries, encoding, and decoding. | https://www.tensortonic.com/research/transformer/transformers-tokenization |
| U-Net Bottleneck | Implement the U-Net bottleneck shape transformation for two unpadded convolutions at the deepest encoder stage. | https://www.tensortonic.com/research/unet/unet-bottleneck |
| U-Net Decoder Block | Implement U-Net decoder shape transformations for up-convolution, cropped skip concatenation, and two valid convolutions. | https://www.tensortonic.com/research/unet/unet-decoder-block |
| Complete U-Net Network | Assemble U-Net encoder, bottleneck, decoder, skip-connection, and output stages while tracking valid-convolution shapes. | https://www.tensortonic.com/research/unet/unet-full-network |
| U-Net Output Layer | Implement the U-Net output layer as a 1x1 channel projection that maps decoder features to per-pixel class logits. | https://www.tensortonic.com/research/unet/unet-output-layer |
| U-Net Skip Connections | Implement U-Net skip connections by center-cropping encoder features and concatenating them with decoder features. | https://www.tensortonic.com/research/unet/unet-skip-connection |
| VAE Decoder | Build a variational autoencoder decoder that maps sampled latent vectors back to reconstructed input probabilities. | https://www.tensortonic.com/research/vae/vae-decoder |
| ELBO Loss Function | Implement the VAE evidence lower bound from reconstruction loss and KL divergence with configurable weighting. | https://www.tensortonic.com/research/vae/vae-elbo-loss |
| VAE Encoder | Implement a variational autoencoder encoder that maps inputs to separate latent mean and log-variance vectors. | https://www.tensortonic.com/research/vae/vae-encoder |
| Complete VAE Network | Assemble a complete variational autoencoder forward pass with encoding, reparameterized sampling, and decoding. | https://www.tensortonic.com/research/vae/vae-full-network |
| KL Divergence Loss | Compute the VAE KL-divergence term between a diagonal Gaussian posterior and the standard normal prior. | https://www.tensortonic.com/research/vae/vae-kl-divergence |
| Reparameterization Trick | Implement the VAE reparameterization trick by sampling latent vectors from the predicted mean and log variance. | https://www.tensortonic.com/research/vae/vae-reparameterization |
| Estimate a Scalar Derivative | Estimate a scalar polynomial derivative with a forward finite difference using coefficients ordered by ascending power. | https://www.tensortonic.com/study-plans/autograd-from-scratch/autograd-l01-finite-difference-derivative |
| Measure Scalar Expression Partials | Estimate the three partial derivatives of a scalar expression by perturbing one input at a time. | https://www.tensortonic.com/study-plans/autograd-from-scratch/autograd-l01-scalar-expression-partials |

View my verified ML profile: [TensorTonic profile](https://www.tensortonic.com/profile/naiwen)
<!-- tensortonic:end -->
