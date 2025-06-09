# Autoencoders

This folder contains simple generic Autoencoder and Variational Autoencoder implementations for understanding their functionalities.

### Autoencoder

An **autoencoder** is a neural network that learns to compress data into a lower-dimensional latent space and then reconstruct it. An autoencoder can be considered a generative model as we can generate data decoding latent variables. An autoencoder consists of two parts:

- **Encoder:** Maps input data $x$ to a latent representation $z$.
- **Decoder:** Reconstructs $x$ from $z$.

The goal is to minimize the reconstruction loss:
$$
\mathcal{L}_{\text{AE}} = \| x - \hat{x} \|^2
$$
where $\hat{x}$ is the reconstruction of $x$ from $z$.

### Variational Autoencoder

A **variational autoencoder (VAE)** is a probabilistic extension of the autoencoder. It models the data distribution $p(x)$ using latent variables $z$:

- **Encoder:** Approximates the posterior $p(z|x)$ with a learned distribution $q(z|x)$.
- **Decoder:** Models the likelihood $p(x|z)$.

The VAE is trained by maximizing the evidence lower bound (ELBO): $\mathcal{L}_{\text{VAE}} = \mathbb{E}_{q(z|x)} [\log p(x|z)] - D_{\mathrm{KL}}(q(z|x) \| p(z))$ where $D_{\mathrm{KL}}$ is the Kullback-Leibler divergence between the approximate posterior and the prior $p(z)$.

#### The Role of the Reparametrization Trick

In VAEs, the encoder outputs parameters of a probability distribution (typically mean $\mu$ and standard deviation $\sigma$) rather than a single latent vector. To enable backpropagation through the sampling process, the **reparametrization trick** is used. 
Specifically, with $\mu$ $\sigma$ parametrized by the network hyperparameters, the gradient of the ELBO is intractable. Hence, instead of sampling $z \sim \mathcal{N}(\mu, \sigma^2)$ directly, we sample $\epsilon \sim \mathcal{N}(0, 1)$ and compute:

$$
z = \mu + \sigma \cdot \epsilon
$$

This allows gradients to flow through $\mu$ and $\sigma$ during training, making the VAE end-to-end differentiable.

## Contents

- `data/` &mdash; Datasets
- `models/` &mdash; Autoencoder models parameters.
- `runs/` &mdash; Tensorboard summaries.
- `src/architecturs/` &mdash; Autoencoders architectures.
- `src/data/` &mdash; Dataset loaders.
- `src/helpers/` &mdash; Helper functions.

## How to train on the MNIST and generate data

### Interesting questions:

 - How does the loss changes with respect to the dimension of the latent space? Is too much compression counter-productive?

`VAE_20250608210335`'s encoder has the following layers dimensions [28*28, 512, 256, 128, 64, 32, 12, 6, 3, 2]. In `VAE_20250608211222`'s encoder the last two layers of dimensions [3, 2] were removed leaving a latent space of dimension `6`. In `VAE_20250608212013`'s encoder the last three layers of dimensions [6, 3, 2] were removing leaving a latent space of dimension `12`.

![tensorboard](resources/latent_dimension_comparizon.png)

 - Is higher `beta` yielding lower reconstruction loss?

Higher beta was observed to yield higher reconstruction loss, and the generated samples appear more blurry.

