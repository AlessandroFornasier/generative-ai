# Image Generator (Flow Matching + Classifier-Free Guidance)

This module trains a flow matching model to generate MNIST digit images. Specifically:

- Uses a flow matching model to learn a time-dependent vector field that transports noise samples to MNIST images.

- Trains with classifier-free guidance (CFG) to enable conditional & unconditional generation using the same model.

For the conceptual and mathematical background of flow matching (probability paths, training objective, ODE-based generation), see the detailed explanation in [flow-matching](../flow-matching/README.md).

## Classifier-Free Guidance (CFG)

Classifier-free guidance improves conditional sample fidelity without needing an external classifier. Specifically:

### Training

During training, with probability $p$, the label for a batch is replaced with a special "null" condition (implemented as -1). Therefore the model learns a conditional vector field: $u^{\text{conditional}}(X_t | z, y)$ as well as an nconditional vector field: $u^{\text{unconditional}}(X_t | z)$

The overall training algorithm is the following. For each batch of data:

- Sample the data $z$ and the label $y$ from the dataset
- Sample $t \sim \text{Unif}(0, 1)$
- Sample $\epsilon \sim \mathcal{N}(0, I)$
- With probability $p$, replace the label $y$ with a null label
- Compute $X_t = (t, \alpha_t z + \beta_t \epsilon)$
- Compute loss: 
   $$\left\|u_{\theta}(X_t) - \left(\dot{\alpha}_t - \frac{\dot{\beta}_t}{\beta_t}\alpha_t\right)z + \frac{\dot{\beta}_t}{\beta_t}X_t\right\|^2$$

### Generation

At generation time the guided vector field  blend them: 

$$
u^{\text{guided}}(X_t) =  (1 - w) u^{\text{unconditional}}(X_t) + w u^{\text{conditional}}(X_t)
$$

where $w > 1$ is the guidance scale. This has the effect to push samples toward better class-consistent structure while retaining diversity. In particular the difference $u^{\text{conditional}}(X_t) - u^{\text{unconditional}}(X_t)$ approximates a directional signal toward the target class.

The overall generation algorithm is the following.

 - Concatenate $X_t = (t, x_t)$
 - Compute $x\_{t + \delta_t} = x\_t + \delta_t u^{\text{guided}}\_{\theta}(X_t)$
 - Update $t \leftarrow t + \delta_t$

 ## Generated path

![Generated Path](image-generator/generated/model_20250913131734_path.png)

This shows the denoising trajectory from random noise to a generated MNIST digit over the flow matching process.