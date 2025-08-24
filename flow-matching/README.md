# Flow matching

Flow matching implementation on Gaussian probability path with Optimal Transport scheduling

## Intuitive (non-rigorous) explanation of flow matching

The overall goal is to be able to sample $x$ from an unknown data distribution (let's say the distribution of images, if we want to create an image generator). 

Before defining the two steps required to achieve the aforementioned goal, let's define some variables:

 - $p_t$ probability distribution at time $t$. $p_t$ is also called a probability path because it interpolates between an initial (known) probability distribution $p_0$ and a final (unknown) probability distribution $p_1$

 - $x_t \sim p_t$ is a sample from the probability path at time $t$

 - $z \sim p_1$ is a sample from the unknown probability distribution (it's a sample from the training dataset)

 The two steps needed to achieve the goal of sampling from the unknown data distribution are the following:

 1. *Training:* During training we want a neural network to learn the parameters $\theta$ of a vector field $u_{t, \theta}$ that defines the evolution of our sample $x_t$ via the following ODE: 
 $$\dot{x}_t = u\_{t, \theta}(x_t)$$

 2. *Generation:* Once the neural network is trained and the vector field $u_{t, \theta}$ is learned, we sample $x_0 \sim p_0$ from a known predefined initial distribution $p_0$, and solve the ODE numerically to get $x_1 \sim p_1$.

Let's break down the two steps in detail, for the simple case of a Gaussian initial distribution. Hence:
$$x_0 \sim \mathcal{N}(0, I)$$

### Training

Define $X_t = (t, x_t)$. This will simplify the notation as $u_{t, \theta}(x_t) = u_{\theta}(X_t)$.

The training objective is to learn the vector field $u_{\theta}$. Let $u^{\text{target}}$ be the target vector field. The flow matching loss is simply:
$$\|u_{\theta}(X_t) - u^{\text{target}}(X_t)\|^2$$

Clearly, we do not know $u^{\text{target}}$, but there is a nice result proving that gradients of the flow matching loss are the same as the conditional flow matching loss:
$$\|u_{\theta}(X_t) - u^{\text{target}}(X_t | z)\|^2$$
and under some mild assumptions we do know the conditional target vector field $u^{\text{target}}(X_t | z)$.

You might be wondering, how do we know the conditional target vector field $u^{\text{target}}(X_t | z)$? I won't explain the mathematical details here, but I will construct an example which clarifies how we know the conditional target vector field.

Think about images. We clearly do not know the complex distribution of images, and we do not know the probability path $p_0 \longrightarrow p_t \longrightarrow p_1$, but we can know the conditional probability path:
$$p_0(\cdot | z) \longrightarrow p_t(\cdot | z) \longrightarrow p_1(\cdot | z)$$

Here, $p_0(\cdot | z) \sim \mathcal{N}(0, I)$ is simply the known initial distribution. $p_1(\cdot | z) \sim \mathcal{N}(z, 0)$ is the Dirac delta $\delta_z$, as the probability of sampling from the data distribution $p_1$ conditioned on the sample $z$ is one. Therefore we can define:
$$p_t \sim \mathcal{N}(\alpha_t z, \beta_t^2 I)$$
for each time $t$ and simply impose that $\alpha_t$ and $\beta_t$ satisfy the following constraints:
$$\alpha_0 = 0, \quad \beta_0 = 1, \quad \alpha_1 = 1, \quad \beta_1 = 0$$

Now that we know the conditional probability path $p_t(\cdot | z)$, we also know the conditional target vector field $u^{\text{target}}(X_t | z)$. It can be shown that the target conditional vector field is written as:

$$
u^{\text{target}}(X_t | z) = \left(\dot{\alpha}_t - \frac{\dot{\beta}_t}{\beta_t}\alpha_t\right)z + \frac{\dot{\beta}_t}{\beta_t}X_t 
$$

That's it! The overall training algorithm is the following. For each batch of data:

 - Sample $z$ from the dataset
 - Sample $t \sim \text{Unif}(0, 1)$
 - Sample $\epsilon \sim \mathcal{N}(0, I)$
 - Compute $X_t = (t, \alpha_t z + \beta_t \epsilon)$
 - Compute loss: 
   $$\left\|u_{\theta}(X_t) - \left(\dot{\alpha}_t - \frac{\dot{\beta}_t}{\beta_t}\alpha_t\right)z + \frac{\dot{\beta}_t}{\beta_t}X_t\right\|^2$$

### Generation

Generation is fairly simple, it's only about solving the aforementioned ODE numerically. The following algorithm shows how it can be done with the simple Euler method. Starts at $t = 0$. Draw a sample $x_0 \sim p_0$, then for each time stap $\delta_t = \frac{1}{n}$:

 - Concatenate $X_t = (t, x_t)$
 - Compute $x\_{t + \delta_t} = x\_t + \delta_t u\_{\theta}(X_t)$
 - Update $t \leftarrow t + \delta_t$

## Questions
 - Could I sample more times for a given sampled data $z$ during training?