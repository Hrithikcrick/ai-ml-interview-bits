# 🧠 Sections 1-2 — Neural Network Fundamentals & Activation Functions

> Source Questions Q1-Q15

---

# Q1. What is a feature in Machine Learning?

## ⚡ 30-second answer

A feature is a measurable property or representation of an input that provides useful information for a prediction task.

Traditional machine learning often relies on manually designed features, while deep neural networks can learn useful representations directly from raw inputs.

## 🧠 Intuition

For image classification, the raw input can simply be pixels.

```text
Pixels
  ↓
Edges and textures
  ↓
Shapes
  ↓
Object parts
  ↓
Task-relevant representation
```

This is hierarchical feature learning.

## ⚠️ Interview trap

A feature is not always just a dataframe column. Internal learned neural representations are also features.

## 📌 Takeaway

> Deep learning can learn the representation instead of requiring every useful feature to be manually designed.

---

# Q2. What are weights in a neural network?

## ⚡ 30-second answer

Weights are learnable parameters that determine how strongly different inputs influence a neuron.

A neuron computes something like:

```text
z = w1*x1 + w2*x2 + ... + wn*xn + b
```

During training, backpropagation calculates how changing each weight affects the loss, and gradient-based optimization updates the weights.

## 🧠 Intuition

A larger weight magnitude allows the corresponding input to have a stronger effect on the pre-activation.

## 📌 Takeaway

> Weights are the learnable parameters through which a neural network learns relationships between inputs and outputs.

---

# Q3. Why do we need bias in a neural network?

## ⚡ 30-second answer

Bias provides an additional learnable offset.

Without bias:

```text
z = w*x
```

a linear decision boundary is constrained to pass through the origin.

With bias:

```text
z = w*x + b
```

the same slope or orientation can be shifted.

## 🧠 Intuition

```text
weights → orientation / influence
bias    → shift
```

## 📌 Takeaway

> Bias increases flexibility by allowing the activation or decision boundary to move.

---

# Q4. What happens if we stack many linear layers without activation functions?

## ⚡ 30-second answer

The network is still equivalent to a single affine transformation.

Suppose:

```text
h1 = W1*x + b1
h2 = W2*h1 + b2
```

Substituting gives:

```text
h2 = (W2*W1)*x + (W2*b1 + b2)
```

This is still affine.

## ⚠️ Interview trap

Depth alone does not create nonlinear expressive power.

## 📌 Takeaway

> Many linear layers without nonlinear activations collapse into one affine transformation.

---

# Q5. Why are activation functions necessary?

## ⚡ 30-second answer

Activation functions introduce nonlinearity into a neural network.

Without them, many layers still behave like one affine transformation.

With an activation:

```text
h = f(W*x + b)
```

the network can represent nonlinear relationships.

## 💡 Example

XOR cannot be represented correctly using a single linear decision boundary.

## 📌 Takeaway

> Activation functions are what allow depth to provide additional representational power.

---

# Q6. What is the dying ReLU problem?

## ⚡ 30-second answer

ReLU is:

```text
ReLU(x) = max(0,x)
```

For negative input its derivative is zero.

If a neuron consistently remains in the negative region, the gradient through that activation becomes zero and its parameters may stop receiving useful updates.

## 🧠 Flow

```text
negative pre-activation
        ↓
zero ReLU derivative
        ↓
zero gradient through activation
        ↓
neuron may become inactive
```

## 📌 Takeaway

> A dying ReLU is effectively stuck in a zero-gradient region.

---

# Q7. Why does Leaky ReLU help with dying ReLU?

## ⚡ 30-second answer

Leaky ReLU gives negative inputs a small non-zero slope.

```text
x > 0  → f(x) = x
x <= 0 → f(x) = alpha*x
```

Therefore the derivative in the negative region is alpha instead of zero.

Some gradient can continue to flow, giving the neuron an opportunity to recover.

## 📌 Takeaway

> Leaky ReLU reduces permanently inactive neurons by keeping a small gradient alive in the negative region.

---

# Q8. Why does sigmoid suffer from vanishing gradients?

## ⚡ 30-second answer

Sigmoid has derivative:

```text
sigma prime(x) = sigma(x) * (1 - sigma(x))
```

Its maximum derivative is 0.25, and in saturation regions the derivative approaches zero.

Backpropagation multiplies derivatives across many layers, so repeated multiplication by small factors can make gradients reaching early layers almost zero.

## 🧠 Intuition

```text
0.2 × 0.2 × 0.2 × 0.2 × ...
        ↓
very small number
```

## 📌 Takeaway

> Sigmoid can make deep gradient flow weak because its derivatives are small, especially when saturated.

---

# Q9. What are activation functions and why do we need them?

## ⚡ 30-second answer

Activation functions introduce nonlinearity so deep networks can learn complex functions rather than collapsing into a single affine transformation.

Examples in the source include:

```text
ReLU
Sigmoid
GELU
Tanh
SiLU / Swish
```

## 📌 Takeaway

> Activation functions give deep networks nonlinear representation power.

---

# Q10. Why can we not simply remove activation functions and use many linear layers?

## ⚡ 30-second answer

Because the composition of linear transformations is still linear.

```text
h1 = W1*x
h2 = W2*h1

h2 = W2*W1*x
```

The two layers can therefore be replaced by one matrix.

## 📌 Takeaway

> More linear layers do not create nonlinear capability.

---

# Q11. What is ReLU and why is it widely used?

## ⚡ 30-second answer

ReLU means Rectified Linear Unit.

```text
ReLU(x) = max(0,x)
```

For positive values, its derivative is 1. This avoids the continuous gradient shrinkage caused by sigmoid derivatives in the positive region.

ReLU is also computationally cheap because it mainly requires comparing the input with zero.

## ⚠️ Interview trap

ReLU does not completely solve gradient problems because its negative-region derivative is zero.

## 📌 Takeaway

> ReLU is simple, cheap and provides strong gradient flow for positive activations.

---

# Q12. What is the dying ReLU problem?

## ⚡ 30-second answer

If a ReLU neuron repeatedly receives negative pre-activations, both its output and local derivative are zero.

Consequently, the gradient through the activation becomes zero and the neuron may stop learning.

## 💡 Example

```text
z = -5
ReLU(-5) = 0
ReLU prime(-5) = 0
```

## 📌 Takeaway

> Repeated negative pre-activations can make a ReLU neuron effectively inactive.

---

# Q13. Why does Leaky ReLU help with the dying ReLU problem?

## ⚡ 30-second answer

Leaky ReLU replaces the zero negative-region slope with a small positive slope.

For example, with alpha = 0.01:

```text
x = -5
f(x) = -0.05
f prime(x) = 0.01
```

The gradient is small, but it is not zero.

## 📌 Takeaway

> Leaky ReLU maintains a small learning signal for negative activations.

---

# Q14. Why does sigmoid suffer from the vanishing-gradient problem?

## ⚡ 30-second answer

The sigmoid derivative is at most 0.25.

Repeated chain-rule multiplication can therefore shrink gradients quickly.

The source gives the example:

```text
0.25 × 0.25 × 0.25 × 0.25
= 0.00390625
```

After many layers, early-layer gradients can become almost zero and learning becomes very slow.

## 📌 Takeaway

> Small sigmoid derivatives compound across deep networks.

---

# Q15. Why is sigmoid usually not used as the activation inside modern deep networks?

## ⚡ 30-second answer

Sigmoid hidden units can saturate near 0 or 1, where their derivatives become very small.

That makes sigmoid a poor default hidden-layer activation for deep networks because gradient flow can become weak.

However, sigmoid remains useful when an output must represent a probability in binary classification and it is also used inside gating mechanisms.

## ⚠️ Interview trap

Do not say sigmoid is obsolete. It is still useful in appropriate output layers and gates.

## 📌 Takeaway

> Sigmoid is usually avoided as the default deep hidden activation, but remains useful for probabilities and gating.

---

# ✅ Q1-Q15 Complete

Next source section:

## Q16-Q23 — Weight Initialization and Gradient Flow
