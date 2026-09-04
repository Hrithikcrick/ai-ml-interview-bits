# 🤖 Multi-Head Attention, Causal Attention & KV Cache

> Interview Questions Q45-Q50

> Understanding how Transformer attention changes from training to real LLM generation.

---

# Q45. Why do Transformers use Multi-Head Attention?

## ⚡ 30-second answer

Multi-Head Attention allows a Transformer to learn multiple attention patterns in parallel.

Instead of forcing one attention mechanism to represent every relationship, different heads can specialize in different token relationships or representation subspaces.

Their outputs are concatenated and passed through a learned output projection.

---

## 🧠 Intuition

Imagine reading this sentence:

```text
The animal was tired because it had run far.
```

One attention head might focus on:

```text
pronoun ↔ entity
it      ↔ animal
```

Another might focus on:

```text
subject ↔ verb
```

Another could focus on:

```text
nearby positional relationships
```

So:

```text
One head
   ↓
one learned attention pattern

Multiple heads
   ↓
multiple learned attention patterns
```

---

## 🔬 How it works

For head i:

```text
Q_i = X * WQ_i
K_i = X * WK_i
V_i = X * WV_i
```

Each head independently computes:

```text
head_i = softmax(Q_i * K_i^T / sqrt(d_k)) * V_i
```

Then:

```text
head_1 ─┐
head_2 ─┤
head_3 ─┼→ Concatenate → Output Projection → Y
...    ─┤
head_h ─┘
```

Formally:

```text
H = Concat(head_1, head_2, ..., head_h)
Y = H * W_O
```

---

## 🧠 Why not just one large head?

A single head has one learned Query-Key matching pattern.

Multiple heads give the model several independently learned attention subspaces.

That creates greater representational flexibility.

---

## ⚠️ Interview trap

Do not claim that every attention head definitely learns a human-interpretable linguistic rule.

The key idea is that different heads **can learn different attention patterns or features**.

---

## 🎤 Follow-up

### What happens after all heads produce outputs?

Their outputs are concatenated and transformed using a learned output projection.

---

## 📌 Takeaway

> Multi-Head Attention gives a Transformer multiple learned views of token relationships.

---

# Q46. Does each attention head increase Transformer computation by the full attention cost?

## ⚡ 30-second answer

No.

In standard Multi-Head Attention, the total model dimension is divided across the heads.

If there are h heads, each head typically works with a dimension around:

```text
d_head = d_model / h
```

Therefore the total attention computation remains approximately comparable to performing attention over the full model dimension rather than becoming h times the full-dimensional cost.

---

## 🧠 Example

Suppose:

```text
d_model = 768
h = 12
```

Then:

```text
d_head = 768 / 12
       = 64
```

So the model does **not** use:

```text
12 heads × 768 dimensions each
```

Instead it uses approximately:

```text
12 heads × 64 dimensions
```

and:

```text
h * d_head = d_model
```

---

## 🔬 Complexity intuition

For one head:

```text
O(n^2 * d_head)
```

Across h heads:

```text
h * O(n^2 * d_head)
```

Since:

```text
h * d_head = d_model
```

we get approximately:

```text
O(n^2 * d_model)
```

---

## 🧠 Mental model

```text
Model representation
        ↓
Split into smaller attention subspaces
        ↓
Head 1  Head 2  ...  Head h
        ↓
Combine again
```

---

## ⚠️ Interview trap

Do not say:

> 12 attention heads means attention becomes exactly 12 times more expensive.

The head dimension is normally reduced so the total representation width remains approximately fixed.

---

## 📌 Takeaway

> More heads provide more attention subspaces without giving every head the full model dimension.

---

# Q47. What is causal masking and why is it required in decoder-only LLMs?

## ⚡ 30-second answer

Causal masking prevents a token from attending to future token positions.

This is required in autoregressive language modeling because the model must predict a token using only information that would already be available at that point.

Without causal masking, training would leak future target information.

---

## 🧠 Intuition

Suppose the sequence is:

```text
x1 x2 x3 x4
```

When processing an earlier position, the model must not be allowed to inspect later future positions.

Conceptually:

```text
x1 → x1
x2 → x1 x2
x3 → x1 x2 x3
x4 → x1 x2 x3 x4
```

---

## 🔬 Causal mask

The allowed-attention pattern is lower triangular:

```text
        Keys
        1  2  3  4

Q1      ✓  X  X  X
Q2      ✓  ✓  X  X
Q3      ✓  ✓  ✓  X
Q4      ✓  ✓  ✓  ✓
```

Future positions are blocked.

---

## 🔬 How are positions blocked?

Before softmax, blocked attention scores conceptually receive:

```text
-infinity
```

Example:

```text
[score_1, score_2, -infinity, -infinity]
```

After softmax:

```text
exp(-infinity) → 0
```

so blocked future tokens receive zero attention probability.

---

## 💡 Example

Suppose training text is:

```text
The cat sat on the mat
```

When learning to predict a later token, the model must not inspect that future target directly.

Otherwise:

```text
future answer
     ↓
visible during training
     ↓
information leakage
```

---

## ⚠️ Interview trap

Causal masking is not the same as padding masking.

Causal masking blocks **future positions**.

Padding masking blocks **padding tokens**.

---

## 📌 Takeaway

> Causal masking enforces the left-to-right information restriction required by autoregressive language modeling.

---

# Q48. Why can Transformer training process all tokens in parallel even though generation is autoregressive?

## ⚡ 30-second answer

During training, the entire target sequence is already known.

Therefore all token positions can be processed in one forward pass while a causal mask prevents each position from seeing future information.

During generation, future tokens do not exist yet, so tokens must be generated sequentially.

---

## 🧠 Training

Suppose the training sequence is:

```text
x1 x2 x3 x4
```

The whole sequence is available.

The Transformer can calculate representations for all positions simultaneously.

The causal mask enforces:

```text
x1 → x1
x2 → x1 x2
x3 → x1 x2 x3
x4 → x1 x2 x3 x4
```

But all these matrix operations can still run together on the GPU.

---

## 🚀 Generation

During inference:

```text
Prompt
  ↓
Generate x1
  ↓
Use x1 to generate x2
  ↓
Use x1,x2 to generate x3
  ↓
...
```

The next generated token is unknown until the current generation step completes.

---

## 📊 Key distinction

```text
TRAINING
Complete target sequence already known
             ↓
Parallel token computation
             +
Causal mask

INFERENCE
Future generated tokens unknown
             ↓
Sequential autoregressive generation
```

---

## 🎤 Interview term

This training setup is commonly described as:

```text
teacher-forced parallel training
```

---

## ⚠️ Interview trap

Parallel training does not mean tokens are allowed to look into the future.

They are computed in parallel, but causal masking still restricts the information available at every position.

---

## 📌 Takeaway

> Training parallelizes known positions; inference must sequentially create unknown future positions.

---

# Q49. What is KV Cache and why is it important for LLM inference?

## ⚡ 30-second answer

During autoregressive generation, Keys and Values calculated for previous tokens do not change.

KV Cache stores those previous K and V representations so they can be reused instead of recomputed at every generation step.

This significantly reduces redundant computation during LLM inference.

---

## 🧠 Without KV Cache

Suppose we already processed:

```text
x1 x2 x3
```

and now generate:

```text
x4
```

Without caching, the model would repeatedly recompute representations such as:

```text
K1 K2 K3
V1 V2 V3
```

even though they have not changed.

---

## 🚀 With KV Cache

The cache already stores:

```text
K cache = [K1, K2, K3]
V cache = [V1, V2, V3]
```

For the new token x4, calculate only its new:

```text
Q4
K4
V4
```

Then append:

```text
K cache = [K1, K2, K3, K4]
V cache = [V1, V2, V3, V4]
```

The new Query Q4 can attend to:

```text
K1 K2 K3 K4
```

and retrieve information from the corresponding Values.

---

## 🧠 Complete generation flow

```text
Previous tokens
      ↓
cached K and V
      ↓
New token
      ↓
compute new Q,K,V
      ↓
append new K,V
      ↓
new Q attends to cached Keys
      ↓
weighted Values
      ↓
next-token representation
```

---

## ⚠️ Interview trap

KV Cache does **not** remove attention.

The new token still attends to the available context.

KV Cache avoids recomputing old Key and Value representations.

---

## 🎤 Follow-up

### Why not cache Queries too?

For the current generation step, the important Query is the Query of the new token.

Previous Queries are not needed to calculate the new token attention in the same way previous Keys and Values are.

---

## 📌 Takeaway

> KV Cache reuses old Keys and Values to make autoregressive generation much more efficient.

---

# Q50. What is the difference between training-time attention and inference-time attention?

## ⚡ 30-second answer

During training, the complete sequence is available, so all positions can be processed in parallel under a causal mask.

During autoregressive inference, future tokens are unknown, so one new token is generated at a time.

KV Cache becomes especially useful during inference because previous Keys and Values can be reused across generation steps.

---

## 🧠 Training-time attention

Suppose training input is:

```text
[x1, x2, x3, x4, x5]
```

All positions already exist.

Therefore:

```text
complete sequence
       ↓
Q, K, V for all positions
       ↓
causal mask
       ↓
parallel attention computation
       ↓
predictions for multiple positions
```

---

## 🚀 Inference-time attention

Suppose we currently have:

```text
x1 x2 x3
```

The model generates:

```text
x4
```

Only after x4 exists can it generate:

```text
x5
```

So:

```text
x1 → x2 → x3 → x4 → x5 → ...
```

Generation has an inherent sequential dependency.

---

## 🔬 Probability view

Autoregressive generation follows the idea:

```text
P(x_t | x1, x2, ..., x_(t-1))
```

The model cannot know x_t until it performs the generation step.

---

## 📊 Interview comparison

| Property | Training | Inference |
|---|---|---|
| Entire sequence known | Yes | No |
| Token computation | Highly parallel | Sequential generation |
| Causal restriction | Yes | Yes |
| KV Cache central to process | Usually not the same need | Very important |

---

## ⚠️ Interview trap

Do not say Transformer inference has no parallelism at all.

The key source-supported distinction here is that **generation across successive tokens is autoregressively sequential**, while training can process known sequence positions simultaneously.

---

## 📌 Takeaway

> Training knows the sequence and parallelizes positions; inference creates the sequence one token at a time.

---

# ✅ Attention & KV Cache Section Complete

```text
Q45  Multi-Head Attention                         ✅
Q46  Cost of multiple attention heads             ✅
Q47  Causal masking                               ✅
Q48  Parallel training vs autoregressive decoding ✅
Q49  KV Cache                                     ✅
Q50  Training vs inference attention              ✅
```

## ⏭️ Next

### Q57-Q62 — Transformer MLP / FFN & Modern LLM Activations

> The interview material numbering moves from Q50 to Q57, so Q51-Q56 are intentionally not invented.
