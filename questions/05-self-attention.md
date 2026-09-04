# 🔍 Self-Attention & Transformer Architecture

> Interview Questions Q38-Q44

> From contextual token interaction to the computational cost of attention.

---

# Q38. What problem does self-attention solve?

## ⚡ 30-second answer

Self-attention allows every token to dynamically incorporate information from other relevant tokens in the sequence.

This lets a Transformer model context-dependent relationships and long-range dependencies instead of processing every token independently.

---

## 🧠 Intuition

Consider the sentence:

> The animal did not cross the road because it was tired.

To understand the word:

```text
it
```

the model needs information from another token:

```text
animal
```

A token-independent mechanism would struggle to model this relationship directly.

Self-attention instead allows:

```text
Current token
      ↓
Look at other tokens
      ↓
Measure relevance
      ↓
Select useful context
      ↓
Update token representation
```

---

## 💡 Mental model

For the token:

```text
it
```

the model might learn that:

```text
animal   → highly relevant
road     → less relevant
because  → less relevant
```

The attention weights are learned dynamically from the token representations.

---

## ⚠️ Interview trap

Do not say self-attention gives every token equal importance.

The whole purpose is to assign different relevance to different token interactions.

---

## 🎤 Follow-up

### Why is this useful for long-range dependencies?

A token can directly interact with another token even when they are far apart in the sequence.

---

## 📌 Takeaway

> Self-attention lets each token selectively use information from the rest of the sequence.

---

# Q39. What are Query, Key and Value in self-attention?

## ⚡ 30-second answer

Query, Key and Value are three learned representations created from the input.

```text
Q = X * WQ
K = X * WK
V = X * WV
```

The Query represents what the current token is looking for.

The Key represents what each token offers for matching.

The Value contains the actual information that can be aggregated.

---

## 🧠 Best mental model

### Query

```text
What information am I looking for?
```

### Key

```text
What type of information do I contain?
```

### Value

```text
What actual information should I pass forward?
```

---

## 🔎 Search analogy

Imagine a search system:

```text
Query → what you search for
Key   → what each item advertises for matching
Value → the actual content returned
```

Attention first compares:

```text
Query ↔ Keys
```

to determine relevance.

Then it combines:

```text
Values
```

according to those relevance weights.

---

## 🔬 Learned projections

The Transformer does not normally use the original X directly as all three representations.

Instead:

```text
X
├── WQ → Q
├── WK → K
└── WV → V
```

Each matrix is learned during training.

---

## ⚠️ Interview trap

Do not simply say:

```text
Q = K = V = X
```

They originate from the same input in self-attention, but standard attention applies different learned projection matrices.

---

## 📌 Takeaway

> Q asks, K matches, V carries the information.

---

# Q40. How is the attention score calculated?

## ⚡ 30-second answer

Scaled dot-product attention calculates Query-Key similarity, scales the scores, converts them into probabilities using softmax, and then uses those probabilities to form a weighted combination of Value vectors.

The complete operation is:

```text
Attention(Q,K,V)
=
softmax(Q * K^T / sqrt(d_k)) * V
```

---

## 🔬 Step 1 — Query-Key similarity

Calculate:

```text
Q * K^T
```

The dot product measures how strongly a Query matches each Key.

Example:

```text
Q · K1 = 2
Q · K2 = 8
Q · K3 = 1
```

The second Key has the highest matching score.

---

## 🔬 Step 2 — Scale

Divide by:

```text
sqrt(d_k)
```

where:

```text
d_k = dimension of the Key / Query vectors
```

So:

```text
S = Q * K^T / sqrt(d_k)
```

---

## 🔬 Step 3 — Softmax

Apply:

```text
A = softmax(S)
```

This converts raw matching scores into normalized attention weights.

The weights approximately sum to:

```text
1
```

---

## 🔬 Step 4 — Combine Values

Finally:

```text
Y = A * V
```

The output is a weighted combination of Value vectors.

---

## 🧠 Complete flow

```text
Q * K^T
   ↓
Query-Key relevance
   ↓
/ sqrt(d_k)
   ↓
controlled score scale
   ↓
softmax
   ↓
attention weights
   ↓
* V
   ↓
weighted information aggregation
```

---

## 🎤 Interview-ready answer

> QK^T calculates Query-Key similarity, division by sqrt(d_k) controls the magnitude of the logits, softmax converts them into attention weights, and multiplying by V produces the context-aware output.

---

## 📌 Takeaway

> Attention = calculate relevance → normalize relevance → aggregate information.

---

# Q41. Why do we divide attention scores by sqrt(d_k)?

## ⚡ 30-second answer

As the Query-Key dimension increases, the variance and typical magnitude of their dot product can increase.

Large attention logits can push softmax into an extremely sharp or saturated region, producing very small gradients.

Dividing by sqrt(d_k) keeps the logits at a more manageable scale.

---

## 🔬 Dot-product intuition

For vectors of dimension d_k:

```text
Q · K
=
Q1*K1 + Q2*K2 + ... + Qd*K_d
```

As the number of summed terms increases, the variance of the result increases.

Under the standard simplifying assumption:

```text
mean ≈ 0
variance ≈ 1
```

we get approximately:

```text
Var(Q · K) ∝ d_k
```

Therefore the standard deviation scales approximately as:

```text
sqrt(d_k)
```

---

## 🧠 Why large logits are bad

Compare:

```text
[1, 2, 3]
```

with:

```text
[10, 20, 30]
```

The second produces a much sharper softmax distribution.

An extreme example:

```text
[0, 100, 0]
```

becomes approximately:

```text
[0, 1, 0]
```

---

## ⚠️ Interview trap

The scaling is not required to make probabilities sum to one.

Softmax already does that.

The purpose of scaling is to control the magnitude of the logits entering softmax.

---

## 📌 Takeaway

> sqrt(d_k) scaling keeps attention logits stable as vector dimensionality grows.

---

# Q42. Why does softmax saturation cause a gradient problem?

## ⚡ 30-second answer

When softmax receives extremely large relative logits, its output can become almost one-hot.

Once probabilities become very close to 0 or 1, small changes in logits may barely change the output, causing many softmax derivatives to become very small.

---

## 🔬 Softmax

Conceptually:

```text
p_i = exp(z_i) / sum(exp(z_j))
```

Consider:

```text
z = [1, 2, 3]
```

The probability mass is distributed across the entries.

Now consider:

```text
z = [0, 100, 0]
```

Then approximately:

```text
softmax(z) = [0, 1, 0]
```

---

## 🔬 Gradient intuition

The softmax derivative contains terms like:

```text
p_i * (delta_ij - p_j)
```

If:

```text
p ≈ 1
```

or:

```text
p ≈ 0
```

many derivative terms become tiny.

---

## 🧠 Chain

```text
large attention logits
        ↓
extremely peaked softmax
        ↓
softmax saturation
        ↓
small gradients
        ↓
less effective learning
```

---

## 🎤 Connection to Q41

This is exactly why attention divides Query-Key scores by:

```text
sqrt(d_k)
```

before softmax.

---

## 📌 Takeaway

> Extremely confident softmax outputs can have tiny derivatives and poor gradient flow.

---

# Q43. Why use different learned projections for Q, K and V?

## ⚡ 30-second answer

Query, Key and Value have different computational jobs.

Separate learned projection matrices allow the model to learn one representation for matching and another representation for the information that should actually be transferred.

---

## 🔬 Standard projections

```text
Q = X * WQ
K = X * WK
V = X * WV
```

with independently learned:

```text
WQ
WK
WV
```

---

## 🧠 Different roles

```text
Query space
→ what am I looking for?

Key space
→ what do I provide for matching?

Value space
→ what information should actually be transferred?
```

---

## 💡 Example

A token may ask:

```text
Which previous token describes the subject?
```

Q and K can learn representations that make subject matching easy.

Once the relevant token is found, its Value vector can contain the information that should be incorporated into the current representation.

---

## ⚠️ Interview trap

Matching information and transferred information do not have to use exactly the same representation.

That is one major reason separate projections are useful.

---

## 📌 Takeaway

> Separate Q, K and V projections let attention specialize matching and information transfer independently.

---

# Q44. What is the computational complexity of self-attention?

## ⚡ 30-second answer

For sequence length n and hidden dimension d, the main Query-Key score computation has approximate time complexity:

```text
O(n^2 * d)
```

The attention matrix has shape:

```text
n × n
```

so its memory requirement with respect to sequence length is:

```text
O(n^2)
```

---

## 🔬 Where does n^2 come from?

Suppose:

```text
Q shape = n × d
K shape = n × d
```

Then:

```text
Q * K^T
```

has shape:

```text
(n × d) * (d × n)
=
n × n
```

Every token can interact with every other token.

That gives approximately:

```text
n^2 pairwise interactions
```

---

## 💡 Example

If:

```text
n = 1000
```

the attention matrix contains approximately:

```text
1000 × 1000
=
1,000,000 positions
```

Now double the context length:

```text
n → 2n
```

Then:

```text
(2n)^2 = 4n^2
```

So doubling sequence length can roughly quadruple the pairwise attention interactions and attention-matrix memory.

---

## 🧠 Why this matters for LLMs

Modern LLMs can process very long contexts.

Therefore:

```text
longer context
      ↓
quadratically larger attention matrix
      ↓
more computation
+
more memory
```

This motivates efficient attention methods and alternative architectures.

---

## 🎤 Interview-ready answer

> Standard self-attention has approximately O(n^2 d) score-computation cost and O(n^2) attention-matrix memory because every token can interact with every other token.

---

## 📌 Takeaway

> The power of global token interaction comes with quadratic sequence-length cost.

---

# ✅ Self-Attention Section Complete

```text
Q38  Problem solved by self-attention             ✅
Q39  Query, Key and Value                         ✅
Q40  Attention-score calculation                  ✅
Q41  Why divide by sqrt(d_k)                      ✅
Q42  Softmax saturation                           ✅
Q43  Separate Q/K/V projections                   ✅
Q44  Self-attention complexity                    ✅
```

## ⏭️ Next

### Q45-Q50 — Multi-Head Attention, Causal Attention & KV Cache
