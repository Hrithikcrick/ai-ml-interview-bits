# 🚀 Complete AI + ML Interview Roadmap

> A structured journey from neural-network fundamentals to production LLM and Agent systems.

---

# 🧠 01. Neural Network Fundamentals

- Features and representation learning
- Weights
- Bias
- Linear layers
- Nonlinearity
- ReLU
- Dying ReLU
- Leaky ReLU
- Sigmoid
- Vanishing gradients

# ⚡ 02. Activation Functions

- Why activations are necessary
- ReLU
- Sigmoid
- Leaky ReLU
- GELU
- Tanh
- SiLU / Swish

# 🎯 03. Weight Initialization & Gradient Flow

- Zero initialization
- Symmetry breaking
- Small random initialization
- Xavier / Glorot initialization
- He initialization
- Xavier vs He
- Vanishing gradients
- Exploding gradients
- Forward signal preservation
- Backward gradient preservation

# ⚖️ 04. Normalization

- Why normalization is needed
- BatchNorm
- Training vs inference BatchNorm
- LayerNorm
- BatchNorm vs LayerNorm
- RMSNorm
- Pre-Norm
- Post-Norm
- Residual connections
- Hidden-state scale

# 🔍 05. Self-Attention

- Query
- Key
- Value
- Attention scores
- Scaled dot-product attention
- Softmax saturation
- Q/K/V projections
- Attention complexity

# 🤖 06. Transformer Attention

- Multi-Head Attention
- Attention heads
- Causal masking
- Parallel Transformer training
- KV Cache
- Training-time vs inference-time attention

# 🧩 07. Transformer FFN / MLP

- Feed-Forward Networks
- Hidden-dimension expansion
- GELU
- SwiGLU
- Modern LLM activations

# 📍 08. Positional Information

- Why position is required
- Absolute positional embeddings
- Sinusoidal positional encoding
- RoPE
- Relative positional information

# 📏 09. Context Length & Modern Transformers

- RoPE limitations
- Context length
- Model dimension
- Attention memory growth
- Padding masks
- Causal masks
- Encoder-only models
- Encoder-decoder models
- Decoder-only models

# 📚 10. Language Modeling

- Decoder-only training objective
- Cross-entropy
- Logits
- Teacher forcing
- Perplexity
- Exposure bias

# ⚙️ 11. Deep Learning Optimization

- Backpropagation vs optimizer
- SGD
- Adam
- AdamW
- First moment
- Second moment
- Learning-rate warmup

# 🛠️ 12. Advanced Training

- L2 regularization
- Weight decay
- Gradient clipping
- Gradient accumulation
- Effective batch size
- Mixed precision
- Loss scaling

# 🚀 13. LLM Inference

- KV Cache
- KV memory
- MHA
- MQA
- GQA
- Speculative decoding
- Quantization

# 🖥️ 14. LLM Serving Systems

- Prefill
- Decode
- TTFT
- TPOT
- Memory bandwidth
- Continuous batching
- PagedAttention
- Multi-user serving

# ⚡ 15. Advanced LLM Inference

- FlashAttention
- Prefix caching
- Model quantization
- KV-cache quantization
- Latency
- Throughput
- Inference trade-offs

# 🧠 16. Mixture of Experts

- MoE architecture
- Experts
- Router
- Top-k routing
- Load balancing
- Expert capacity
- Router collapse
- Expert parallelism

# 🌐 17. Distributed AI Training

- Data parallelism
- Model parallelism
- Tensor parallelism
- Pipeline parallelism
- Expert parallelism
- All-reduce
- All-to-all communication

# 🏗️ 18. Modern Transformer Architecture

- RMSNorm
- SwiGLU
- MQA
- GQA
- Pre-Norm
- RoPE
- Complete Transformer forward pass

# 🏋️ 19. LLM Pretraining

- Causal language modeling
- Cross-entropy training
- Teacher forcing
- Parallel token training
- Causal masking
- Training vs inference

# 🎯 20. LLM Alignment

- Supervised fine-tuning
- RLHF
- Reward models
- PPO
- DPO
- Reward hacking

# 🔎 21. RAG Fundamentals

- Retrieval-Augmented Generation
- Complete RAG pipeline
- Embeddings
- Vector similarity
- Chunking
- Dense retrieval
- Keyword retrieval

# 🔥 22. Advanced RAG

- Top-k retrieval
- Reranking
- Hybrid search
- Query transformation
- Failure modes
- RAG evaluation
- RAG vs fine-tuning
- Metadata filtering

# 🏭 23. Production RAG

- Parent-child retrieval
- Contextual compression
- Multi-hop RAG
- Conversational RAG
- PDF and table handling
- Grounding
- Citations
- Prompt injection
- Freshness
- Versioning
- Caching
- Latency
- Cost

# 🤖 24. Agentic RAG

- Agentic RAG
- RAG vs long-context
- SQL vs RAG
- Retrieval metrics
- Generation metrics
- Production RAG architecture

# 🔗 25. LangChain

- Documents
- Document loaders
- Text splitters
- Embeddings
- Vector stores
- Retrievers
- Chains
- LCEL
- Runnables
- RunnableMap
- Invoke
- Stream
- Batch

# 🕸️ 26. LangGraph & Agents

- Tools
- Tool calling
- Agents
- Chain vs Agent
- Memory
- Conversational memory
- LangGraph
- State
- Graph workflows

# 🧪 27. RAG & Agent Evaluation

- Query rewriting
- Multi-query retrieval
- RAG Fusion
- Lost-in-the-middle
- Hallucination
- Faithfulness
- Citation correctness
- Observability
- Tracing
- Guardrails
- Human-in-the-loop

# 🔧 28. Fine-Tuning & Efficient Adaptation

- Fine-tuning vs RAG
- LoRA
- PEFT
- Quantization

# 🎲 29. LLM Generation

- Temperature
- Sampling
- Top-k sampling
- Top-p sampling
- Tokenization

# 🏁 Final Goal

Be able to move from:

```text
Neural Network Fundamentals
        ↓
Transformers
        ↓
LLM Training
        ↓
LLM Inference
        ↓
RAG
        ↓
Agents
        ↓
Production AI Systems
```
