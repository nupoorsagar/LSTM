# LSTM Text Generation - Architecture Visualization

## Model Architecture Diagram

```
╔═══════════════════════════════════════════════════════════════════╗
║                    INPUT SEQUENCE                                 ║
║            [character indices: length=100]                        ║
║                   shape: (batch_size, 100)                        ║
╚═══════════════════════════════════════════════════════════════════╝
                              │
                              ▼
┌───────────────────────────────────────────────────────────────────┐
│                    EMBEDDING LAYER                                │
│                                                                    │
│  Purpose: Convert character indices to dense vectors              │
│  Input:   (batch_size, 100)                                      │
│  Output:  (batch_size, 100, 256)                                 │
│  Params:  vocab_size × embedding_dim = 37 × 256 = 9,472         │
│                                                                    │
│  Each character → 256-dimensional vector                          │
│  Learned during training                                          │
└───────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌───────────────────────────────────────────────────────────────────┐
│                    LSTM LAYER 1                                    │
│                                                                    │
│  ┌──────────────────────────────────────────────────┐            │
│  │  Hidden State (h)        Cell State (c)          │            │
│  │      256 units              256 units             │            │
│  │         ↓                      ↓                  │            │
│  │    ┌─────────────────────────────────┐           │            │
│  │    │     LSTM Cell (t=0)              │           │            │
│  │    │  ┌────────┬────────┬─────────┐  │           │            │
│  │    │  │ Forget │ Input  │ Output  │  │           │            │
│  │    │  │  Gate  │  Gate  │  Gate   │  │           │            │
│  │    │  └────────┴────────┴─────────┘  │           │            │
│  │    └─────────────────────────────────┘           │            │
│  │         ↓                      ↓                  │            │
│  │    [Process all 100 time steps]                   │            │
│  │         ↓                      ↓                  │            │
│  │    ┌─────────────────────────────────┐           │            │
│  │    │     LSTM Cell (t=99)             │           │            │
│  │    └─────────────────────────────────┘           │            │
│  └──────────────────────────────────────────────────┘            │
│                                                                    │
│  Input:   (batch_size, 100, 256)                                 │
│  Output:  (batch_size, 100, 256)                                 │
│  Params:  4 × (256 × 256 + 256 × 256 + 256) ≈ 525K             │
│                                                                    │
│  Returns sequences for next layer                                 │
└───────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌───────────────────────────────────────────────────────────────────┐
│                    DROPOUT LAYER 1                                 │
│                                                                    │
│  Purpose: Regularization to prevent overfitting                   │
│  Rate:    0.2 (20% of neurons randomly set to 0)                 │
│  Input:   (batch_size, 100, 256)                                 │
│  Output:  (batch_size, 100, 256)                                 │
│  Params:  0                                                        │
│                                                                    │
│  Only active during training                                      │
└───────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌───────────────────────────────────────────────────────────────────┐
│                    LSTM LAYER 2                                    │
│                                                                    │
│  Similar to LSTM Layer 1 but:                                     │
│  - Processes output from Layer 1                                  │
│  - return_sequences=False                                         │
│  - Returns only final hidden state                                │
│                                                                    │
│  Input:   (batch_size, 100, 256)                                 │
│  Output:  (batch_size, 256)                                      │
│  Params:  4 × (256 × 256 + 256 × 256 + 256) ≈ 525K             │
│                                                                    │
│  Final hidden state encodes entire sequence                       │
└───────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌───────────────────────────────────────────────────────────────────┐
│                    DROPOUT LAYER 2                                 │
│                                                                    │
│  Rate:    0.2 (20% dropout)                                       │
│  Input:   (batch_size, 256)                                      │
│  Output:  (batch_size, 256)                                      │
│  Params:  0                                                        │
└───────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌───────────────────────────────────────────────────────────────────┐
│                    DENSE OUTPUT LAYER                              │
│                                                                    │
│  Purpose: Map LSTM output to character probabilities              │
│  Activation: Softmax                                              │
│  Input:   (batch_size, 256)                                      │
│  Output:  (batch_size, 37)  [probability for each character]     │
│  Params:  256 × 37 + 37 = 9,509                                  │
│                                                                    │
│  Softmax: exp(zi) / Σexp(zj)                                     │
│  Ensures outputs sum to 1.0                                       │
└───────────────────────────────────────────────────────────────────┘
                              │
                              ▼
╔═══════════════════════════════════════════════════════════════════╗
║                    OUTPUT PREDICTIONS                              ║
║          [probability distribution over vocabulary]                ║
║                   shape: (batch_size, 37)                         ║
║                                                                    ║
║  Example: [0.05, 0.12, 0.03, ..., 0.08]                          ║
║           Higher probability = more likely next character         ║
╚═══════════════════════════════════════════════════════════════════╝
```

## Total Parameters

```
Layer                    Parameters
─────────────────────────────────────
Embedding                9,472
LSTM Layer 1             525,312
Dropout 1                0
LSTM Layer 2             525,312
Dropout 2                0
Dense Output             9,509
─────────────────────────────────────
TOTAL                    ~1,070,000
```

## Data Flow Example

```
Input Sequence (100 characters):
"to be or not to be that is the question whether tis nobler in the mind to suffer the slings and ar"

                              ↓
                     Convert to indices
                              ↓
Indices: [t=31, o=26, ' '=0, b=13, e=16, ...]
Shape: (1, 100)
                              ↓
                     Embedding Layer
                              ↓
Embeddings: Each index → 256-dim vector
Shape: (1, 100, 256)
                              ↓
                     LSTM Layer 1
                              ↓
Sequences: Process all time steps
Shape: (1, 100, 256)
                              ↓
                     Dropout (training)
                              ↓
                     LSTM Layer 2
                              ↓
Final State: Encode entire sequence
Shape: (1, 256)
                              ↓
                     Dense + Softmax
                              ↓
Probabilities: One per character
Shape: (1, 37)

Example Output:
[' ': 0.05, 'a': 0.12, 'b': 0.03, ..., 'r': 0.18, ..., 'z': 0.01]
                              ↓
              Sample next character: 'r'
                              ↓
           Append to sequence, repeat
```

## LSTM Cell Internal Structure

```
                    ┌──────────────────────────────┐
                    │        LSTM CELL             │
                    │                              │
        xt          │  ┌────────────────────────┐ │
    (input) ────────┼─→│   Forget Gate (ft)     │ │
                    │  │   σ(Wf·[ht-1,xt]+bf)  │ │──→ ×
        ht-1        │  └────────────────────────┘ │   │
    (prev hidden) ──┤                              │   ↓
                    │  ┌────────────────────────┐ │  ct-1
        ct-1        │  │   Input Gate (it)      │ │ (prev cell)
    (prev cell) ────┼─→│   σ(Wi·[ht-1,xt]+bi)  │ │
                    │  └────────────────────────┘ │
                    │           ×                  │
                    │  ┌────────────────────────┐ │
                    │  │   Cell Candidate (C̃t) │ │
                    │  │  tanh(Wc·[ht-1,xt]+bc)│ │
                    │  └────────────────────────┘ │
                    │           │                  │
                    │           ↓        ┌────────┐│
                    │          ct  ─────→│  tanh  ││
                    │  (new cell state)  └────────┘│
                    │                       │       │
                    │  ┌────────────────────────┐  │
                    │  │   Output Gate (ot)     │  │
                    │  │   σ(Wo·[ht-1,xt]+bo)  │  │
                    │  └────────────────────────┘  │
                    │           ↓         ×         │
                    │           └─────────┘         │
                    │                 ↓             │
                    │                ht             │
                    │          (new hidden state)   │
                    └──────────────────────────────┘

Key Components:
- ft (Forget Gate): What to forget from cell state
- it (Input Gate): What new info to add to cell state
- C̃t (Cell Candidate): New candidate values
- ot (Output Gate): What to output
- ct (Cell State): Long-term memory
- ht (Hidden State): Short-term memory / output
```

## Training Process Flow

```
┌─────────────────────────────────────────────────────────┐
│                    TRAINING LOOP                         │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼
        ┌──────────────────────────────────────┐
        │    Load Batch (128 sequences)        │
        │    X: (128, 100) indices            │
        │    y: (128, 37) one-hot             │
        └──────────────────────────────────────┘
                           │
                           ▼
        ┌──────────────────────────────────────┐
        │         Forward Pass                  │
        │  - Embedding                          │
        │  - LSTM Layer 1                       │
        │  - Dropout                            │
        │  - LSTM Layer 2                       │
        │  - Dropout                            │
        │  - Dense + Softmax                    │
        │  → predictions: (128, 37)            │
        └──────────────────────────────────────┘
                           │
                           ▼
        ┌──────────────────────────────────────┐
        │      Calculate Loss                   │
        │  Categorical Cross-Entropy:           │
        │  L = -Σ(y_true × log(y_pred))        │
        └──────────────────────────────────────┘
                           │
                           ▼
        ┌──────────────────────────────────────┐
        │      Backward Pass                    │
        │  - Calculate gradients                │
        │  - Backpropagate through time         │
        │  - Gradient clipping (if needed)      │
        └──────────────────────────────────────┘
                           │
                           ▼
        ┌──────────────────────────────────────┐
        │      Update Weights                   │
        │  Adam Optimizer:                      │
        │  - Update embedding                   │
        │  - Update LSTM weights                │
        │  - Update dense layer                 │
        └──────────────────────────────────────┘
                           │
                           ▼
        ┌──────────────────────────────────────┐
        │      Validation Check                 │
        │  Every epoch:                         │
        │  - Calculate val_loss                 │
        │  - Check for improvement              │
        │  - Save best model                    │
        │  - Early stopping check               │
        └──────────────────────────────────────┘
                           │
                           ▼
                  Continue until:
           - Max epochs reached (30)
           - Early stopping triggered
           - Loss plateau detected
```

## Text Generation Process

```
┌─────────────────────────────────────────────────────┐
│              TEXT GENERATION                         │
└─────────────────────────────────────────────────────┘
                        │
                        ▼
        ┌────────────────────────────────┐
        │   Initialize with seed text    │
        │   "to be or not to be"         │
        │   (pad/truncate to 100 chars)  │
        └────────────────────────────────┘
                        │
                        ▼
        ┌────────────────────────────────┐
        │   Convert to indices           │
        │   [31,26,0,13,16,...]          │
        └────────────────────────────────┘
                        │
                        ▼
        ┌────────────────────────────────┐
        │   Forward pass through model   │
        │   → probability distribution   │
        │   [' ':0.05, 'a':0.12, ...]   │
        └────────────────────────────────┘
                        │
                        ▼
        ┌────────────────────────────────┐
        │   Apply temperature            │
        │   T=0.5: More conservative     │
        │   T=1.0: Balanced              │
        │   T=1.5: More creative         │
        │                                │
        │   p' = exp(log(p)/T)          │
        │   p' = p' / sum(p')           │
        └────────────────────────────────┘
                        │
                        ▼
        ┌────────────────────────────────┐
        │   Sample next character        │
        │   Use probabilities            │
        │   → 'r' (example)             │
        └────────────────────────────────┘
                        │
                        ▼
        ┌────────────────────────────────┐
        │   Append to generated text     │
        │   "to be or not to be r"      │
        └────────────────────────────────┘
                        │
                        ▼
        ┌────────────────────────────────┐
        │   Update input sequence        │
        │   Shift left, add new char     │
        │   "o be or not to be r"       │
        └────────────────────────────────┘
                        │
                        ▼
            Repeat for N characters
                        │
                        ▼
        ┌────────────────────────────────┐
        │   Return generated text        │
        │   (seed + new characters)      │
        └────────────────────────────────┘
```

## Architecture Comparison Visual

```
┌────────────────────────────────────────────────────────────────┐
│                   ARCHITECTURE COMPARISON                       │
└────────────────────────────────────────────────────────────────┘

Small LSTM (128 units, 1 layer)
Params: ~500K
Speed:  ████████████████████ (Fastest)
Quality: ████████████ (Good)

Standard LSTM (256 units, 2 layers)
Params: ~2M
Speed:  ████████████ (Medium)
Quality: ████████████████ (Better)

Large LSTM (512 units, 2 layers)
Params: ~8M
Speed:  ██████ (Slow)
Quality: ████████████████████ (Best)

Deep LSTM (256 units, 4 layers)
Params: ~3M
Speed:  ████████ (Slow)
Quality: ████████████████████ (Best)

Bidirectional LSTM (256 units, 2 layers)
Params: ~4M
Speed:  ████ (Slowest)
Quality: ████████████████████ (Excellent)

GRU (256 units, 2 layers)
Params: ~1.5M
Speed:  ████████████████ (Faster)
Quality: ████████████████ (Good)
```

## Memory and Computation Requirements

```
┌────────────────────────────────────────────────────────┐
│              RESOURCE REQUIREMENTS                      │
└────────────────────────────────────────────────────────┘

Training (Standard LSTM):
┌────────────────────┬────────────────────────┐
│ Component          │ Memory                 │
├────────────────────┼────────────────────────┤
│ Model parameters   │ ~25 MB                 │
│ Optimizer state    │ ~50 MB                 │
│ Gradients          │ ~25 MB                 │
│ Activations        │ ~100-500 MB (varies)   │
│ Training data      │ ~50 MB (batch)         │
├────────────────────┼────────────────────────┤
│ TOTAL (CPU)        │ ~500 MB - 1 GB         │
│ TOTAL (GPU)        │ ~2 GB - 4 GB           │
└────────────────────┴────────────────────────┘

Inference (Generation):
┌────────────────────┬────────────────────────┐
│ Component          │ Memory                 │
├────────────────────┼────────────────────────┤
│ Model parameters   │ ~25 MB                 │
│ Input sequence     │ < 1 MB                 │
│ Activations        │ ~10 MB                 │
├────────────────────┼────────────────────────┤
│ TOTAL              │ ~50 MB                 │
└────────────────────┴────────────────────────┘

Computation (per epoch on 5MB dataset):
┌────────────────────┬────────────────────────┐
│ Hardware           │ Time                   │
├────────────────────┼────────────────────────┤
│ CPU (Intel i7)     │ ~2 minutes             │
│ GPU (RTX 3060)     │ ~20 seconds            │
│ GPU (RTX 4090)     │ ~10 seconds            │
│ TPU (Cloud)        │ ~5 seconds             │
└────────────────────┴────────────────────────┘
```

This visualization guide helps understand the complete architecture and data flow through the LSTM text generation system.
