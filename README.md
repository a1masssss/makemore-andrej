# Makemore: Character-Level Language Models for Name Generation

A neural network journey from bigram models to deep MLPs for generating human-like names. Implemented as part of [Andrej Karpathy's Neural Networks: Zero to Hero](https://www.youtube.com/watch?v=P6sfmUTpUmc&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&index=4) lecture series.

## 📊 Dataset Statistics

- **Total names**: 32,033
- **Unique characters**: 26 (a-z) + 1 special token (.)
- **Total bigrams**: 228,146
- **Vocabulary size**: 27

### Dataset Splits (makemore2)
- **Training set**: 182,580 examples (80%)
- **Validation set**: 22,767 examples (10%)
- **Test set**: 22,799 examples (10%)

---

## 🔬 Part 1: Bigram Language Model (`makemore1.ipynb`)

### Overview
Implementation of a character-level bigram model using two approaches:
1. **Count-based statistical model** with smoothing
2. **Neural network approach** learning the same statistics through gradient descent

### Architecture
- **Input**: One-hot encoded characters (27-dimensional)
- **Model**: Single linear layer (27 × 27 weight matrix)
- **Output**: Probability distribution over next character
- **Loss**: Negative log-likelihood + L2 regularization (0.01)

### Training Configuration
```python
Training iterations: 100
Learning rate: 0.1
Regularization: 0.01 * (W**2).mean()
Optimizer: Manual SGD
```

### Results
- **Final Loss**: 2.425
- Successfully generates plausible names like:
  - `anugeenvi`
  - `mabidushan`
  - `stan`
  - `silaylelaremah`

### Key Insights
- Neural network learns the same probability distribution as the count-based model
- Smoothing (adding 1 to counts) is crucial for handling unseen bigrams
- Log-likelihood measures model confidence in predictions

### Example Probability Analysis
For the name "almajq":
```
.a: 0.138
al: 0.075
lm: 0.004
ma: 0.389
aj: 0.005
jq: 0.000
q.: 0.097
Average negative log-likelihood: 3.79
```

---

## 🚀 Part 2: Multi-Layer Perceptron with Embeddings (`makemore2.ipynb`)

### Overview
A significantly more sophisticated model using:
- **Character embeddings** (learned distributed representations)
- **Context window** (3 previous characters)
- **Deep architecture** (5 hidden layers)
- **Batch normalization** for training stability

### Architecture Details

#### Model Configuration
```python
Embedding dimension: 10
Block size (context): 3 characters
Hidden layer size: 100 neurons
Number of hidden layers: 5
Activation function: Tanh
Normalization: Batch Normalization
```

#### Layer Structure
```
Input: 3 characters → Embedding (27 × 10)
  ↓
Linear(30 → 100) → Tanh
  ↓
Linear(100 → 100) → BatchNorm → Tanh
  ↓
Linear(100 → 100) → BatchNorm → Tanh
  ↓
Linear(100 → 100) → BatchNorm → Tanh
  ↓
Linear(100 → 100) → BatchNorm → Tanh
  ↓
Linear(100 → 27) → BatchNorm
  ↓
Output: Probability distribution (27 classes)
```

### Model Statistics
- **Total parameters**: 47,351
- **Trainable parameters**: 47,351

#### Parameter Breakdown
| Layer | Parameters | Shape |
|-------|-----------|-------|
| Embedding (C) | 270 | (27, 10) |
| Layer 1 Weight | 3,000 | (100, 30) |
| Layer 1 Bias | 100 | (100,) |
| Layer 2-5 Weights | 10,000 each | (100, 100) |
| Layer 2-5 Biases | 100 each | (100,) |
| Output Weight | 2,700 | (27, 100) |
| Output Bias | 27 | (27,) |
| BatchNorm params | 200 × 5 | γ and β |

### Training Configuration
```python
Maximum steps: 200,000 (early stopped at 1,000)
Batch size: 32
Initial learning rate: 0.1 (steps 0-100k)
Reduced learning rate: 0.01 (steps 100k-200k)
Optimizer: Manual SGD with gradient descent
Weight initialization: Kaiming/He initialization (scaled by fan_in^0.5)
Gain for hidden layers: 5/3 (accounts for Tanh nonlinearity)
```

### Results

#### Initial Training Loss
```
Step       0 / 200000: 3.2972
Step   1,000 / 200000: [training continued]
```

#### Activation Statistics (After Training)
Analysis of Tanh layer saturations:

| Layer | Mean | Std Dev | Saturation % |
|-------|------|---------|--------------|
| Layer 1 | +0.02 | 0.77 | 24.81% |
| Layer 4 | -0.01 | 0.63 | 3.34% |
| Layer 7 | +0.01 | 0.64 | 2.88% |
| Layer 10 | +0.00 | 0.64 | 2.47% |
| Layer 13 | -0.00 | 0.65 | 1.75% |

**Key Observation**: BatchNorm successfully prevents saturation in deeper layers (from 24.81% in layer 1 to 1.75% in layer 13).

#### Gradient Statistics
Healthy gradient flow through the network:

| Layer | Mean | Std Dev |
|-------|------|---------|
| Layer 1 | +0.000000 | 3.23e-03 |
| Layer 4 | +0.000000 | 3.01e-03 |
| Layer 7 | -0.000000 | 2.70e-03 |
| Layer 10 | -0.000000 | 2.47e-03 |
| Layer 13 | +0.000000 | 2.44e-03 |

**Key Observation**: Gradients remain consistent across layers, indicating no vanishing gradient problem.

#### Weight Gradient Analysis

| Weight Matrix | Shape | Grad Mean | Grad Std | Grad:Data Ratio |
|---------------|-------|-----------|----------|-----------------|
| Embedding | (27, 10) | +4.8e-05 | 8.86e-03 | 8.62e-03 |
| Hidden 1 | (100, 30) | -8.0e-05 | 9.04e-03 | 2.93e-02 |
| Hidden 2 | (100, 100) | +5.3e-05 | 6.78e-03 | 4.07e-02 |
| Hidden 3 | (100, 100) | +1.4e-04 | 6.11e-03 | 3.65e-02 |
| Hidden 4 | (100, 100) | -1.1e-05 | 5.82e-03 | 3.46e-02 |
| Hidden 5 | (100, 100) | -6.2e-05 | 5.12e-03 | 3.04e-02 |
| Output | (27, 100) | -3.9e-05 | 9.08e-03 | 5.32e-02 |

**Optimal Grad:Data Ratio**: Should be around 1e-3 (0.001) for stable training.

---

## 🛠️ Custom Implementations

### Linear Layer
```python
class Linear:
    def __init__(self, fan_in, fan_out, bias=True):
        self.weight = torch.randn((fan_out, fan_in)) / fan_in ** 0.5
        self.bias = torch.zeros(fan_out) if bias else None
    
    def __call__(self, x):
        self.out = x @ self.weight.T
        if self.bias is not None:
            self.out += self.bias
        return self.out
```

### Batch Normalization Layer
```python
class BatchNorm1d:
    def __init__(self, dim, eps=1e-05, momentum=0.1):
        self.gamma = torch.ones(dim)   # scale parameter
        self.beta = torch.zeros(dim)   # shift parameter
        self.running_mean = torch.zeros(dim)
        self.running_var = torch.ones(dim)
```

**Benefits**:
- Normalizes activations to zero mean and unit variance
- Reduces internal covariate shift
- Allows higher learning rates
- Acts as regularization

### Tanh Activation
```python
class Tanh:
    def __call__(self, x):
        self.out = torch.tanh(x)
        return self.out
```

---

## 📈 Diagnostic Visualizations

### 1. Activation Distribution
Visualizes the distribution of activations in each Tanh layer to detect:
- **Saturation**: When too many neurons output values near ±1
- **Dead neurons**: When neurons consistently output zero
- **Healthy distribution**: Centered around zero with good variance

### 2. Gradient Distribution
Monitors gradient flow through the network to identify:
- **Vanishing gradients**: When gradients become too small
- **Exploding gradients**: When gradients become too large
- **Gradient health**: Consistent gradient magnitudes across layers

### 3. Weight Gradient Distribution
Analyzes the distribution of gradients for weight matrices:
- Ensures all layers are learning
- Verifies proper weight initialization
- Monitors update-to-data ratios

### 4. Update-to-Data Ratio Tracking
Plots the ratio of `(learning_rate × gradient_std) / parameter_std` over time:
- **Optimal range**: ~1e-3 (0.001)
- **Too high**: Learning rate might be too large
- **Too low**: Learning rate might be too small

---

## 🎯 Key Techniques & Concepts

### 1. Weight Initialization
- **Kaiming/He Initialization**: `W = randn(n_out, n_in) / sqrt(n_in)`
- **Gain Adjustment**: Multiply by 5/3 for Tanh to maintain variance
- **Last Layer Scaling**: Multiply by 0.1 to start with lower confidence

### 2. Learning Rate Schedule
```python
lr = 0.1 if step < 100000 else 0.01
```
Step decay reduces learning rate for fine-tuning in later stages.

### 3. Batch Processing
- Enables parallel computation
- Provides better gradient estimates
- Required for Batch Normalization

### 4. Context Window
Using 3 previous characters provides more context than bigrams:
- Bigram: `[a] → [b]`
- Trigram+: `[a, b, c] → [d]`

### 5. Gradient Retention
```python
layer.out.retain_grad()
```
Allows inspection of gradients for intermediate (non-leaf) tensors for debugging.

---

## 🔍 Comparison: Bigram vs MLP

| Metric | Bigram Model | MLP Model |
|--------|--------------|-----------|
| Parameters | 729 | 47,351 |
| Context | 1 character | 3 characters |
| Architecture | Single layer | 6 layers (5 hidden) |
| Training examples | 228,146 | 182,580 |
| Loss (initial) | ~3.3 | 3.297 |
| Loss (final) | 2.425 | [training in progress] |
| Training time | ~100 iterations | 200,000 steps |
| Expressiveness | Low | High |
| Overfitting risk | Low | Higher |

---

## 🧪 Training Diagnostics

### What to Monitor

1. **Loss Curve**: Should decrease smoothly
2. **Activation Saturation**: Should stay below 10-15%
3. **Gradient Magnitudes**: Should be consistent across layers
4. **Update Ratios**: Should be around 1e-3
5. **Parameter Norms**: Should not explode or vanish

### Common Issues & Solutions

| Problem | Symptom | Solution |
|---------|---------|----------|
| Vanishing Gradients | Early layers don't learn | Better initialization, BatchNorm |
| Exploding Gradients | Loss becomes NaN | Lower learning rate, gradient clipping |
| Dead Neurons | No gradient flow | Change initialization, use ReLU |
| Slow Learning | Loss decreases very slowly | Increase learning rate, remove regularization |
| Overfitting | Val loss > Train loss | Add dropout, reduce capacity |

---

## 💡 Key Learnings

1. **Neural networks can learn statistical patterns** that would traditionally be computed with counting and probability
2. **Deeper networks need careful initialization** to avoid vanishing/exploding gradients
3. **Batch Normalization** is crucial for training deep networks effectively
4. **Diagnostic visualizations** are essential for understanding what's happening during training
5. **Character embeddings** capture meaningful relationships between characters
6. **Context matters**: Using multiple previous characters (trigrams) improves generation quality over bigrams
7. **Update-to-data ratios** provide insight into learning rate appropriateness

---

## 🚀 Future Improvements

1. **Complete training**: Run full 200k steps to see final performance
2. **Hyperparameter tuning**: Experiment with:
   - Different embedding dimensions
   - Varying hidden layer sizes
   - Different numbers of layers
   - Learning rate schedules
3. **Advanced architectures**:
   - Residual connections
   - Layer normalization
   - Dropout for regularization
4. **Better evaluation**:
   - Calculate validation loss
   - Measure perplexity
   - Human evaluation of generated names
5. **Generation improvements**:
   - Temperature sampling
   - Top-k sampling
   - Beam search

---

## 📚 References & Credits

This project is based on **Andrej Karpathy's** excellent lecture series:
- 📺 [Neural Networks: Zero to Hero - Lecture 4](https://www.youtube.com/watch?v=P6sfmUTpUmc&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&index=4)
- 🧠 Building character-level language models from scratch
- 🎓 Understanding neural network internals through implementation

### Key Papers
- **BatchNorm**: [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/abs/1502.03167)
- **Weight Init**: [Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification](https://arxiv.org/abs/1502.01852)

---

## 🛠️ Requirements

```python
torch>=1.9.0
matplotlib>=3.3.0
numpy>=1.19.0
```

---

## 📝 License

Educational project for learning purposes.

---

## 🙏 Acknowledgments

Special thanks to **Andrej Karpathy** for making advanced neural network concepts accessible through clear explanations and hands-on implementation. This project follows his teaching methodology of building everything from scratch to truly understand how neural networks work.

---

**Note**: This is a learning project focused on understanding neural network fundamentals rather than achieving state-of-the-art performance. The goal is education, not production deployment.

