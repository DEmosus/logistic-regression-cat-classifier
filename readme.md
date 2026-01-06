# 🐱 Logistic Regression Classifier (Cat vs Non-Cat)

> **A from-scratch implementation of logistic regression as a single-layer neural network — the foundational building block for deep learning.**

---

## 📌 Project Overview

This project implements a **binary logistic regression classifier** to distinguish images of _cats_ from _non-cats_, built **entirely from scratch** using only NumPy.

Though simple, the implementation follows the full deep learning training pipeline:

> **Initialize → Forward Propagation → Cost → Backward Propagation → Gradient Descent → Prediction → Evaluation**

This serves as a crucial pedagogical step toward understanding modern neural networks.

---

## 🎯 Goal

- ✅ Build a binary image classifier without ML frameworks (no TensorFlow/PyTorch).
- ✅ Understand the full training loop at a granular level.
- ✅ Gain intuition for core deep learning concepts:
  - Forward & backward propagation
  - Binary cross-entropy loss
  - Gradient descent optimization
  - Vectorization & computational efficiency
- ✅ Lay the groundwork for multi-layer neural networks.

---

## 🧠 Key Concepts Covered

| Concept                     | Relevance                                         |
| --------------------------- | ------------------------------------------------- |
| Logistic Regression         | Treated as a 1-neuron neural network              |
| Sigmoid Activation          | Non-linearity for binary output                   |
| Binary Cross-Entropy Loss   | Standard loss for binary classification           |
| Vectorized NumPy Operations | Avoid loops for efficiency & scalability          |
| Gradient Descent            | Manual parameter updates via derivatives          |
| Learning Rate Tuning        | Diagnose convergence behavior                     |
| Error Analysis              | Visualize failure cases to motivate deeper models |

---

## 📂 Project Structure

```text
project-root/
├── datasets/                     # Data storage (HDF5 format)
│   ├── train_catvnoncat.h5       # Training set (209 images)
│   └── test_catvnoncat.h5        # Test set (50 images)
├── lr_utils.py                   # Helper functions for loading/preprocessing data
├── logistic_regression.ipynb     # Main Jupyter Notebook (implementation + experiments)
└── README.md                     # Project documentation
```

---

## 📊 Dataset

| Set          | # Images | Shape (before flattening) | Features (after flattening) |
| ------------ | -------- | ------------------------- | --------------------------- |
| **Training** | 209      | `(209, 64, 64, 3)`        | `12,288` (64×64×3)          |
| **Test**     | 50       | `(50, 64, 64, 3)`         | `12,288`                    |

- **Label encoding**:
  - `0` → non-cat
  - `1` → cat
- **Preprocessing**:
  - Flatten: `(m, 64, 64, 3)` → `(12288, m)`
  - Normalize: divide pixel values by `255` → values in `[0, 1]`

---

## 🔄 Workflow

1. **Load Data**

   - Use `lr_utils.py` to load HDF5 files.
   - Visualize sample images & labels.

2. **Preprocess**

   - Flatten & normalize inputs.

3. **Build Model**
   ```python
   def sigmoid(z): ...
   def propagate(w, b, X, Y): ...  # forward + backward
   def optimize(w, b, X, Y, num_iterations, learning_rate): ...
   def predict(w, b, X): ...
   def model(X_train, Y_train, X_test, Y_test, ...): ...
   ```
4. **\*Train & Evaluate**
   - Train via gradient descent.
   - Report train/test accuracy.
   - Plot learning curves.
   - Visualize misclassified examples.
   - Compare learning rates.

# 🧮 Mathematical Model

## Forward Propagation

$$
z = w^T x + b
$$

$$
\hat{y} = \sigma(z) = \frac{1}{1 + e^{-z}}
$$

---

## Binary Cross-Entropy Loss

$$
J = -\frac{1}{m} \sum\_{i=1}^{m} \Big[ y^{(i)} \log(\hat{y}^{(i)}) + (1 - y^{(i)}) \log(1 - \hat{y}^{(i)}) \Big]
$$

---

## Gradients

$$
dw = \frac{1}{m} X ( \hat{Y} - Y )^T
$$

$$
db = \frac{1}{m} \sum\_{i=1}^{m} \big( \hat{y}^{(i)} - y^{(i)} \big)
$$

---

## Gradient Descent Updates

$$
w := w - \alpha \cdot dw
$$

$$
b := b - \alpha \cdot db
$$

---

## 📈 Results

| Metric         | Typical Range |
| -------------- | ------------- |
| Train Accuracy | 70% – 90%     |
| Test Accuracy  | 30% – 40%     |

⚠️ **Note:** Low test accuracy is expected — the model is linear, the dataset is tiny, and overfitting is inevitable. This highlights the limitations of shallow models and motivates deeper architectures.

---

# 🔍 Error Analysis & Learning Curves

- ✅ Visualized misclassified images (e.g., blurry cats, non-cats with cat-like texture).
- ✅ Plotted cost vs. iterations for different learning rates:

$$
\alpha = 0.01 \quad \rightarrow \quad \text{May diverge or oscillate}
$$

$$
\alpha = 0.001 \quad \rightarrow \quad \text{Good convergence (often optimal)}
$$

$$
\alpha = 0.0001 \quad \rightarrow \quad \text{Slow but stable convergence}
$$

---

# 🛠 Technologies Used

| Tool        | Purpose                     |
| ----------- | --------------------------- |
| Python      | Core language               |
| NumPy       | Vectorized math (no loops!) |
| Matplotlib  | Plotting & visualization    |
| h5py        | Read HDF5 datasets          |
| PIL / SciPy | Image loading & display     |

✅ No high-level ML libraries — everything implemented manually.

---

# 🧠 Big-Picture Takeaway

Deep learning is just repeated applications of the same **6-step loop**:

$$
\text{Input} \rightarrow\ \text{Linear} \rightarrow\ \text{Activation} \rightarrow \text{Loss} \rightarrow \text{Gradient} \rightarrow \text{Update}
$$

---

