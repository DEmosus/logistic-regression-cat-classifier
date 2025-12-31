## Purpose

This notes is a foundation builder for deep learning. It focuses on:

- Writing clean, vectorized Python
- Using NumPy instead of loops
- Understanding broadcasting
- Implementing core math functions used in neural networks

> 🔑 **Rule of thumb**:  
> If you’re using loops on NumPy arrays, you’re probably doing it wrong (unless explicitly told).

---

## 1. Python + NumPy Basics

### 1.1 Why NumPy?

Deep learning works with vectors, matrices, and tensors. NumPy allows:

- Element-wise operations
- Fast computation (written in C under the hood)
- Broadcasting (automatic shape matching)

---

## 2. The Sigmoid Function

### 2.1 Mathematical Definition

$$\sigma(x) = \frac{1}{1 + e^{-x}}$$

Used in:

- Logistic Regression
- Neural network activations

### 2.2 Basic Sigmoid (Scalar Only)

```python
import math

def basic_sigmoid(x):
    return 1 / (1 + math.exp(-x))
```

✅ Works only for single numbers  
❌ Fails for lists or arrays:

```python
basic_sigmoid([1,2,3]) # TypeError
```

### 2.3 Vectorized Sigmoid (Correct Way)

```python
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))
```

✔ Works for:

- Scalars
- Vectors
- Matrices

**Example:**

```python
x = np.array([1, 2, 3])
sigmoid(x)
```

**Output:**

```
[0.73105858 0.88079708 0.95257413]
```

---

## 3. Sigmoid Derivative (Backprop Foundation)

### Formula

$$
\sigma'(x) = \sigma(x)(1 - \sigma(x))
$$

### Code

```python
def sigmoid_derivative(x):
    s = sigmoid(x)
    return s \* (1 - s)
```

**Example:**

```python
sigmoid_derivative(np.array([1, 2, 3]))
```

---

## 4. Reshaping Arrays (Very Important)

### 4.1 Shape Basics

- `x.shape` → returns dimensions
- `x.reshape()` → changes dimensions

### 4.2 Image to Vector (Unrolling)

**Why?**  
Neural networks expect column vectors, not images.

**Function**

```python
def image2vector(image):
    return image.reshape(image.shape[0] _ image.shape[1] _ image.shape[2], 1)
```

- **Input shape**: `(`length`, `height`, `depth`)`
- **Output shape**: `(`length × height × depth`, 1)`

---

## 5. Normalizing Rows (Unit Vectors)

### Purpose

- Improves gradient descent convergence
- Ensures consistent scale

### Formula

$$
x_{\text{normalized}} = \frac{x}{\|x\|}
$$

### Implementation

```python
def normalizeRows(x):
    x_norm = np.linalg.norm(x, axis=1, keepdims=True)
    return x / x_norm
```

> **Why `keepdims=True`?**  
> It preserves dimension for broadcasting:  
> `(`2,3`) / (`2,1`) → works`

---

## 6. Broadcasting (Core NumPy Superpower)

Broadcasting allows NumPy to:

- Perform operations on arrays of different shapes
- Avoid explicit loops

**Example:**

```python
x = np.array([[1,2,3],
              [4,5,6]])

x / np.array([[1],[2]]) # Broadcasts [1,2] as column vector
```

Result:

```
[[1.  2.  3. ]
 [2.  2.5 3.]]
```

---

## 7. Softmax Function (Multi-Class Classification)

### Purpose

- Converts scores into probabilities
- Each row sums to 1
- Used in classification outputs

### Formula (Row-Wise)

$$
\text{softmax}(x_i) = \frac{e^{x_i}}{\sum_j e^{x_j}}
$$

### Implementation

```python
def softmax(x):
    x_exp = np.exp(x)
    x_sum = np.sum(x_exp, axis=1, keepdims=True)
    return x_exp / x_sum
```

**Example:**

```python
softmax(np.array([[9,2,5,0,0],
                  [7,5,0,0,0]]))
```

---

## 8. Vectorization vs Loops (Performance Matters)

### Why Vectorization?

- Faster
- Cleaner
- Less error-prone

### Dot Product (Bad vs Good)

❌ **Loop version:**

```python
dot = 0
for i in range(len(x)):
    dot += x[i] \* y[i]
```

✅ **Vectorized:**

```python
np.dot(x, y)
```

### Element-Wise Multiplication

```python
np.multiply(x, y)
```

---

## 9. Loss Functions

### 9.1 L1 Loss (Absolute Error)

$$
L_1 = \sum |y - \hat{y}|
$$

```python
def L1(yhat, y):
    return np.sum(np.abs(yhat - y))
```

### 9.2 L2 Loss (Squared Error)

$$
L_2 = \sum (y - \hat{y})^2
$$

```python
def L2(yhat, y):
    diff = yhat - y
    return np.dot(diff, diff) # Equivalent to np.sum(diff**2)
```

---

## 10. Key Takeaways (Very Important)

> 🔑 **Must-Remember Concepts**

- Always vectorize
- Use NumPy, not loops
- Broadcasting enables shape flexibility
- Shapes matter more than values
- Neural networks are just math + vectors

---

## 11. Mental Model for Deep Learning

Think of deep learning as:

**Input → Linear Algebra → Activation → Loss → Gradient → Update**

> Every function you coded here appears again and again in neural networks.
