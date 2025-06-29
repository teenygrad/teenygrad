# Autodifferentiation in Teenygrad

This document explains how to use the autodifferentiation system in the Teenygrad tensor implementation.

## Overview

The autodifferentiation system allows you to automatically compute gradients of functions with respect to their inputs. This is essential for training neural networks and other machine learning models.

## Key Components

### 1. Value Struct

Each tensor element is represented by a `Value` struct that contains:

- `data`: The actual numerical value (if computed)
- `grad`: The gradient with respect to this value
- `requires_grad`: Whether this value needs gradients computed
- `operation`: The operation that produced this value
- `dependencies`: References to input values

### 2. Tensor Operations

All tensor operations (add, sub, mult, relu, sigmoid, etc.) automatically build a computation graph and set up gradient computation.

### 3. Backward Pass

The `backward()` method propagates gradients through the computation graph using the chain rule.

## Basic Usage

### Creating Tensors

```rust
// Create a tensor that requires gradients (default)
let mut x = Tensor::new(vec![2, 2]);

// Create a tensor that doesn't require gradients
let y = Tensor::new_no_grad(vec![2, 2]);
```

### Setting Values

```rust
// Set values in the tensor
x.values[0].borrow_mut().data = Some(1.0);
x.values[1].borrow_mut().data = Some(2.0);
```

### Building Computation Graphs

```rust
// Create a computation: z = relu(x + y) * 2
let z1 = x.add(&y);        // x + y
let z2 = z1.relu();        // relu(x + y)
let z3 = z2.mean();        // mean(relu(x + y))
```

### Computing Gradients

```rust
// Zero gradients before backward pass
x.zero_grad();
y.zero_grad();

// Compute gradients
z3.backward();

// Get gradients
let x_grads = x.gradients();
let y_grads = y.gradients();
```

### Optimization

```rust
// Update parameters using gradients
let learning_rate = 0.1;
x.update(learning_rate);
```

## Complete Example

```rust
use teeny_core::tensor::Tensor;

fn main() {
    // Create input tensors
    let mut a = Tensor::new(vec![2, 2]);
    let mut b = Tensor::new(vec![2, 2]);
    
    // Set initial values
    a.values[0].borrow_mut().data = Some(1.0);
    a.values[1].borrow_mut().data = Some(2.0);
    a.values[2].borrow_mut().data = Some(3.0);
    a.values[3].borrow_mut().data = Some(4.0);
    
    b.values[0].borrow_mut().data = Some(5.0);
    b.values[1].borrow_mut().data = Some(6.0);
    b.values[2].borrow_mut().data = Some(7.0);
    b.values[3].borrow_mut().data = Some(8.0);
    
    // Create computation graph
    let c = a.add(&b);        // A + B
    let d = c.relu();         // relu(A + B)
    let result = d.mean();    // mean(relu(A + B))
    
    // Zero gradients
    a.zero_grad();
    b.zero_grad();
    
    // Backward pass
    result.backward();
    
    // Print gradients
    println!("Gradients for A: {:?}", a.gradients());
    println!("Gradients for B: {:?}", b.gradients());
}
```

## Supported Operations

The following operations support autodifferentiation:

- **Arithmetic**: `add`, `sub`, `mult` (matrix multiplication)
- **Activation Functions**: `relu`, `sigmoid`
- **Mathematical**: `log`, `mean`
- **Shape Operations**: `transpose`

## Gradient Computation Rules

### Addition (z = x + y)

- ∂z/∂x = 1
- ∂z/∂y = 1

### Subtraction (z = x - y)

- ∂z/∂x = 1
- ∂z/∂y = -1

### ReLU (z = max(0, x))

- ∂z/∂x = 1 if x > 0, else 0

### Sigmoid (z = 1/(1 + e^(-x)))

- ∂z/∂x = z * (1 - z)

### Log (z = ln(x))

- ∂z/∂x = 1/x if x > 0, else 0

### Mean (z = mean(x₁, x₂, ..., xₙ))

- ∂z/∂xᵢ = 1/n for all i

## Best Practices

1. **Always zero gradients** before each backward pass to avoid gradient accumulation
2. **Use `requires_grad`** to control which tensors need gradients
3. **Check gradient shapes** to ensure they match your expectations
4. **Use appropriate learning rates** when updating parameters

## Limitations

- Matrix multiplication gradients are simplified and may not be fully accurate for complex cases
- The system currently supports only scalar gradients
- Memory usage can be high for large computation graphs
- No support for higher-order derivatives

## Future Improvements

- More sophisticated matrix multiplication gradients
- Support for higher-order derivatives
- Memory optimization for large graphs
- GPU acceleration
- More mathematical operations (exp, sin, cos, etc.)
