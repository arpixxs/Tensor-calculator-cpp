# Tensor-calculator-cpp

A small N-dimensional tensor library and interactive calculator written in modern C++ (C++17)

## Features

- N-dimensional `Tensor` class backed by a flat `std::vector<double>`
- Element-wise `+`, `-`, `*`, `/` (with scalar broadcasting)
- Matrix multiplication and transpose (2D)
- Reshape, sum, mean, and dot product
- Nested-bracket pretty printing
- Interactive CLI: create named tensors and combine them by name

## Project structure

```
tensor-calculator-cpp/
├── README.md
├── Tensor.hpp      # Core tensor class (storage + operations)
└── main.cpp        # Interactive CLI calculator
```

## Build

Requires a C++17 compiler (g++, clang++, etc.). Both files live in the same
directory, so no include-path flags are needed.

```bash
g++ -std=c++17 -O2 -Wall -o tensor_calculator main.cpp
```

This produces an executable named `tensor_calculator` (add `.exe` on Windows).
Run it with:

```bash
./tensor_calculator
```

Running the program launches a menu-driven calculator:

```
===== Tensor Calculator =====
 1. Create/store a tensor
 2. Print a stored tensor
 3. Add (A + B)
 4. Subtract (A - B)
 5. Element-wise multiply (A * B)
 6. Element-wise divide (A / B)
 7. Matrix multiply (2D only)
 8. Transpose (2D only)
 9. Reshape
10. Scale by scalar
11. Sum / Mean
12. Dot product (1D only)
13. List stored tensors
 0. Exit
```

Tensors are created by name, then referenced by that name in later operations —
so you can build up a small workspace of tensors and chain operations on them.

### Example session

```
Choice: 1
Name for this tensor: A
Enter rank (number of dimensions) for A: 2
  dim[0] size: 2
  dim[1] size: 2
Enter 4 values for A (row-major order):
1 2 3 4
A stored.

Choice: 1
Name for this tensor: B
Enter rank (number of dimensions) for B: 2
  dim[0] size: 2
  dim[1] size: 2
Enter 4 values for B (row-major order):
5 6 7 8
B stored.

Choice: 7
Name of A: A
Name of B: B
Result:
[[19.0000, 22.0000],
 [43.0000, 50.0000]]
```

## Using `Tensor.hpp` as a library

```cpp
#include "Tensor.hpp"

Tensor a({2, 2}, {1, 2, 3, 4});
Tensor b({2, 2}, {5, 6, 7, 8});

Tensor sum = a + b;
Tensor product = a.matmul(b);
Tensor t = a.transpose();

sum.print();
```



MIT (or your choice — add a `LICENSE` file before publishing).
