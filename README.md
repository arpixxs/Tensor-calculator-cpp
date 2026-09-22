# Tensor-library-cpp

A small, header-only N-dimensional tensor library written in modern C++ (C++17), plus an interactive command-line calculator built on top of it. No dependencies, just drop `Tensor.hpp` into your project.

```cpp
Tensor<> a({2, 2}, {1, 2, 3, 4});
Tensor<> b({2, 2}, {5, 6, 7, 8});

std::cout << a.matmul(b) << std::endl;
// [[19.0000, 22.0000],
//  [43.0000, 50.0000]]
```

## Features

- **Templated:** `Tensor<T>` works with `int`, `float`, `double`, `long long` and other number types. `T` defaults to `double`, so `Tensor<>` is `Tensor<double>`. Shorthands `Tensord`, `Tensorf` and `Tensori` are included.
- N-dimensional, backed by a flat `std::vector<T>` in row-major order
- Element-wise `+`, `-`, `*`, `/`, unary `-`, and `+=` `-=` `*=` `/=`
- Scalar broadcasting: a single-element tensor (shape `{1}`) works against any other tensor, on either side
- Matrix multiplication and transpose (2D)
- `reshape`, `sum`, `mean`, `dot`, and `cast<U>()` to convert between element types
- `==` / `!=` for comparing tensors
- Nested-bracket pretty printing, with `operator<<` for direct streaming
- Clear exceptions for bad shapes, bad indices and integer division by zero
- Interactive CLI: create named tensors and combine them by name
- Unit tests included (no test framework needed)

## Project structure

```
tensor-library-cpp/
├── README.md
├── LICENSE
├── CMakeLists.txt
├── Tensor.hpp              # the tensor class (header-only)
├── main.cpp                # interactive CLI calculator
├── tests/
│   └── test_tensor.cpp     # unit tests
└── .github/workflows/
    └── ci.yml              # builds + runs tests on Linux and macOS
```


## Getting started

Requires a C++17 compiler (g++, clang++, etc) Both files are in the same folder, so `#include "Tensor.hpp"` works with no include-path flags.

**With g++ / clang++:**

```bash
g++ -std=c++17 -O2 -Wall -o tensor-library main.cpp
./tensor-library
```

(on Windows the executable is `tensor-library.exe`)

**With CMake:**

```bash
cmake -S . -B build
cmake --build build
./build/tensor-library
```

**Run the tests:**

```bash
ctest --test-dir build --output-on-failure
```

or without CMake:

```bash
g++ -std=c++17 -O2 -Wall -o test_tensor tests/test_tensor.cpp
./test_tensor
```

## The calculator

Running the program starts a menu-driven calculator:

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
 The calculator works with `double` values.

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
Save result as (blank to skip): C
Saved as C.
```

## Using `Tensor.hpp` as a library

It's header-only, so copy `Tensor.hpp` next to your code and include it:

```cpp
#include "Tensor.hpp"

Tensor<>    a({2, 2}, {1, 2, 3, 4});   // double (the default)
Tensor<int> x({2, 3}, {1, 2, 3, 4, 5, 6});

std::cout << a + a << std::endl;
std::cout << a.transpose() << std::endl;
std::cout << x.matmul(x.transpose()) << std::endl;   // int math, prints without decimals
```

```
[[2.0000, 4.0000],
 [6.0000, 8.0000]]
[[1.0000, 3.0000],
 [2.0000, 4.0000]]
[[14, 32],
 [32, 77]]
```

### Different types, broadcasting, casting

```cpp
Tensor<int> x({2, 3}, {1, 2, 3, 4, 5, 6});

// a {1} tensor broadcasts against anything, on either side
Tensor<int> ten({1}, {10});
std::cout << x + ten << std::endl;
// [[11, 12, 13],
//  [14, 15, 16]]

// convert to another element type
Tensor<double> ratio = x.cast<double>() / Tensor<double>({1}, {4});
std::cout << ratio << std::endl;
// [[0.2500, 0.5000, 0.7500],
//  [1.0000, 1.2500, 1.5000]]
```

### Errors

Bad input throws instead of silently giving a wrong answer:

```cpp
Tensor<int> zeros({2, 3}, 0);
x / zeros;            // std::domain_error: integer division by zero
                      // (float/double follow normal IEEE rules and give inf/nan)

Tensor<int> s({1}, {1});
s += x;               // std::invalid_argument: += can't change s's shape
s = s + x;            // this is fine, it just makes a new tensor
```

### API overview

| Member | Description |
|---|---|
| `Tensor<T>(shape, fill = 0)` | tensor of the given shape filled with one value |
| `Tensor<T>(shape, values)` | tensor from a flat list of values (row-major), throws if the count is wrong |
| `shape`, `data` | public members: the shape vector and the flat data |
| `size()`, `rank()`, `strides()` | number of elements, number of dimensions, row-major strides |
| `at({i, j, ...})` | element access with bounds checking |
| `+ - * /`, unary `-` | element-wise, returns a new tensor |
| `+= -= *= /=` | element-wise in place (shape never changes) |
| `scale(k)` | multiply every element by `k` |
| `matmul(other)` | 2D x 2D matrix multiplication |
| `transpose()` | 2D transpose |
| `reshape(new_shape)` | same data, new shape (element count must match) |
| `cast<U>()` | copy converted to `Tensor<U>` |
| `sum()`, `mean()` | `sum()` returns `T`, `mean()` always returns `double` |
| `dot(other)` | dot product of two 1D tensors |
| `to_string()`, `print()`, `<<` | pretty printing |
| `==`, `!=` | exact comparison of shape and data |

### Things worth knowing

- **Scalars are shape `{1}`.** Broadcasting only handles that one case, not general NumPy-style broadcasting.
- **A single value in braces is a fill value.** `Tensor<int>({3}, {7})` gives `[7, 7, 7]`, not an error. Two or more values in braces are read as data, so `Tensor<int>({3}, {7, 8, 9})` does what you'd expect.
- **`==` is exact.** Comparing floating-point results can fail because of rounding, so it's best used with integers or exactly representable values.
- **`reshape` and `cast` make copies.** There are no views yet.
- **`T` must be a number type** (checked at compile time). `bool` isn't supported.
- Integer overflow isn't checked.

### License

This project is licensed under the MIT License.
