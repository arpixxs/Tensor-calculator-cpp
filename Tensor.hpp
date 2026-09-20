#pragma once
#include <cstddef>
#include <vector>
#include <string>
#include <stdexcept>
#include <numeric>
#include <functional>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <type_traits>
#include <utility>

// N-dimensional tensor, flat vector<T> underneath.
// T is the element type and defaults to double, so Tensor<> is the same as
// Tensor<double>. int, long, float etc all work too.
template <typename T = double>
class Tensor {
    static_assert(std::is_arithmetic<T>::value && !std::is_same<T, bool>::value,
                  "Tensor<T> only works with number types (int, float, double, ...)");

public:
    std::vector<size_t> shape;   // e.g. {2,3} = 2x3 matrix
    std::vector<T> data;         // row-major

    Tensor() = default;

    explicit Tensor(const std::vector<size_t>& shape_, T fill = T(0))
        : shape(shape_), data(numel(shape_), fill) {}

    Tensor(const std::vector<size_t>& shape_, std::vector<T> data_)
        : shape(shape_), data(std::move(data_)) {
        if (data.size() != numel(shape))
            throw std::invalid_argument("data/shape size mismatch");
    }

    static size_t numel(const std::vector<size_t>& shape_) {
        return std::accumulate(shape_.begin(), shape_.end(),
                                (size_t)1, std::multiplies<size_t>());
    }

    size_t size() const { return data.size(); }
    size_t rank() const { return shape.size(); }

    std::vector<size_t> strides() const {
        // row-major, so the last dim has stride 1 and it builds up from there
        std::vector<size_t> s(shape.size(), 1);
        for (int i = (int)shape.size() - 2; i >= 0; --i)
            s[i] = s[i + 1] * shape[i + 1];
        return s;
    }

    size_t flatten_index(const std::vector<size_t>& idx) const {
        if (idx.size() != shape.size())
            throw std::invalid_argument("wrong number of indices for this tensor's rank");
        auto s = strides();
        size_t flat = 0;
        for (size_t i = 0; i < idx.size(); ++i) {
            if (idx[i] >= shape[i])
                throw std::out_of_range("index out of range");
            flat += idx[i] * s[i];
        }
        return flat;
    }

    T& at(const std::vector<size_t>& idx) { return data[flatten_index(idx)]; }
    T at(const std::vector<size_t>& idx) const { return data[flatten_index(idx)]; }

    // handles A+B elementwise, plus a hack for broadcasting a single scalar tensor
    // (shape {1}) against anything else. works with the scalar on either side.
    // doesn't handle general broadcasting.
    Tensor elementwise(const Tensor& other, const std::function<T(T,T)>& op) const {
        bool this_is_scalar = shape.size() == 1 && shape[0] == 1;
        bool other_is_scalar = other.shape.size() == 1 && other.shape[0] == 1;

        if (other_is_scalar && !this_is_scalar) {
            Tensor result(shape);
            for (size_t i = 0; i < data.size(); ++i) result.data[i] = op(data[i], other.data[0]);
            return result;
        }
        if (this_is_scalar && !other_is_scalar) {
            Tensor result(other.shape);
            for (size_t i = 0; i < other.data.size(); ++i) result.data[i] = op(data[0], other.data[i]);
            return result;
        }
        if (shape != other.shape)
            throw std::invalid_argument("shapes don't match");
        Tensor result(shape);
        for (size_t i = 0; i < data.size(); ++i) result.data[i] = op(data[i], other.data[i]);
        return result;
    }

    Tensor operator+(const Tensor& o) const { return elementwise(o, std::plus<T>()); }
    Tensor operator-(const Tensor& o) const { return elementwise(o, std::minus<T>()); }
    Tensor operator*(const Tensor& o) const { return elementwise(o, std::multiplies<T>()); }
    Tensor operator/(const Tensor& o) const { return elementwise(o, divide); }
    Tensor operator-() const { return scale(static_cast<T>(-1)); }

    // the compound ones are not allowed to change A's shape. before, a {1} scalar
    // doing += matrix quietly turned into the matrix, now it throws instead.
    // (result gets built first so if something throws, A is left untouched)
    Tensor& operator+=(const Tensor& o) { return elementwise_inplace(o, std::plus<T>()); }
    Tensor& operator-=(const Tensor& o) { return elementwise_inplace(o, std::minus<T>()); }
    Tensor& operator*=(const Tensor& o) { return elementwise_inplace(o, std::multiplies<T>()); }
    Tensor& operator/=(const Tensor& o) { return elementwise_inplace(o, divide); }

    // exact comparison, so be careful with float/double
    bool operator==(const Tensor& o) const { return shape == o.shape && data == o.data; }
    bool operator!=(const Tensor& o) const { return !(*this == o); }

    Tensor scale(T k) const {
        Tensor result(shape);
        for (size_t i = 0; i < data.size(); ++i) result.data[i] = data[i] * k;
        return result;
    }

    // only does 2D x 2D (standard matrix multiplication), didn't need anything fancier for this
    Tensor matmul(const Tensor& other) const {
        if (rank() != 2 || other.rank() != 2)
            throw std::invalid_argument("matmul only works on 2D tensors");
        size_t n = shape[0], k = shape[1], k2 = other.shape[0], m = other.shape[1];
        if (k != k2)
            throw std::invalid_argument("inner dimensions don't match");
        Tensor result({n, m});
        // index the flat buffers directly here (row-major, so row i starts at i*k) -
        // going through at() per element would recompute+allocate a strides vector
        // for every single multiply, which gets brutally slow on anything but toy sizes.
        // loop order is i,p,j so the inner loop walks both rows left to right
        // (cache friendly), the sums still get added up in the same order as before.
        for (size_t i = 0; i < n; ++i) {
            for (size_t p = 0; p < k; ++p) {
                T a = data[i * k + p];
                for (size_t j = 0; j < m; ++j)
                    result.data[i * m + j] += a * other.data[p * m + j];
            }
        }
        return result;
    }

    Tensor transpose() const {
        if (rank() != 2)
            throw std::invalid_argument("transpose only works on 2D for now");
        size_t rows = shape[0], cols = shape[1];
        Tensor result({cols, rows});
        for (size_t i = 0; i < rows; ++i)
            for (size_t j = 0; j < cols; ++j)
                result.data[j * rows + i] = data[i * cols + j];
        return result;
    }

    Tensor reshape(const std::vector<size_t>& new_shape) const {
        if (numel(new_shape) != data.size())
            throw std::invalid_argument("reshape can't change the total number of elements");
        return Tensor(new_shape, data);
    }

    // convert to another element type, e.g. t.cast<int>() (truncates like a normal static_cast)
    template <typename U>
    Tensor<U> cast() const {
        Tensor<U> result(shape);
        for (size_t i = 0; i < data.size(); ++i) result.data[i] = static_cast<U>(data[i]);
        return result;
    }

    T sum() const {
        T s = 0;
        for (T v : data) s += v;
        return s;
    }

    // always a double, even for int tensors (otherwise the mean of {1,2} would come out as 1)
    double mean() const {
        return data.empty() ? 0.0 : static_cast<double>(sum()) / static_cast<double>(data.size());
    }

    T dot(const Tensor& other) const {
        if (rank() != 1 || other.rank() != 1 || shape[0] != other.shape[0])
            throw std::invalid_argument("dot needs two 1D tensors of the same length");
        T s = 0;
        for (size_t i = 0; i < shape[0]; ++i) s += data[i] * other.data[i];
        return s;
    }

    std::string to_string() const {
        // default constructed tensor has a shape of {} but no data at all
        if (shape.empty() && data.empty()) return "[]";
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(4);
        print_recursive(oss, 0, 0, data.size());
        return oss.str();
    }

    void print() const { std::cout << to_string() << std::endl; }

    friend std::ostream& operator<<(std::ostream& os, const Tensor& t) {
        return os << t.to_string();
    }

private:
    // int / 0 is undefined behaviour (usually a crash), so throw for the integer types.
    // doubles/floats just give inf/nan like normal.
    static T divide(T a, T b) {
        if constexpr (std::is_integral<T>::value) {
            if (b == 0) throw std::domain_error("integer division by zero");
        }
        return a / b;
    }

    Tensor& elementwise_inplace(const Tensor& other, const std::function<T(T,T)>& op) {
        Tensor result = elementwise(other, op);
        if (result.shape != shape)
            throw std::invalid_argument("in-place op can't change the shape (use the normal operator instead)");
        data = std::move(result.data);
        return *this;
    }

    // builds up the nested [ ] formatting one dimension at a time
    // (the + in front of the values makes char-sized ints print as numbers, not letters)
    void print_recursive(std::ostringstream& oss, size_t dim, size_t offset, size_t count) const {
        if (dim == shape.size()) {
            oss << +data[offset];
            return;
        }
        if (dim == shape.size() - 1) {
            oss << "[";
            for (size_t i = 0; i < shape[dim]; ++i) {
                if (i) oss << ", ";
                oss << +data[offset + i];
            }
            oss << "]";
            return;
        }
        // a 0 in a non-last dimension (like {0,3}) used to divide by zero right here
        size_t block = shape[dim] == 0 ? 0 : count / shape[dim];
        oss << "[";
        for (size_t i = 0; i < shape[dim]; ++i) {
            if (i) oss << ",\n" << std::string(dim + 1, ' ');
            print_recursive(oss, dim + 1, offset + i * block, block);
        }
        oss << "]";
    }
};

// shorthands so you don't have to write Tensor<double> everywhere
using Tensord = Tensor<double>;
using Tensorf = Tensor<float>;
using Tensori = Tensor<int>;