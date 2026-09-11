#pragma once
#include <vector>
#include <stdexcept>
#include <numeric>
#include <functional>
#include <sstream>
#include <iostream>
#include <iomanip>

// N-dimensional tensor, flat vector<double> underneath.
class Tensor {
public:
    std::vector<size_t> shape;   // e.g. {2,3} = 2x3 matrix
    std::vector<double> data;    // row-major

    Tensor() = default;

    explicit Tensor(const std::vector<size_t>& shape_, double fill = 0.0)
        : shape(shape_), data(numel(shape_), fill) {}

    Tensor(const std::vector<size_t>& shape_, std::vector<double> data_)
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

    double& at(const std::vector<size_t>& idx) { return data[flatten_index(idx)]; }
    double at(const std::vector<size_t>& idx) const { return data[flatten_index(idx)]; }

    // handles A+B elementwise, plus a hack for broadcasting a single scalar tensor
    // (shape {1}) against anything else. doesn't handle general broadcasting.
    Tensor elementwise(const Tensor& other, const std::function<double(double,double)>& op) const {
        if (other.shape.size() == 1 && other.shape[0] == 1) {
            Tensor result(shape);
            for (size_t i = 0; i < data.size(); ++i) result.data[i] = op(data[i], other.data[0]);
            return result;
        }
        if (shape != other.shape)
            throw std::invalid_argument("shapes don't match");
        Tensor result(shape);
        for (size_t i = 0; i < data.size(); ++i) result.data[i] = op(data[i], other.data[i]);
        return result;
    }

    Tensor operator+(const Tensor& o) const { return elementwise(o, std::plus<double>()); }
    Tensor operator-(const Tensor& o) const { return elementwise(o, std::minus<double>()); }
    Tensor operator*(const Tensor& o) const { return elementwise(o, std::multiplies<double>()); }
    Tensor operator/(const Tensor& o) const { return elementwise(o, std::divides<double>()); }

    Tensor scale(double k) const {
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
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < m; ++j) {
                double sum = 0.0;
                for (size_t p = 0; p < k; ++p)
                    sum += at({i, p}) * other.at({p, j});
                result.at({i, j}) = sum;
            }
        }
        return result;
    }

    Tensor transpose() const {
        if (rank() != 2)
            throw std::invalid_argument("transpose only works on 2D for now");
        Tensor result({shape[1], shape[0]});
        for (size_t i = 0; i < shape[0]; ++i)
            for (size_t j = 0; j < shape[1]; ++j)
                result.at({j, i}) = at({i, j});
        return result;
    }

    Tensor reshape(const std::vector<size_t>& new_shape) const {
        if (numel(new_shape) != data.size())
            throw std::invalid_argument("reshape can't change the total number of elements");
        return Tensor(new_shape, data);
    }

    double sum() const {
        double s = 0.0;
        for (double v : data) s += v;
        return s;
    }

    double mean() const { return data.empty() ? 0.0 : sum() / data.size(); }

    double dot(const Tensor& other) const {
        if (rank() != 1 || other.rank() != 1 || shape[0] != other.shape[0])
            throw std::invalid_argument("dot needs two 1D tensors of the same length");
        double s = 0.0;
        for (size_t i = 0; i < shape[0]; ++i) s += data[i] * other.data[i];
        return s;
    }

    std::string to_string() const {
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(4);
        print_recursive(oss, 0, 0, data.size());
        return oss.str();
    }

    void print() const { std::cout << to_string() << std::endl; }

private:
    // builds up the nested [ ] formatting one dimension at a time
    void print_recursive(std::ostringstream& oss, size_t dim, size_t offset, size_t count) const {
        if (dim == shape.size()) {
            oss << data[offset];
            return;
        }
        if (dim == shape.size() - 1) {
            oss << "[";
            for (size_t i = 0; i < shape[dim]; ++i) {
                if (i) oss << ", ";
                oss << data[offset + i];
            }
            oss << "]";
            return;
        }
        size_t block = count / shape[dim];
        oss << "[";
        for (size_t i = 0; i < shape[dim]; ++i) {
            if (i) oss << ",\n" << std::string(dim + 1, ' ');
            print_recursive(oss, dim + 1, offset + i * block, block);
        }
        oss << "]";
    }
};