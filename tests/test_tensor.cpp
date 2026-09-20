#include "../Tensor.hpp"
#include <iostream>
#include <sstream>
#include <type_traits>
#include <limits>

// tiny test setup 
//  main returns non-zero at the end if anything went wrong.
static int failures = 0;

#define CHECK(cond) \
    do { if (!(cond)) { ++failures; std::cerr << __FILE__ << ":" << __LINE__ << "  FAILED: " #cond "\n"; } } while (0)

#define CHECK_THROWS(expr, ExType) \
    do { \
        bool threw_right_type = false; \
        try { (void)(expr); } catch (const ExType&) { threw_right_type = true; } catch (...) {} \
        if (!threw_right_type) { ++failures; std::cerr << __FILE__ << ":" << __LINE__ << "  DIDN'T THROW " #ExType ": " #expr "\n"; } \
    } while (0)

// Tensor<> with no argument should be the same thing as Tensor<double>
static_assert(std::is_same<Tensor<>, Tensor<double>>::value, "default type should be double");
static_assert(std::is_same<Tensord, Tensor<double>>::value, "Tensord alias");
static_assert(std::is_same<Tensori, Tensor<int>>::value, "Tensori alias");

void test_construction()
{
    Tensord t({2, 3, 4});
    CHECK(t.rank() == 3);
    CHECK(t.size() == 24);
    CHECK(t.sum() == 0.0);
    CHECK(t.strides() == std::vector<size_t>({12, 4, 1}));

    Tensori filled({3}, 7);
    CHECK(filled.sum() == 21);

    CHECK_THROWS(Tensord({2, 2}, {1, 2, 3}), std::invalid_argument);

    Tensord empty;
    CHECK(empty.size() == 0);

    // a shape whose element count overflows size_t must throw, not wrap around
    size_t big = (size_t)1 << 32;
    CHECK_THROWS(Tensord({big, big}), std::length_error);
    CHECK(Tensord({0, 3}).size() == 0);   // a zero dimension is still fine
}

void test_indexing()
{
    Tensord m({2, 3}, {1, 2, 3, 4, 5, 6});
    CHECK(m.at({0, 1}) == 2);
    CHECK(m.at({1, 2}) == 6);
    m.at({1, 0}) = 40;
    CHECK(m.data[3] == 40);
    CHECK_THROWS(m.at({2, 0}), std::out_of_range);
    CHECK_THROWS(m.at({0, 3}), std::out_of_range);
    CHECK_THROWS(m.at({0}), std::invalid_argument);
}

void test_elementwise()
{
    Tensord a({2, 2}, {1, 2, 3, 4});
    Tensord b({2, 2}, {5, 6, 7, 8});
    CHECK(a + b == Tensord({2, 2}, {6, 8, 10, 12}));
    CHECK(b - a == Tensord({2, 2}, {4, 4, 4, 4}));
    CHECK(a * b == Tensord({2, 2}, {5, 12, 21, 32}));
    CHECK(Tensord({2}, {6, 8}) / Tensord({2}, {3, 2}) == Tensord({2}, {2, 4}));
    CHECK(-a == Tensord({2, 2}, {-1, -2, -3, -4}));
    CHECK(a == a);
    CHECK(a != b);
    CHECK_THROWS(a + Tensord({3}, {1, 2, 3}), std::invalid_argument);
}

void test_scalar_broadcast()
{
    Tensord a({2, 2}, {1, 2, 3, 4});
    Tensord s({1}, {10});
    CHECK(a + s == Tensord({2, 2}, {11, 12, 13, 14}));
    CHECK(s + a == Tensord({2, 2}, {11, 12, 13, 14}));
    CHECK(s - a == Tensord({2, 2}, {9, 8, 7, 6}));   // scalar on the left keeps its order
    CHECK(a - s == Tensord({2, 2}, {-9, -8, -7, -6}));
    CHECK(a * s == Tensord({2, 2}, {10, 20, 30, 40}));
    CHECK(Tensord({1}, {12}) / Tensord({3}, {1, 2, 3}) == Tensord({3}, {12, 6, 4}));
}

void test_compound_assignment()
{
    Tensord a({2, 2}, {1, 2, 3, 4});
    Tensord b({2, 2}, {5, 6, 7, 8});

    Tensord c = a;
    c += b;
    CHECK(c == a + b);
    c -= b;
    CHECK(c == a);
    c *= b;
    CHECK(c == a * b);
    c /= b;
    CHECK(c == a);

    // matrix op= scalar is fine, the shape stays the same
    c += Tensord({1}, {1});
    CHECK(c == Tensord({2, 2}, {2, 3, 4, 5}));

    // scalar op= matrix used to silently turn the scalar into a matrix. now it throws
    // and leaves the scalar alone
    Tensord s({1}, {1});
    CHECK_THROWS(s += a, std::invalid_argument);
    CHECK(s == Tensord({1}, {1}));
    CHECK_THROWS(s *= a, std::invalid_argument);

    // scalar op= scalar is fine
    s += Tensord({1}, {2});
    CHECK(s == Tensord({1}, {3}));

    // x += x
    Tensord d({2}, {1, 2});
    d += d;
    CHECK(d == Tensord({2}, {2, 4}));

    CHECK_THROWS(c += Tensord({3}, {1, 2, 3}), std::invalid_argument);
}

void test_scale()
{
    Tensord a({3}, {1, 2, 3});
    CHECK(a.scale(2.0) == Tensord({3}, {2, 4, 6}));
    CHECK(a.scale(0.5) == Tensord({3}, {0.5, 1, 1.5}));
}

void test_matmul()
{
    Tensord a({2, 2}, {1, 2, 3, 4});
    Tensord b({2, 2}, {5, 6, 7, 8});
    CHECK(a.matmul(b) == Tensord({2, 2}, {19, 22, 43, 50}));

    Tensord x({2, 3}, {1, 2, 3, 4, 5, 6});
    Tensord y({3, 2}, {7, 8, 9, 10, 11, 12});
    CHECK(x.matmul(y) == Tensord({2, 2}, {58, 64, 139, 154}));
    CHECK(y.matmul(x).shape == std::vector<size_t>({3, 3}));

    // multiplying by the identity gives the same matrix back
    Tensord id({3, 3}, {1, 0, 0, 0, 1, 0, 0, 0, 1});
    CHECK(x.matmul(id) == x);

    CHECK_THROWS(x.matmul(x), std::invalid_argument);                 // 3 != 2
    CHECK_THROWS(Tensord({3}, {1, 2, 3}).matmul(y), std::invalid_argument);   // not 2D
}

void test_transpose_reshape()
{
    Tensord x({2, 3}, {1, 2, 3, 4, 5, 6});
    Tensord t = x.transpose();
    CHECK(t == Tensord({3, 2}, {1, 4, 2, 5, 3, 6}));
    CHECK(t.transpose() == x);
    CHECK_THROWS(Tensord({3}, {1, 2, 3}).transpose(), std::invalid_argument);

    CHECK(x.reshape({3, 2}) == Tensord({3, 2}, {1, 2, 3, 4, 5, 6}));
    CHECK(x.reshape({6}).rank() == 1);
    CHECK_THROWS(x.reshape({4, 2}), std::invalid_argument);
}

void test_reductions()
{
    Tensord x({2, 3}, {1, 2, 3, 4, 5, 6});
    CHECK(x.sum() == 21.0);
    CHECK(x.mean() == 3.5);
    CHECK(Tensord({0}).mean() == 0.0);   // no dividing by zero on empty tensors

    Tensord u({3}, {1, 2, 3});
    Tensord v({3}, {4, 5, 6});
    CHECK(u.dot(v) == 32.0);
    CHECK_THROWS(u.dot(Tensord({2}, {1, 2})), std::invalid_argument);
    CHECK_THROWS(x.dot(x), std::invalid_argument);
}

void test_int_tensors()
{
    Tensori a({2}, {7, 8});
    CHECK(a / Tensori({2}, {2, 2}) == Tensori({2}, {3, 4}));      // integer division
    CHECK_THROWS(a / Tensori({2}, {2, 0}), std::domain_error);    // would crash otherwise
    CHECK_THROWS(a / Tensori({1}, {0}), std::domain_error);
        CHECK_THROWS(a / Tensori({1}, {0}), std::domain_error);
    CHECK_THROWS(Tensori({1}, {std::numeric_limits<int>::min()}) / Tensori({1}, {-1}),
                 std::overflow_error); 
    
  Tensori b = a;
    CHECK_THROWS(b /= Tensori({2}, {0, 1}), std::domain_error);
    CHECK(b == a);   // failed /= leaves it untouched

    // mean stays a double so it doesn't get rounded
    CHECK(Tensori({2}, {1, 2}).mean() == 1.5);
    CHECK(Tensori({2, 2}, {1, 2, 3, 4}).matmul(Tensori({2, 2}, {5, 6, 7, 8})) == Tensori({2, 2}, {19, 22, 43, 50}));

    Tensorf f({2}, {0.5f, 1.5f});
    CHECK(f.sum() == 2.0f);
    CHECK(f.mean() == 1.0);
    CHECK(f.dot(f) == 2.5f);
}

void test_cast()
{
    Tensord d({2}, {1.9, -2.9});
    Tensori i = d.cast<int>();
    CHECK(i == Tensori({2}, {1, -2}));   // truncates like static_cast
    CHECK(i.cast<double>() == Tensord({2}, {1, -2}));
    CHECK(Tensord({2, 2}).cast<float>().shape == std::vector<size_t>({2, 2}));
}

void test_printing()
{
    CHECK(Tensord({2, 2}, {1, 2, 3, 4}).to_string() == "[[1.0000, 2.0000],\n [3.0000, 4.0000]]");
    CHECK(Tensori({3}, {1, 2, 3}).to_string() == "[1, 2, 3]");
    CHECK(Tensori({2, 2, 2}, {1, 2, 3, 4, 5, 6, 7, 8}).to_string()
          == "[[[1, 2],\n  [3, 4]],\n [[5, 6],\n  [7, 8]]]");

    // 0 sized dimensions used to divide by zero while printing
    CHECK(Tensord({0, 3}).to_string() == "[]");
    CHECK(Tensord({0}).to_string() == "[]");
    CHECK(Tensord({2, 0}).to_string() == "[[],\n []]");
    CHECK(Tensord().to_string() == "[]");

    // rank 0 (just a single number)
    CHECK(Tensori({}, {5}).to_string() == "5");

    // chars should print as numbers, not letters
    CHECK(Tensor<signed char>({2}, {65, 66}).to_string() == "[65, 66]");

    Tensori t({2}, {1, 2});
    std::ostringstream oss;
    oss << t;
    CHECK(oss.str() == t.to_string());
}

int main()
{
    test_construction();
    test_indexing();
    test_elementwise();
    test_scalar_broadcast();
    test_compound_assignment();
    test_scale();
    test_matmul();
    test_transpose_reshape();
    test_reductions();
    test_int_tensors();
    test_cast();
    test_printing();

    if (failures == 0) {
        std::cout << "all tests passed\n";
        return 0;
    }
    std::cout << failures << " check(s) failed\n";
    return 1;
}