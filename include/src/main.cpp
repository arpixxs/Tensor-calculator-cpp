#include "Tensor.hpp"
#include <iostream>
#include <map>
#include <sstream>
#include <string>

// Reads a tensor from stdin: asks for shape, then values.
Tensor read_tensor(const std::string& label) 
{
    std::cout << "Enter rank (number of dimensions) for " << label << ": ";
    size_t rank;
    std::cin >> rank;

    std::vector<size_t> shape(rank);
    for (size_t i = 0; i < rank; ++i) {
        std::cout << "  dim[" << i << "] size: ";
        std::cin >> shape[i];
    }

    size_t n = Tensor::numel(shape);
    std::vector<double> values(n);
    std::cout << "Enter " << n << " values for " << label << " (row-major order):\n";
    for (size_t i = 0; i < n; ++i) std::cin >> values[i];

    return Tensor(shape, values);
}

void print_menu() 
{
    std::cout << "\n===== Tensor Calculator =====\n"
              << " 1. Create/store a tensor\n"
              << " 2. Print a stored tensor\n"
              << " 3. Add (A + B)\n"
              << " 4. Subtract (A - B)\n"
              << " 5. Element-wise multiply (A * B)\n"
              << " 6. Element-wise divide (A / B)\n"
              << " 7. Matrix multiply (2D only)\n"
              << " 8. Transpose (2D only)\n"
              << " 9. Reshape\n"
              << "10. Scale by scalar\n"
              << "11. Sum / Mean\n"
              << "12. Dot product (1D only)\n"
              << "13. List stored tensors\n"
              << " 0. Exit\n"
              << "Choice: ";
}

int main()
 {
    std::map<std::string, Tensor> store;
    std::cout << "Tensor Calculator (C++). Tensors are stored by name for reuse.\n";

    while (true) {
        print_menu();
        int choice;
        if (!(std::cin >> choice)) break;

        try {
            if (choice == 0) {
                break;
            } else if (choice == 1) {
                std::cout << "Name for this tensor: ";
                std::string name; std::cin >> name;
                store[name] = read_tensor(name);
                std::cout << name << " stored.\n";
            } else if (choice == 2) {
                std::cout << "Name: ";
                std::string name; std::cin >> name;
                store.at(name).print();
            } else if (choice >= 3 && choice <= 6) {
                std::cout << "Name of A: "; std::string a; std::cin >> a;
                std::cout << "Name of B: "; std::string b; std::cin >> b;
                Tensor result;
                if (choice == 3) result = store.at(a) + store.at(b);
                if (choice == 4) result = store.at(a) - store.at(b);
                if (choice == 5) result = store.at(a) * store.at(b);
                if (choice == 6) result = store.at(a) / store.at(b);
                std::cout << "Result:\n"; result.print();
                std::cout << "Save result as (name, or blank to skip): ";
                std::string out; std::cin >> out;
                if (!out.empty() && out != "-") store[out] = result;
            } else if (choice == 7) {
                std::cout << "Name of A: "; std::string a; std::cin >> a;
                std::cout << "Name of B: "; std::string b; std::cin >> b;
                Tensor result = store.at(a).matmul(store.at(b));
                std::cout << "Result:\n"; result.print();
            } else if (choice == 8) {
                std::cout << "Name: "; std::string a; std::cin >> a;
                Tensor result = store.at(a).transpose();
                std::cout << "Result:\n"; result.print();
            } else if (choice == 9) {
                std::cout << "Name: "; std::string a; std::cin >> a;
                std::cout << "New rank: "; size_t r; std::cin >> r;
                std::vector<size_t> shape(r);
                for (size_t i = 0; i < r; ++i) { std::cout << "  dim[" << i << "]: "; std::cin >> shape[i]; }
                Tensor result = store.at(a).reshape(shape);
                std::cout << "Result:\n"; result.print();
            } else if (choice == 10) {
                std::cout << "Name: "; std::string a; std::cin >> a;
                std::cout << "Scalar: "; double k; std::cin >> k;
                Tensor result = store.at(a).scale(k);
                std::cout << "Result:\n"; result.print();
            } else if (choice == 11) {
                std::cout << "Name: "; std::string a; std::cin >> a;
                std::cout << "Sum = " << store.at(a).sum()
                          << ", Mean = " << store.at(a).mean() << "\n";
            } else if (choice == 12) {
                std::cout << "Name of A: "; std::string a; std::cin >> a;
                std::cout << "Name of B: "; std::string b; std::cin >> b;
                std::cout << "Dot product = " << store.at(a).dot(store.at(b)) << "\n";
            } else if (choice == 13) {
                std::cout << "Stored tensors:\n";
                for (auto& kv : store) {
                    std::cout << "  " << kv.first << "  shape=(";
                    for (size_t i = 0; i < kv.second.shape.size(); ++i)
                        std::cout << kv.second.shape[i] << (i + 1 < kv.second.shape.size() ? "," : "");
                    std::cout << ")\n";
                }
            } else {
                std::cout << "Unknown choice.\n";
            }
        } catch (const std::exception& e) {
            std::cout << "Error: " << e.what() << "\n";
        }
    }

    std::cout << "Goodbye.\n";
    return 0;
}