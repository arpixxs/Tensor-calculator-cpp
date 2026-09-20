#include "Tensor.hpp"
#include <iostream>
#include <map>
#include <sstream>
#include <string>

// the calculator only works on doubles, the Tensor class itself is generic
using Store = std::map<std::string, Tensord>;

// thrown when stdin runs out (ctrl+d, or piped input ending) so we don't loop forever
struct InputClosed {};

// everything reads a whole line at a time. mixing cin >> with getline is what
// broke "blank to skip" before (>> just skips the blank line and keeps waiting)
std::string read_line(const std::string& prompt)
{
    std::cout << prompt;
    std::string line;
    if (!std::getline(std::cin, line)) throw InputClosed();
    return line;
}

// reads a single number, complains about junk like "3abc" or "1 2"
template <typename V>
V read_value(const std::string& prompt)
{
    std::istringstream in(read_line(prompt));
    V v{};
    std::string junk;
    if (!(in >> v) || (in >> junk))
        throw std::invalid_argument("that isn't a valid number");
    return v;
}

size_t read_size(const std::string& prompt)
{
    long long v = read_value<long long>(prompt);
    if (v < 0) throw std::invalid_argument("expected a number that's 0 or more");
    return (size_t)v;
}

// one word from a line. empty line gives "", more than one word is an error
std::string one_word(const std::string& line)
{
    std::istringstream in(line);
    std::string word, extra;
    in >> word;
    if (in >> extra) throw std::invalid_argument("names can't have spaces in them");
    return word;
}

std::string read_name(const std::string& prompt)
{
    std::string name = one_word(read_line(prompt));
    if (name.empty()) throw std::invalid_argument("name can't be empty");
    return name;
}

// nicer error than the "map::at" you get from store.at()
const Tensord& find_tensor(const Store& store, const std::string& name)
{
    auto it = store.find(name);
    if (it == store.end())
        throw std::out_of_range("no tensor named '" + name + "'");
    return it->second;
}

// asks for a name and looks it up right away, so a typo fails before we ask for anything else.
// (prompt is a const char* on purpose - passing a std::string temporary into something that
// returns a reference makes gcc's -Wdangling-reference throw a false alarm)
const Tensord& ask_tensor(const Store& store, const char* prompt)
{
    std::string name = read_name(prompt);
    return find_tensor(store, name);
}

std::vector<size_t> read_shape(const std::string& rank_prompt)
{
    size_t rank = read_size(rank_prompt);
    std::vector<size_t> shape(rank);
    for (size_t i = 0; i < rank; ++i)
        shape[i] = read_size("  dim[" + std::to_string(i) + "] size: ");
    return shape;
}

// asks the user for a shape + values and builds a tensor out of it
Tensord read_tensor(const std::string& label)
{
    std::vector<size_t> shape = read_shape("Enter rank (number of dimensions) for " + label + ": ");

    size_t n = Tensord::numel(shape);
    std::cout << "Enter " << n << " values for " << label << " (row-major order):\n";

    // values can be spread over as many lines as you like
    std::vector<double> values;
    while (values.size() < n) {
        std::istringstream in(read_line(""));
        double v;
        while (values.size() < n && in >> v) values.push_back(v);
        if (in.fail() && !in.eof())
            throw std::invalid_argument("that isn't a valid number");
        std::string extra;
        if (values.size() == n && (in >> extra))
            throw std::invalid_argument("got more values than the shape needs");
    }

    return Tensord(shape, values);
}

// shows a result and lets the user keep it. blank line = don't save
void show_result(Store& store, const Tensord& result)
{
    std::cout << "Result:\n";
    result.print();

    std::string out = one_word(read_line("Save result as (blank to skip): "));
    if (out.empty()) return;
    store[out] = result;
    std::cout << "Saved as " << out << ".\n";
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
              << " 0. Exit\n";
}

int main()
{
    Store store;
    std::cout << "Tensor Calculator (C++). Tensors are stored by name so you can reuse them.\n";

    while (true) {
        print_menu();

        try {
            int choice = read_value<int>("Choice: ");

            if (choice == 0) {
                break;
            }
            else if (choice == 1) {
                std::string name = read_name("Name for this tensor: ");
                Tensord t = read_tensor(name);
                bool existed = store.count(name) > 0;
                store[name] = std::move(t);
                std::cout << name << (existed ? " updated.\n" : " stored.\n");
            }
            else if (choice == 2) {
                ask_tensor(store, "Name: ").print();
            }
            else if (choice >= 3 && choice <= 6) {
                const Tensord& A = ask_tensor(store, "Name of A: ");
                const Tensord& B = ask_tensor(store, "Name of B: ");

                Tensord result;
                switch (choice) {
                    case 3: result = A + B; break;
                    case 4: result = A - B; break;
                    case 5: result = A * B; break;
                    case 6: result = A / B; break;
                }
                show_result(store, result);
            }
            else if (choice == 7) {
                const Tensord& A = ask_tensor(store, "Name of A: ");
                const Tensord& B = ask_tensor(store, "Name of B: ");
                show_result(store, A.matmul(B));
            }
            else if (choice == 8) {
                const Tensord& A = ask_tensor(store, "Name: ");
                show_result(store, A.transpose());
            }
            else if (choice == 9) {
                const Tensord& A = ask_tensor(store, "Name: ");
                std::vector<size_t> shape = read_shape("New rank: ");
                show_result(store, A.reshape(shape));
            }
            else if (choice == 10) {
                const Tensord& A = ask_tensor(store, "Name: ");
                double k = read_value<double>("Scalar: ");
                show_result(store, A.scale(k));   // this one didn't offer to save before, now it does
            }
            else if (choice == 11) {
                const Tensord& A = ask_tensor(store, "Name: ");
                std::cout << "Sum = " << A.sum() << ", Mean = " << A.mean() << "\n";
            }
            else if (choice == 12) {
                const Tensord& A = ask_tensor(store, "Name of A: ");
                const Tensord& B = ask_tensor(store, "Name of B: ");
                std::cout << "Dot product = " << A.dot(B) << "\n";
            }
            else if (choice == 13) {
                if (store.empty()) {
                    std::cout << "(nothing stored yet)\n";
                } else {
                    std::cout << "Stored tensors:\n";
                    for (auto& kv : store) {
                        std::cout << "  " << kv.first << "  shape=(";
                        for (size_t i = 0; i < kv.second.shape.size(); ++i)
                            std::cout << kv.second.shape[i] << (i + 1 < kv.second.shape.size() ? "," : "");
                        std::cout << ")\n";
                    }
                }
            }
            else {
                std::cout << "not a valid option\n";
            }
        } catch (const InputClosed&) {
            std::cout << "\n";
            break;
        } catch (const std::exception& e) {
            std::cout << "Error: " << e.what() << "\n";
        }
    }

    std::cout << "Goodbye.\n";
    return 0;
}