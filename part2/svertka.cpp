#include "svertka.hpp"
#include <iostream>
#include <fstream>
#include <chrono>

Sequence load_signal(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + filename);
    }
    size_t size_bytes = file.tellg();
    file.seekg(0, std::ios::beg);
    
    size_t num_doubles = size_bytes / sizeof(double);
    size_t N = num_doubles / 2;
    
    std::vector<double> buffer(num_doubles);
    file.read(reinterpret_cast<char*>(buffer.data()), size_bytes);
    file.close();
    
    Sequence signal(N);
    for (size_t i = 0; i < N; ++i) {
        signal[i] = Complex(buffer[2 * i], buffer[2 * i + 1]);
    }
    
    return signal;
}

int main() {
    try {
        Sequence x = load_signal("performance_signals/переменный_1024.bin");
        Sequence y = load_signal("performance_signals/фиксированный_512.bin");
        std::cout << "сигнал x длиной: " << x.size() << std::endl;
        std::cout << "сигнал y длиной: " << y.size() << std::endl;
        Sequence u = Convolution::linear_convolution(x, y);
        
        std::cout << "Результат свертки длиной: " << u.size() << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Ошибка: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}