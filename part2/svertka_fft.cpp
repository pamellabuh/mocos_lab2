#include <iostream>
#include <fstream>
#include <vector>
#include <complex>
#include <cmath>
#include <chrono>
#include <algorithm>

typedef std::complex<double> Complex;
typedef std::vector<Complex> Sequence;

Sequence fft(const Sequence& x) {
    size_t N = x.size();
    if (N == 1) return x;
    
    Sequence first_half(N/2);
    Sequence second_half(N/2);
    
    for (size_t k = 0; k < N/2; ++k) {
        first_half[k] = x[k] + x[k + N/2];
        Complex temp = x[k] - x[k + N/2];
        double angle = -2.0 * M_PI * k / N;
        second_half[k] = temp * Complex(std::cos(angle), std::sin(angle));
    }
    
    Sequence first_fft = fft(first_half);
    Sequence second_fft = fft(second_half);
    
    Sequence result(N);
    for (size_t k = 0; k < N/2; ++k) {
        result[k] = first_fft[k];
        result[k + N/2] = second_fft[k];
    }
    
    return result;
}

Sequence ifft(const Sequence& x) {
    size_t N = x.size();
    if (N == 1) return x;
    
    Sequence first_half(N/2);
    Sequence second_half(N/2);
    
    for (size_t k = 0; k < N/2; ++k) {
        first_half[k] = x[k];
        second_half[k] = x[k + N/2];
    }
    
    Sequence first_ifft = ifft(first_half);
    Sequence second_ifft = ifft(second_half);
    
    Sequence result(N);
    for (size_t k = 0; k < N/2; ++k) {
        double angle = 2.0 * M_PI * k / N;
        Complex twiddle = Complex(std::cos(angle), std::sin(angle));
        result[k] = first_ifft[k] + twiddle * second_ifft[k];
        result[k + N/2] = first_ifft[k] - twiddle * second_ifft[k];
    }
    
    return result;
}

size_t next_power_of_two(size_t n) {
    size_t power = 1;
    while (power < n) {
        power <<= 1;
    }
    return power;
}

Sequence fft_convolution(const Sequence& x, const Sequence& y) {
    size_t M = x.size();
    size_t L = y.size();
    size_t Lu = M + L - 1;
    

    size_t N_required = std::max(2 * L, 2 * M);
    size_t N = next_power_of_two(N_required);

    Sequence X = x;
    X.resize(N, Complex(0.0, 0.0));
    Sequence Y = y;
    Y.resize(N, Complex(0.0, 0.0));
    
    Sequence X_hat = fft(X);
    Sequence Y_hat = fft(Y);
    
    Sequence U_hat(N);
    double scale = std::sqrt(2.0 * N);
    for (size_t i = 0; i < N; ++i) {
        U_hat[i] = scale * X_hat[i] * Y_hat[i];
    }

    Sequence U = ifft(U_hat);

    Sequence result(Lu);
    for (size_t i = 0; i < Lu; ++i) {
        result[i] = U[i];
    }
    
    return result;
}


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
        
        std::cout << "x длиной: " << x.size() << std::endl;
        std::cout << "y длиной: " << y.size() << std::endl;

        Sequence u = fft_convolution(x, y);
        std::cout << "Результат свертки длиной: " << u.size() << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Ошибка: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}