#include <iostream>
#include <fstream>
#include <vector>
#include <complex>
#include <chrono>
#include <cmath>
#include <algorithm>

typedef std::complex<double> Complex;
typedef std::vector<Complex> Sequence;

class Convolution {
public:
    static Sequence linear_convolution(const Sequence& x, const Sequence& y) {
        size_t M = x.size();
        size_t L = y.size();
        size_t Lu = M + L - 1;
        
        Sequence u(Lu, Complex(0.0, 0.0));
        
        for (size_t n = 0; n < Lu; ++n) {
            Complex sum(0.0, 0.0);
            for (size_t k = 0; k < M; ++k) {
                if (n >= k && (n - k) < L) {
                    sum += x[k] * y[n - k];
                }
            }
            u[n] = sum;
        }
        return u;
    }
};


class FFTConvolution {
public:
    static size_t next_power_of_two(size_t n) {
        size_t power = 1;
        while (power < n) {
            power <<= 1;
        }
        return power;
    }
    
    static Sequence fft(const Sequence& x) {
        size_t N = x.size();
        if (N <= 1) return x;
        
        Sequence even(N/2);
        Sequence odd(N/2);
        
        for (size_t i = 0; i < N/2; ++i) {
            even[i] = x[2*i];
            odd[i] = x[2*i + 1];
        }
        
        Sequence even_fft = fft(even);
        Sequence odd_fft = fft(odd);
        
        Sequence result(N);
        for (size_t k = 0; k < N/2; ++k) {
            double angle = -2.0 * M_PI * k / N;
            Complex twiddle = Complex(cos(angle), sin(angle));
            result[k] = even_fft[k] + twiddle * odd_fft[k];
            result[k + N/2] = even_fft[k] - twiddle * odd_fft[k];
        }
        
        return result;
    }
    
    static Sequence ifft(const Sequence& x) {
        size_t N = x.size();
        if (N <= 1) return x;
        
        Sequence even(N/2);
        Sequence odd(N/2);
        
        for (size_t i = 0; i < N/2; ++i) {
            even[i] = x[2*i];
            odd[i] = x[2*i + 1];
        }
        
        Sequence even_ifft = ifft(even);
        Sequence odd_ifft = ifft(odd);
        
        Sequence result(N);
        for (size_t k = 0; k < N/2; ++k) {
            double angle = 2.0 * M_PI * k / N;
            Complex twiddle = Complex(cos(angle), sin(angle));
            result[k] = even_ifft[k] + twiddle * odd_ifft[k];
            result[k + N/2] = even_ifft[k] - twiddle * odd_ifft[k];
        }
        
        return result;
    }

    static Sequence scale_ifft(const Sequence& x) {
        size_t N = x.size();
        Sequence result = ifft(x);
        for (size_t i = 0; i < N; ++i) {
            result[i] /= double(N);
        }
        return result;
    }

    static Sequence fft_convolution(const Sequence& x, const Sequence& y) {
        size_t M = x.size();
        size_t L = y.size();
        size_t Lu = M + L - 1;
        
        size_t N = next_power_of_two(Lu);
        
        Sequence x_padded = x;
        x_padded.resize(N, Complex(0.0, 0.0));
        
        Sequence y_padded = y;
        y_padded.resize(N, Complex(0.0, 0.0));

        Sequence X = fft(x_padded);
        Sequence Y = fft(y_padded);

        Sequence U_hat(N);
        for (size_t i = 0; i < N; ++i) {
            U_hat[i] = X[i] * Y[i];
        }

        Sequence U_padded = scale_ifft(U_hat);
        
        Sequence result(Lu);
        for (size_t i = 0; i < Lu; ++i) {
            result[i] = U_padded[i];
        }
        
        return result;
    }
};

// Загрузка сигнала
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

// Норма разности
double calculate_norm(const Sequence& a, const Sequence& b) {
    if (a.size() != b.size()) {
        return -1.0;
    }
    
    double norm = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        double diff_real = a[i].real() - b[i].real();
        double diff_imag = a[i].imag() - b[i].imag();
        norm += diff_real * diff_real + diff_imag * diff_imag;
    }
    return std::sqrt(norm);
}

void save_to_file(const Sequence& data, const std::string& filename) {
    std::ofstream file(filename, std::ios::binary);
    std::vector<double> buffer(2 * data.size());
    
    for (size_t i = 0; i < data.size(); ++i) {
        buffer[2*i] = data[i].real();
        buffer[2*i+1] = data[i].imag();
    }
    
    file.write(reinterpret_cast<const char*>(buffer.data()), buffer.size() * sizeof(double));
    file.close();
}

int main() {
    try {
        Sequence x = load_signal("performance_signals/переменный_1024.bin");
        Sequence y = load_signal("performance_signals/переменный_1024.bin");
        
        Sequence u_direct = Convolution::linear_convolution(x, y);
        Sequence u_fft = FFTConvolution::fft_convolution(x, y);
        double error_direct_vs_fft = calculate_norm(u_direct, u_fft);
        std::cout << "Норма разности (прямая и БПФ): " << error_direct_vs_fft << std::endl;
        
        save_to_file(u_direct, "convolution_direct.bin");
        save_to_file(u_fft, "convolution_fft.bin");
        save_to_file(x, "signal_x1.bin");
        save_to_file(y, "signal_y1.bin");
        
        std::cout << "Прямая свертка: ";
        for (int i = 0; i < 3 && i < u_direct.size(); ++i) {
            std::cout << u_direct[i] << " ";
        }
        std::cout << std::endl;
        
        std::cout << "БПФ-свертка:    ";
        for (int i = 0; i < 3 && i < u_fft.size(); ++i) {
            std::cout << u_fft[i] << " ";
        }
  
        if (error_direct_vs_fft < 1e-10) {
            std::cout << "\nнорм" << std::endl;
        } else {
            std::cout << "\nошибка" << error_direct_vs_fft << std::endl;
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Ошибка: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}