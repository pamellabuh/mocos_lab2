#include <iostream>
#include <fstream>
#include <vector>
#include <complex>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <iomanip>

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

// БПФ-свертка
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
std::pair<double, double> measure_convolution_time(const Sequence& x, const Sequence& y, int iterations = 3) {
    double time_direct = 0.0;
    double time_fft = 0.0;
    for (int i = 0; i < iterations; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        Sequence result = Convolution::linear_convolution(x, y);
        auto end = std::chrono::high_resolution_clock::now();
        time_direct += std::chrono::duration<double, std::milli>(end - start).count();
    }
    time_direct /= iterations;

    for (int i = 0; i < iterations; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        Sequence result = FFTConvolution::fft_convolution(x, y);
        auto end = std::chrono::high_resolution_clock::now();
        time_fft += std::chrono::duration<double, std::milli>(end - start).count();
    }
    time_fft /= iterations;
    
    return std::make_pair(time_direct, time_fft);
}

int main() {
    const std::vector<int> SIZES = {64, 128, 256, 512, 1024, 2048, 4096};
    const int ITERATIONS = 3;

    Sequence fixed_signal;
    try {
        fixed_signal = load_signal("performance_signals/фиксированный_512.bin");
        std::cout << "Загружен фиксированный сигнал длиной: " << fixed_signal.size() << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Не удалось загрузить фиксированный сигнал: " << e.what() << std::endl;
        return 1;
    }
    std::cout << std::setw(8) << "L" << std::setw(15) << "Direct (ms)" 
              << std::setw(15) << "FFT (ms)" << std::setw(12) << "Speedup" 
              << std::setw(12) << "Lu" << std::setw(12) << "N_padded" << std::endl;
    std::cout << std::string(70, '-') << std::endl;
    
    std::vector<int> case1_sizes;
    std::vector<double> case1_direct, case1_fft;
    
    for (int size : SIZES) {
        try {
            std::string filename = "performance_signals/переменный_" + std::to_string(size) + ".bin";
            Sequence var_signal = load_signal(filename);
            
            auto times = measure_convolution_time(fixed_signal, var_signal, ITERATIONS);
            
            size_t Lu = fixed_signal.size() + var_signal.size() - 1;
            size_t N_padded = FFTConvolution::next_power_of_two(Lu);
            double speedup = times.first / times.second;
            
            std::cout << std::setw(8) << size 
                      << std::setw(15) << std::fixed << std::setprecision(3) << times.first
                      << std::setw(15) << std::fixed << std::setprecision(3) << times.second
                      << std::setw(12) << std::fixed << std::setprecision(1) << speedup << "x"
                      << std::setw(12) << Lu
                      << std::setw(12) << N_padded << std::endl;
            
            case1_sizes.push_back(size);
            case1_direct.push_back(times.first);
            case1_fft.push_back(times.second);
            
        } catch (const std::exception& e) {
            std::cerr << "Ошибка при размере " << size << ": " << e.what() << std::endl;
        }
    }
    
    std::cout << std::setw(8) << "N" << std::setw(15) << "Direct (ms)" 
              << std::setw(15) << "FFT (ms)" << std::setw(12) << "Speedup" 
              << std::setw(12) << "Lu" << std::setw(12) << "N_padded" << std::endl;
    std::cout << std::string(70, '-') << std::endl;
    
    std::vector<int> case2_sizes;
    std::vector<double> case2_direct, case2_fft;
    
    for (int size : SIZES) {
        try {
            std::string filename = "performance_signals/переменный_" + std::to_string(size) + ".bin";
            Sequence signal1 = load_signal(filename);
            Sequence signal2 = load_signal(filename);  // Используем тот же сигнал
            
            auto times = measure_convolution_time(signal1, signal2, ITERATIONS);
            
            size_t Lu = signal1.size() + signal2.size() - 1;
            size_t N_padded = FFTConvolution::next_power_of_two(Lu);
            double speedup = times.first / times.second;
            
            std::cout << std::setw(8) << size 
                      << std::setw(15) << std::fixed << std::setprecision(3) << times.first
                      << std::setw(15) << std::fixed << std::setprecision(3) << times.second
                      << std::setw(12) << std::fixed << std::setprecision(1) << speedup << "x"
                      << std::setw(12) << Lu
                      << std::setw(12) << N_padded << std::endl;
            
            case2_sizes.push_back(size);
            case2_direct.push_back(times.first);
            case2_fft.push_back(times.second);
            
        } catch (const std::exception& e) {
            std::cerr << "Ошибка при размере " << size << ": " << e.what() << std::endl;
        }
    }

    std::ofstream file("convolution_performance.csv");
    file << "Case,Size,Direct_Time,FFT_Time\n";

    for (size_t i = 0; i < case1_sizes.size(); ++i) {
        file << "Fixed_M_512," << case1_sizes[i] << "," << case1_direct[i] << "," << case1_fft[i] << "\n";
    }
 
    for (size_t i = 0; i < case2_sizes.size(); ++i) {
        file << "Variable_Both," << case2_sizes[i] << "," << case2_direct[i] << "," << case2_fft[i] << "\n";
    }
    
    file.close();
    
    return 0;
}