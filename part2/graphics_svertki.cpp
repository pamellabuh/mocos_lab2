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

const double PI = 3.14159265358979323846;
class FFT_DIF {
public:
    static Complex wk(int j, int k, int n) {
        double angle = -2.0 * PI * j / pow(2, n + 1 - k);
        return Complex(cos(angle), sin(angle));
    }

    static Sequence fft_dif(const Sequence& x) {
        int N = x.size();
        int n = 0;
        while (pow(2, n) < N) n++;
        
        Sequence y = x;
        
        for (int k = 1; k <= n; k++) {
            Sequence temp(N);
            int block_size = pow(2, k);
            int half_block = pow(2, k - 1);
            int num_blocks = N / block_size;
            
            for (int j = 0; j < num_blocks; j++) {
                for (int l = 0; l < half_block; l++) {
                    int idx1 = j * block_size + l;
                    int idx2 = idx1 + half_block;
                    int src_idx1 = j * half_block + l;
                    int src_idx2 = src_idx1 + (N / 2);
                    
                    temp[idx1] = y[src_idx1] + y[src_idx2];
                    Complex diff = y[src_idx1] - y[src_idx2];
                    temp[idx2] = diff * wk(j, k, n);
                }
            }
            y = temp;
        }
        return y;
    }

    static Sequence ifft_via_fft(const Sequence& A) {
        int N = A.size();
        Sequence U(N);
        for (int i = 0; i < N; i++) {
            U[i] = std::conj(A[i]);
        }
        Sequence V = fft_dif(U);
        Sequence B(N);
        for (int i = 0; i < N; i++) {
            B[i] = std::conj(V[i]);
        }
        for (int i = 0; i < N; i++) {
            B[i] /= double(N);
        }
        return B;
    }
};

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
    
    static Sequence fft_convolution(const Sequence& x, const Sequence& y) {
        size_t M = x.size();
        size_t L = y.size();
        size_t Lu = M + L - 1;
        size_t N_required = std::max(2 * L, 2 * M);
        size_t N = next_power_of_two(N_required);
        
        Sequence x_padded = x;
        x_padded.resize(N, Complex(0.0, 0.0));
        Sequence y_padded = y;
        y_padded.resize(N, Complex(0.0, 0.0));
        Sequence X = FFT_DIF::fft_dif(x_padded);
        Sequence Y = FFT_DIF::fft_dif(y_padded);
        
        Sequence U_hat(N);
        for (size_t i = 0; i < N; ++i) {
            U_hat[i] = X[i] * Y[i];
        }
        
        Sequence U_padded = FFT_DIF::ifft_via_fft(U_hat);
        
        Sequence result(Lu);
        for (size_t i = 0; i < Lu; ++i) {
            result[i] = U_padded[i];
        }
        
        return result;
    }
};

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
    fixed_signal = load_signal("performance_signals/фиксированный_512.bin");
    std::cout << "Загружен фиксированный сигнал длиной: " << fixed_signal.size() << std::endl;
    
    // Случай 1: Фиксированная длина M=512, переменная длина L
    std::cout << "Случай 1: M=512 (фиксированная), L - переменная" << std::endl;
    std::cout << std::setw(8) << "L" << std::setw(15) << "Direct (ms)" 
              << std::setw(15) << "FFT (ms)" << std::endl;
    
    std::vector<int> case1_sizes;
    std::vector<double> case1_direct, case1_fft;
    
    for (int size : SIZES) {
        try {
            std::string filename = "performance_signals/переменный_" + std::to_string(size) + ".bin";
            Sequence var_signal = load_signal(filename);
            
            auto times = measure_convolution_time(fixed_signal, var_signal, ITERATIONS);
            
            std::cout << std::setw(8) << size 
                      << std::setw(15) << std::fixed << std::setprecision(3) << times.first
                      << std::setw(15) << std::fixed << std::setprecision(3) << times.second << std::endl;
            
            case1_sizes.push_back(size);
            case1_direct.push_back(times.first);
            case1_fft.push_back(times.second);
            
        } catch (const std::exception& e) {
            std::cerr << "Ошибка при размере " << size << ": " << e.what() << std::endl;
        }
    }
    
    // Случай 2: Обе длины переменные и одинаковые
    std::cout << "\nСлучай 2: M = L = N (обе переменные и одинаковые)" << std::endl;
    std::cout << std::setw(8) << "N" << std::setw(15) << "Direct (ms)" 
              << std::setw(15) << "FFT (ms)"  << std::endl;
    
    std::vector<int> case2_sizes;
    std::vector<double> case2_direct, case2_fft;
    
    for (int size : SIZES) {
        try {
            std::string filename = "performance_signals/переменный_" + std::to_string(size) + ".bin";
            Sequence signal1 = load_signal(filename);
            Sequence signal2 = load_signal(filename);  
            
            auto times = measure_convolution_time(signal1, signal2, ITERATIONS);
            
            std::cout << std::setw(8) << size 
                      << std::setw(15) << std::fixed << std::setprecision(3) << times.first
                      << std::setw(15) << std::fixed << std::setprecision(3) << times.second << std::endl;
            
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