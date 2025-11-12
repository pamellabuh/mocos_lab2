#include <iostream>
#include <fstream>
#include <vector>
#include <complex>
#include <chrono>
#include <cmath>
#include <algorithm>

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
        
        std::cout << "Lu = " << Lu << std::endl;
        std::cout <<"N = " << N << std::endl;

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
        // Загружаем два сигнала по 1024 точки (2^10)
        Sequence x = load_signal("performance_signals/переменный_1024.bin");
        Sequence y = load_signal("performance_signals/переменный_1024.bin");
        
        std::cout << "Загружены сигналы по " << x.size() << " точек" << std::endl;

        auto start1 = std::chrono::high_resolution_clock::now();
        Sequence u_direct = Convolution::linear_convolution(x, y);
        auto end1 = std::chrono::high_resolution_clock::now();
        auto start2 = std::chrono::high_resolution_clock::now();
        Sequence u_fft = FFTConvolution::fft_convolution(x, y);
        auto end2 = std::chrono::high_resolution_clock::now();
        auto time_direct = std::chrono::duration_cast<std::chrono::milliseconds>(end1 - start1);
        auto time_fft = std::chrono::duration_cast<std::chrono::milliseconds>(end2 - start2);
        double error_direct_vs_fft = calculate_norm(u_direct, u_fft);
        
        std::cout << "Время прямой свертки: " << time_direct.count() << " мс" << std::endl;
        std::cout << "Время БПФ-свертки: " << time_fft.count() << " мс" << std::endl;
        std::cout << "Норма разности (прямая и БПФ): " << error_direct_vs_fft << std::endl;

        save_to_file(u_direct, "convolution_direct.bin");
        save_to_file(u_fft, "convolution_fft.bin");
        save_to_file(x, "signal_x.bin");
        save_to_file(y, "signal_y.bin");      
    } catch (const std::exception& e) {
        std::cerr << "Ошибка: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}