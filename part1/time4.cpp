#include <iostream>
#include <fstream>
#include <vector>
#include <complex>
#include <cmath>
#include <chrono>
#include <string>

using namespace std;
using namespace std::chrono;

typedef complex<double> Complex;
const double PI = 3.14159265358979323846;

// ==================== ПРАВИЛЬНАЯ РЕАЛИЗАЦИЯ БПФ С ПРОРЕЖИВАНИЕМ ПО ЧАСТОТЕ ====================

// Вычисление поворотного коэффициента
Complex twiddle(int j, int k, int n) {
    double angle = -2.0 * PI * j / (1 << (n + 1 - k));
    return Complex(cos(angle), sin(angle));
}

// Прямое БПФ с прореживанием по частоте (БЕЗ масштабирования)
vector<Complex> fft_dif(const vector<Complex>& x) {
    int N = x.size();
    int n = 0;
    while ((1 << n) < N) n++;
    
    vector<Complex> y = x;
    
    // k-й шаг (k = 1, 2, ..., n)
    for (int k = 1; k <= n; k++) {
        vector<Complex> temp(N);
        
        // Размер блока на текущем шаге
        int block_size = 1 << k;           // 2^k
        int half_block = 1 << (k - 1);     // 2^{k-1}
        int num_blocks = N / block_size;
        
        for (int j = 0; j < num_blocks; j++) {
            for (int l = 0; l < half_block; l++) {
                int idx1 = j * block_size + l;              // j2^k + l
                int idx2 = idx1 + half_block;               // j2^k + l + 2^{k-1}
                
                int src_idx1 = j * half_block + l;          // j2^{k-1} + l
                int src_idx2 = src_idx1 + (N / 2);          // 2^{n-1} + j2^{k-1} + l
                
                // Бабочка согласно формулам:
                // y(j2^k + l) = x(j2^{k-1} + l) + x(2^{n-1} + j2^{k-1} + l)
                temp[idx1] = y[src_idx1] + y[src_idx2];
                
                // y(j2^k + l + 2^{k-1}) = [x(j2^{k-1} + l) - x(2^{n-1} + j2^{k-1} + l)] * ω_{n+1-k}^j
                Complex diff = y[src_idx1] - y[src_idx2];
                temp[idx2] = diff * twiddle(j, k, n);
            }
        }
        
        y = temp;
    }
    
    return y;
}

// БПФ с масштабированием 1/√N (унитарное преобразование)
vector<Complex> scaleFFT(const vector<Complex>& x) {
    int N = x.size();
    vector<Complex> result = fft_dif(x);
    double scale = 1.0 / sqrt(N);
    
    for (int i = 0; i < N; i++) {
        result[i] *= scale;
    }
    return result;
}

// ==================== ДПФ (остается без изменений) ====================

vector<Complex> computeDFT(const vector<Complex>& input) {
    int N = input.size();
    vector<Complex> output(N);
    double scale = 1.0 / sqrt(N);
    
    for (int k = 0; k < N; k++) {
        Complex sum(0.0, 0.0);
        for (int j = 0; j < N; j++) {
            double angle = -2.0 * PI * k * j / N;
            sum += input[j] * Complex(cos(angle), sin(angle));
        }
        output[k] = sum * scale;
    }
    return output;
}

// ==================== ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ====================

vector<Complex> readBinaryFile(const string& filename) {
    ifstream file(filename, ios::binary | ios::ate);
    if (!file.is_open()) {
        throw runtime_error("Не могу открыть файл: " + filename);
    }
    
    streamsize size_bytes = file.tellg();
    file.seekg(0, ios::beg);
    
    int num_doubles = size_bytes / sizeof(double);
    int N = num_doubles / 2;
    
    vector<double> buffer(num_doubles);
    file.read((char*)buffer.data(), size_bytes);
    file.close();
    
    vector<Complex> signal(N);
    for (int i = 0; i < N; i++) {
        signal[i] = Complex(buffer[2*i], buffer[2*i+1]);
    }
    return signal;
}

// Создание тестового сигнала если файл не найден
vector<Complex> createTestSignal(int N) {
    vector<Complex> signal(N);
    for (int i = 0; i < N; i++) {
        double t = 2.0 * PI * i / N;
        // Синусоида с несколькими частотами + шум
        signal[i] = Complex(sin(5 * t) + 0.5 * sin(15 * t) + 0.2 * sin(25 * t), 0);
    }
    return signal;
}

pair<double, double> measurePerformance(int N, int num_runs = 5) {
    vector<Complex> signal;
    
    try {
        string filename = "performance_signals/переменный_" + to_string(N) + ".bin";
        signal = readBinaryFile(filename);
    } catch (const exception& e) {
        // Если файл не найден, создаем тестовый сигнал
        cout << "Файл для N=" << N << " не найден, создаем тестовый сигнал" << endl;
        signal = createTestSignal(N);
    }
    
    double total_dft = 0.0, total_fft = 0.0;
    
    for (int run = 0; run < num_runs; run++) {
        // Измеряем время ДПФ
        auto start_dft = high_resolution_clock::now();
        vector<Complex> dft_result = computeDFT(signal);
        auto end_dft = high_resolution_clock::now();
        total_dft += duration_cast<microseconds>(end_dft - start_dft).count();
        
        // Измеряем время БПФ
        auto start_fft = high_resolution_clock::now();
        vector<Complex> fft_result = scaleFFT(signal);
        auto end_fft = high_resolution_clock::now();
        total_fft += duration_cast<microseconds>(end_fft - start_fft).count();
    }
    
    double time_dft = total_dft / (1000.0 * num_runs);  // переводим в миллисекунды
    double time_fft = total_fft / (1000.0 * num_runs);
    
    cout << "N = " << N << ":\tДПФ = " << time_dft << " мс,\tБПФ = " << time_fft << " мс" << endl;
    
    return make_pair(time_dft, time_fft);
}

int main() {
    vector<int> sizes;
    vector<double> times_dft, times_fft;
    
    cout << "Измерение времени выполнения ДПФ и БПФ:" << endl;
    cout << "N\t\tДПФ (мс)\t\tБПФ (мс)" << endl;
    cout << "----------------------------------------" << endl;
    
    // N = 2^n, где n ∈ {6,7,...,12}
    for (int n = 6; n <= 12; n++) {
        int N = pow(2, n);
        sizes.push_back(N);
        
        auto times = measurePerformance(N);
        times_dft.push_back(times.first);
        times_fft.push_back(times.second);
    }
    
    // Сохраняем результаты в CSV файл
    ofstream file("performance_results.csv");
    file << "N,DFT_Time_ms,FFT_Time_ms\n";
    for (size_t i = 0; i < sizes.size(); i++) {
        file << sizes[i] << "," << times_dft[i] << "," << times_fft[i] << "\n";
    }
    file.close();
    
    cout << "\nРезультаты сохранены в performance_results.csv" << endl;
    
    return 0;
}