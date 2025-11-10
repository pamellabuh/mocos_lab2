#include <iostream>
#include <fstream>
#include <vector>
#include <complex>
#include <cmath>

using namespace std;

typedef complex<double> Complex;
const double PI = 3.14159265358979323846;

// Вычисление поворотного коэффициента
Complex twiddle(int j, int k, int n) {
    double angle = -2.0 * PI * j / (1 << (n + 1 - k));
    return Complex(cos(angle), sin(angle));
}

// Прямое БПФ с прореживанием по частоте (по формулам из изображения)
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

// Обратное БПФ через комплексное сопряжение (по методике из изображения)
vector<Complex> ifft_via_fft(const vector<Complex>& A) {
    int N = A.size();
    
    // Шаг 1: U ← A̅ (комплексное сопряжение входного вектора)
    vector<Complex> U(N);
    for (int i = 0; i < N; i++) {
        U[i] = conj(A[i]);
    }
    
    // Шаг 2: V = 𝕎U (прямое БПФ от U)
    vector<Complex> V = fft_dif(U);
    
    // Шаг 3: B ← V̅ (комплексное сопряжение результата)
    vector<Complex> B(N);
    for (int i = 0; i < N; i++) {
        B[i] = conj(V[i]);
    }
    
    // Масштабирование (деление на N)
    for (int i = 0; i < N; i++) {
        B[i] /= double(N);
    }
    
    return B;
}

// Чтение бинарного файла
vector<Complex> readBinaryFile(const string& filename) {
    ifstream file(filename, ios::binary | ios::ate);
    if (!file.is_open()) {
        cerr << "Ошибка: не могу открыть файл " << filename << endl;
        return vector<Complex>();
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

// Запись бинарного файла
void writeBinaryFile(const vector<Complex>& data, const string& filename) {
    ofstream file(filename, ios::binary);
    vector<double> buffer(2 * data.size());
    
    for (size_t i = 0; i < data.size(); i++) {
        buffer[2*i] = data[i].real();
        buffer[2*i+1] = data[i].imag();
    }
    
    file.write((char*)buffer.data(), buffer.size() * sizeof(double));
    file.close();
}

int main() {
    cout << "=== БПФ с прореживанием по частоте ===" << endl;
    
    vector<Complex> input = readBinaryFile("performance_signals/переменный_64.bin");
    if (input.empty()) {
        cerr << "Ошибка чтения файла!" << endl;
        return 1;
    }
    
    cout << "Прочитано " << input.size() << " точек сигнала" << endl;
    
    // Проверяем, что размер является степенью двойки
    int N = input.size();
    if ((N & (N - 1)) != 0) {
        cerr << "Ошибка: размер сигнала должен быть степенью двойки!" << endl;
        return 1;
    }
    
    // Прямое БПФ
    vector<Complex> fft_result = fft_dif(input);
    writeBinaryFile(fft_result, "результат_БПФ.bin");
    cout << "Прямое БПФ завершено" << endl;
    
    // Обратное БПФ через комплексное сопряжение
    vector<Complex> ifft_result = ifft_via_fft(fft_result);
    writeBinaryFile(ifft_result, "результат_ОБПФ.bin");
    cout << "Обратное БПФ завершено" << endl;
    
    // Проверка точности
    double max_error = 0.0;
    for (size_t i = 0; i < input.size(); i++) {
        double error = abs(input[i] - ifft_result[i]);
        if (error > max_error) max_error = error;
    }
    
    cout << "Максимальная ошибка восстановления: " << max_error << endl;
    
    return 0;
}