#include <iostream>
#include <fstream>
#include <vector>
#include <complex>
#include <cmath>
using namespace std;

typedef complex<double> Complex;
const double PI = 3.14159265358979323846; 

// поворотный коэффициент
Complex wk(int j, int k, int n) {
    double angle = -2.0 * PI * j / pow(2, n + 1 - k);
    return Complex(cos(angle), sin(angle));
}

// Прямое БПФ
vector<Complex> fft_dif(const vector<Complex>& x) {
    int N = x.size();
    int n = 0;
    while (pow(2, n) < N) n++;
    
    vector<Complex> y = x;
    
    for (int k = 1; k <= n; k++) {
        vector<Complex> temp(N);
        
        // Размер блока на текущем шаге
        int block_size = pow(2, k);           // 2^k
        int half_block = pow(2, k - 1);       // 2^{k-1}
        int num_blocks = N / block_size;
        
        for (int j = 0; j < num_blocks; j++) {
            for (int l = 0; l < half_block; l++) {
                int idx1 = j * block_size + l;              // j*2^k + l
                int idx2 = idx1 + half_block;               // j*2^k + l + 2^{k-1}
                
                int src_idx1 = j * half_block + l;          // j*2^{k-1} + l
                int src_idx2 = src_idx1 + (N / 2);          // 2^{n-1} + j*2^{k-1} + l
                
                // Бабочка согласно формулам:
                // y(j*2^k + l) = x(j*2^{k-1} + l) + x(2^{n-1} + j*2^{k-1} + l)
                temp[idx1] = y[src_idx1] + y[src_idx2];
                
                // y(j*2^k + l + 2^{k-1}) = [x(j*2^{k-1} + l) - x(2^{n-1} + j*2^{k-1} + l)] * ω_{n+1-k}^j
                Complex diff = y[src_idx1] - y[src_idx2];
                temp[idx2] = diff * wk(j, k, n);
            }
        }
        
        y = temp;
    }
    
    return y;
}

// Обратное БПФ
vector<Complex> ifft_via_fft(const vector<Complex>& A) {
    int N = A.size();
    vector<Complex> U(N);
    for (int i = 0; i < N; i++) {
        U[i] = conj(A[i]);
    }
    vector<Complex> V = fft_dif(U);
    vector<Complex> B(N);
    for (int i = 0; i < N; i++) {
        B[i] = conj(V[i]) / double(N);
    }
    return B;
}

// Прямое ДПФ 
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
        output[k] = sum*scale;
    }
    return output;
}

// Обратное ДПФ
vector<Complex> computeIDFT(const vector<Complex>& input) {
    int N = input.size();
    vector<Complex> output(N);
    double scale = 1.0 / sqrt(N);
    
    for (int j = 0; j < N; j++) {
        Complex sum(0.0, 0.0);
        for (int k = 0; k < N; k++) {
            double angle = 2.0 * PI * k * j / N;
            sum += input[k] * Complex(cos(angle), sin(angle));
        }
        output[j] = sum*scale;
    }
    return output;
}
// Чтение бинарного файла
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

// Норма разности векторов
double vectorNorm(const vector<Complex>& a, const vector<Complex>& b) {
    double norm = 0.0;
    for (size_t i = 0; i < a.size(); i++) {
        double diff = abs(a[i] - b[i]);
        norm += diff * diff;
    }
    return sqrt(norm);
}

// Максимальная ошибка
double maxError(const vector<Complex>& a, const vector<Complex>& b) {
    double max_err = 0.0;
    for (size_t i = 0; i < a.size(); i++) {
        double err = abs(a[i] - b[i]);
        if (err > max_err) max_err = err;
    }
    return max_err;
}

int main() {
    cout << "Размер сигнала: N = 1024" << endl;
    
    try {
        vector<Complex> X = readBinaryFile("performance_signals/переменный_1024.bin");        

        vector<Complex> dft_X = computeDFT(X);
        vector<Complex> idft_dft_X = computeIDFT(dft_X);
        double error_dft = maxError(X, idft_dft_X);
        cout << "X = ОДПФ(ДПФ(X)): макс ошибка = " << scientific << error_dft << endl;
        

        vector<Complex> fft_X = fft_dif(X);
        vector<Complex> ifft_fft_X = ifft_via_fft(fft_X);
        double error_fft = maxError(X, ifft_fft_X);
        cout << "X = ОБПФ(БПФ(X)): макс ошибка = " << scientific << error_fft << endl;
        
        vector<Complex> fft_X_scaled = fft_X;
        double scale = 1.0 / sqrt(X.size());
        for (size_t i = 0; i < fft_X_scaled.size(); i++) {
            fft_X_scaled[i] *= scale;
        }
        double norm_diff_dft_fft = vectorNorm(dft_X, fft_X_scaled);
        cout << "Норма разности ДПФ(X) и БПФ(X): " << scientific << norm_diff_dft_fft << endl;
        
        writeBinaryFile(dft_X, "dft_result.bin");
        writeBinaryFile(fft_X, "fft_result.bin");
        writeBinaryFile(X, "original_signal.bin");
        
    } catch (const exception& e) {
        cerr << "Ошибка: " << e.what() << endl;
    }
    return 0;
}