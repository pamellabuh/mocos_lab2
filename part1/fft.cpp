#include <iostream>
#include <fstream>
#include <vector>
#include <complex>
#include <cmath>
using namespace std;

typedef complex<double> Complex;
const double PI = 3.14;
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

// Обратное БПФ через комплексное сопряжение 
vector<Complex> ifft_via_fft(const vector<Complex>& A) {
    int N = A.size();
    
    // (комплексное сопряжение входного вектора)
    vector<Complex> U(N);
    for (int i = 0; i < N; i++) {
        U[i] = conj(A[i]);
    }
    
    // (прямое БПФ от U)
    vector<Complex> V = fft_dif(U);
    
    //(комплексное сопряжение результата)
    vector<Complex> B(N);
    for (int i = 0; i < N; i++) {
        B[i] = conj(V[i]);
    }
    
    // Масштабирование
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
    cout << "БПФ с прореживанием по частоте" << endl;
    vector<Complex> input = readBinaryFile("performance_signals/переменный_64.bin");    
    cout  << input.size() << " точек сигнала" << endl;
    vector<Complex> fft_result = fft_dif(input);
    writeBinaryFile(fft_result, "результат_БПФ.bin");
    vector<Complex> ifft_result = ifft_via_fft(fft_result);
    writeBinaryFile(ifft_result, "результат_ОБПФ.bin");
    double max_error = 0.0;
    for (size_t i = 0; i < input.size(); i++) {
        double error = abs(input[i] - ifft_result[i]);
        if (error > max_error) max_error = error;
    }
    cout << "Максимальная ошибка восстановления: " << max_error << endl;
    return 0;
}