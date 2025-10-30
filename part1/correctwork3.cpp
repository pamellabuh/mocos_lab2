#include <iostream>
#include <fstream>
#include <vector>
#include <complex>
#include <cmath>

using namespace std;

typedef complex<double> Complex;
const double PI = 3.14159265358979323846;

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
        output[j] = sum * scale;
    }
    return output;
}

// Прямое БПФ
vector<Complex> fft(const vector<Complex>& x) {
    int N = x.size();
    if (N == 1) {
        return x;
    }
    
    vector<Complex> first_half(N/2);
    vector<Complex> second_half(N/2);
    
    for (int k = 0; k < N/2; k++) {
        first_half[k] = x[k] + x[k + N/2];
        Complex temp = x[k] - x[k + N/2];
        double angle = -2.0 * PI * k / N;
        second_half[k] = temp * Complex(cos(angle), sin(angle));
    }
    
    vector<Complex> first_fft = fft(first_half);
    vector<Complex> second_fft = fft(second_half);
    vector<Complex> result(N);
    for (int k = 0; k < N/2; k++) {
        result[k] = first_fft[k];
        result[k + N/2] = second_fft[k];
    }
    
    return result;
}
//обпф
vector<Complex> ifft(const vector<Complex>& x) {
    int N = x.size();
    if (N == 1) {
        return x;
    }
    
    vector<Complex> first_half(N/2);
    vector<Complex> second_half(N/2);
    
    for (int k = 0; k < N/2; k++) {
        first_half[k] = x[k];
        second_half[k] = x[k + N/2];
    }
    vector<Complex> first_ifft = ifft(first_half);
    vector<Complex> second_ifft = ifft(second_half);
    vector<Complex> result(N);
    for (int k = 0; k < N/2; k++) {
        double angle = 2.0 * PI * k / N;  
        Complex twiddle = Complex(cos(angle), sin(angle));
        result[k] = first_ifft[k] + twiddle * second_ifft[k];
        result[k + N/2] = first_ifft[k] - twiddle * second_ifft[k];
    }
    
    return result;
}

// Масштабирование результата БПФ
vector<Complex> scaleFFT(const vector<Complex>& x) {
    int N = x.size();
    vector<Complex> result = fft(x);
    double scale = 1.0 / sqrt(N);
    
    for (int i = 0; i < N; i++) {
        result[i] *= scale;
    }
    return result;
}
vector<Complex> scaleIFFT(const vector<Complex>& x) {
    int N = x.size();
    vector<Complex> result = ifft(x);
    double scale = 1.0 / sqrt(N); 
    
    for (int i = 0; i < N; i++) {
        result[i] *= scale;
    }
    return result;
}

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

double vectorNorm(const vector<Complex>& a, const vector<Complex>& b) {
    double norm = 0.0;
    for (size_t i = 0; i < a.size(); i++) {
        double diff = abs(a[i] - b[i]);
        norm += diff * diff;
    }
    return sqrt(norm);
}

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
        cout << "X = ОДПФ(ДПФ(X)): максимальная ошибка = " << scientific << error_dft << endl; 
        vector<Complex> fft_X = scaleFFT(X);
        vector<Complex> ifft_fft_X = scaleIFFT(fft_X);  
        double error_fft = maxError(X, ifft_fft_X);
        cout << "X = ОБПФ(БПФ(X)): максимальная ошибка = " << scientific << error_fft << endl;
        double norm_diff_dft_fft = vectorNorm(dft_X, fft_X);
        cout << "Норма разности ДПФ(X) и БПФ(X): " << scientific << norm_diff_dft_fft << endl;
        writeBinaryFile(dft_X, "dft_result.bin");
        writeBinaryFile(fft_X, "fft_result.bin");
        writeBinaryFile(X, "original_signal.bin");
        
    } catch (const exception& e) {
        cerr << "Ошибка: " << e.what() << endl;
        return 1;
    }
    
    return 0;
}