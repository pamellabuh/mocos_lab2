#include <iostream>
#include <fstream>
#include <vector>
#include <complex>
#include <cmath>

using namespace std;

typedef complex<double> Complex;
const double PI = 3.14159265358979323846;

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

// Масштабирование обратного БПФ 
vector<Complex> scaleIFFT(const vector<Complex>& x) {
    int N = x.size();
    vector<Complex> result = ifft(x);
    
    for (int i = 0; i < N; i++) {
        result[i] /= double(N);
    }
    
    return result;
}

// Чтение бинарного файла
vector<Complex> readBinaryFile(const string& filename) {
    ifstream file(filename, ios::binary | ios::ate);
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
    cout << "=== Быстрое преобразование Фурье (БПФ) ===" << endl;
    
    vector<Complex> input = readBinaryFile("performance_signals/переменный_64.bin");
    cout << input.size() << " точек сигнала" << endl;
    vector<Complex> fft_result = fft(input);
    writeBinaryFile(fft_result, "результат_БПФ.bin");
    vector<Complex> ifft_result = scaleIFFT(fft_result);
    writeBinaryFile(ifft_result, "результат_ОБПФ.bin");
    double max_error = 0.0;
    for (size_t i = 0; i < input.size(); i++) {
        double error = abs(input[i] - ifft_result[i]);
        if (error > max_error) max_error = error;
    }
    cout << "Максимальная ошибка восстановления: " << max_error << endl;
    
    return 0;
}