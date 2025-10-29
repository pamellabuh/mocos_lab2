#include <iostream>
#include <fstream>
#include <vector>
#include <complex>
#include <cmath>

using namespace std;

typedef complex<double> Complex;
const double PI = 3.14159265358979323846;

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
        output[k] = sum * scale;
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
        output[j] = sum * scale;
    }
    return output;
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
    cout << "=== Вычисление ДПФ и ОДПФ ===" << endl;
    
    // Чтение входного сигнала
    vector<Complex> input = readBinaryFile("performance_signals/фиксированный_512.bin");
    cout << "Загружено " << input.size() << " точек сигнала" << endl;
    
    // Прямое ДПФ
    vector<Complex> dft_result = computeDFT(input);
    writeBinaryFile(dft_result, "результат_ДПФ.bin");
    cout << "ДПФ вычислено и сохранено" << endl;
    
    // Обратное ДПФ
    vector<Complex> idft_result = computeIDFT(dft_result);
    writeBinaryFile(idft_result, "результат_ОДПФ.bin");
    cout << "ОДПФ вычислено и сохранено" << endl;
    
    // Проверка точности
    double max_error = 0.0;
    for (size_t i = 0; i < input.size(); i++) {
        double error = abs(input[i] - idft_result[i]);
        if (error > max_error) max_error = error;
    }
    cout << "Максимальная ошибка восстановления: " << max_error << endl;
    
    return 0;
}