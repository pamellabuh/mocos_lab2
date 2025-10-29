#include <iostream>
#include <fstream>
#include <vector>
#include <complex>
#include <cmath>

using namespace std;

typedef complex<double> Complex;
const double PI = 3.14159265358979323846;

// Прямое БПФ с прореживанием по частоте
vector<Complex> fft(const vector<Complex>& x) {
    int N = x.size();
    
    // Базовый случай рекурсии
    if (N == 1) {
        return x;
    }
    
    // Делим на четные и нечетные частотные компоненты
    vector<Complex> even(N/2);
    vector<Complex> odd(N/2);
    
    // Выполняем операцию "бабочка"
    for (int k = 0; k < N/2; k++) {
        even[k] = x[k] + x[k + N/2];
        Complex temp = x[k] - x[k + N/2];
        double angle = -2.0 * PI * k / N;
        odd[k] = temp * Complex(cos(angle), sin(angle));
    }
    
    // Рекурсивно вычисляем БПФ для половин
    vector<Complex> even_fft = fft(even);
    vector<Complex> odd_fft = fft(odd);
    
    // Объединяем результаты
    vector<Complex> result(N);
    for (int k = 0; k < N/2; k++) {
        result[k] = even_fft[k];
        result[k + N/2] = odd_fft[k];
    }
    
    return result;
}

// Обратное БПФ с прореживанием по частоте
vector<Complex> ifft(const vector<Complex>& x) {
    int N = x.size();
    
    // Базовый случай рекурсии
    if (N == 1) {
        return x;
    }
    
    // Делим на четные и нечетные частотные компоненты
    vector<Complex> even(N/2);
    vector<Complex> odd(N/2);
    
    // Выполняем операцию "бабочка" для обратного преобразования
    for (int k = 0; k < N/2; k++) {
        even[k] = x[k] + x[k + N/2];
        Complex temp = x[k] - x[k + N/2];
        double angle = 2.0 * PI * k / N;  // знак плюс для обратного преобразования
        odd[k] = temp * Complex(cos(angle), sin(angle));
    }
    
    // Рекурсивно вычисляем обратное БПФ для половин
    vector<Complex> even_ifft = ifft(even);
    vector<Complex> odd_ifft = ifft(odd);
    
    // Объединяем результаты
    vector<Complex> result(N);
    for (int k = 0; k < N/2; k++) {
        result[2 * k] = even_ifft[k];
        result[2 * k + 1] = odd_ifft[k];
    }
    
    return result;
}

// Масштабирование результата обратного БПФ (умножение на 1/N)
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
    
    // Чтение входного сигнала (используем файл со степенью двойки)
    vector<Complex> input = readBinaryFile("performance_signals/переменный_64.bin");
    cout << "Загружено " << input.size() << " точек сигнала" << endl;
    
    // Прямое БПФ
    vector<Complex> fft_result = fft(input);
    writeBinaryFile(fft_result, "результат_БПФ.bin");
    cout << "Прямое БПФ вычислено и сохранено" << endl;
    
    // Обратное БПФ
    vector<Complex> ifft_result = scaleIFFT(fft_result);
    writeBinaryFile(ifft_result, "результат_ОБПФ.bin");
    cout << "Обратное БПФ вычислено и сохранено" << endl;
    
    // Проверка точности восстановления
    double max_error = 0.0;
    for (size_t i = 0; i < input.size(); i++) {
        double error = abs(input[i] - ifft_result[i]);
        if (error > max_error) max_error = error;
    }
    
    cout << "Максимальная ошибка восстановления: " << max_error << endl;
    
    return 0;
}