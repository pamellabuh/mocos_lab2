#include <iostream>
#include <fstream>
#include <vector>
#include <complex>
#include <cmath>
#include <algorithm>

using namespace std;

typedef complex<double> Complex;
const double PI = 3.14159265358979323846;

// ================== ДПФ функции ==================
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

// ================== БПФ функции ==================
// Прямое БПФ с правильным масштабированием
vector<Complex> fft(const vector<Complex>& x) {
    int N = x.size();
    if (N == 1) return x;
    
    vector<Complex> even(N/2);
    vector<Complex> odd(N/2);
    
    for (int k = 0; k < N/2; k++) {
        even[k] = x[k] + x[k + N/2];
        Complex temp = x[k] - x[k + N/2];
        double angle = -2.0 * PI * k / N;
        odd[k] = temp * Complex(cos(angle), sin(angle));
    }
    
    vector<Complex> even_fft = fft(even);
    vector<Complex> odd_fft = fft(odd);
    
    vector<Complex> result(N);
    for (int k = 0; k < N/2; k++) {
        result[k] = even_fft[k];
        result[k + N/2] = odd_fft[k];
    }
    
    // Масштабирование для соответствия ДПФ
    double scale = 1.0 / sqrt(N);
    for (int i = 0; i < N; i++) {
        result[i] *= scale;
    }
    
    return result;
}

// Обратное БПФ с правильным масштабированием
vector<Complex> ifft(const vector<Complex>& x) {
    int N = x.size();
    if (N == 1) return x;
    
    vector<Complex> even(N/2);
    vector<Complex> odd(N/2);
    
    for (int k = 0; k < N/2; k++) {
        even[k] = x[k] + x[k + N/2];
        Complex temp = x[k] - x[k + N/2];
        double angle = 2.0 * PI * k / N;
        odd[k] = temp * Complex(cos(angle), sin(angle));
    }
    
    vector<Complex> even_ifft = ifft(even);
    vector<Complex> odd_ifft = ifft(odd);
    
    vector<Complex> result(N);
    for (int k = 0; k < N/2; k++) {
        result[2 * k] = even_ifft[k];
        result[2 * k + 1] = odd_ifft[k];
    }
    
    // Масштабирование для соответствия ОДПФ
    double scale = 1.0 / sqrt(N);
    for (int i = 0; i < N; i++) {
        result[i] *= scale;
    }
    
    return result;
}

// Функция scaleIFFT (которая отсутствовала)
vector<Complex> scaleIFFT(const vector<Complex>& x) {
    return ifft(x);  // Просто вызываем ifft, который уже масштабирован
}

// ================== Вспомогательные функции ==================
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
        norm += abs(a[i] - b[i]) * abs(a[i] - b[i]);
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
    cout << "=== ПРОВЕРКА КОРРЕКТНОСТИ АЛГОРИТМОВ ===" << endl;
    cout << "Размер сигнала: N = 1024 (2¹⁰)" << endl << endl;
    
    // Чтение сигнала
    vector<Complex> X = readBinaryFile("performance_signals/переменный_1024.bin");
    cout << "1. Загружен сигнал X длиной " << X.size() << " точек" << endl;
    
    // ================== ПРОВЕРКА 1: Восстановление сигнала ==================
    cout << endl << "1. ПРОВЕРКА ВОССТАНОВЛЕНИЯ СИГНАЛА:" << endl;
    
    // ДПФ → ОДПФ
    vector<Complex> dft_X = computeDFT(X);
    vector<Complex> idft_dft_X = computeIDFT(dft_X);
    double error_dft = maxError(X, idft_dft_X);
    cout << "   X = ОДПФ(ДПФ(X)): ошибка = " << scientific << error_dft;
    if (error_dft < 1e-10) cout << " ✓ УСПЕХ";
    else cout << " ⚠ ОШИБКА";
    cout << endl;
    
    // БПФ → ОБПФ
    vector<Complex> fft_X = fft(X);
    vector<Complex> ifft_fft_X = scaleIFFT(fft_X);  // Теперь эта функция есть!
    double error_fft = maxError(X, ifft_fft_X);
    cout << "   X = ОБПФ(БПФ(X)): ошибка = " << scientific << error_fft;
    if (error_fft < 1e-10) cout << " ✓ УСПЕХ";
    else cout << " ⚠ ОШИБКА";
    cout << endl;
    
    // ================== ПРОВЕРКА 2: Сравнение ДПФ и БПФ ==================
    cout << endl << "2. СРАВНЕНИЕ ДПФ И БПФ:" << endl;
    
    // Поскольку и ДПФ и БПФ теперь используют одинаковое масштабирование,
    // их результаты должны совпадать
    double norm_diff_dft_fft = vectorNorm(dft_X, fft_X);
    cout << "   Норма разности ДПФ(X) и БПФ(X): " << scientific << norm_diff_dft_fft;
    if (norm_diff_dft_fft < 1e-10) cout << " ✓ РЕЗУЛЬТАТЫ СОВПАДАЮТ";
    else cout << " ⚠ РАЗЛИЧИЯ";
    cout << endl;
    
    // ================== ПРОВЕРКА 3: Сохранение для сравнения с Python ==================
    cout << endl << "3. СОХРАНЕНИЕ РЕЗУЛЬТАТОВ ДЛЯ СРАВНЕНИЯ:" << endl;
    
    writeBinaryFile(X, "signal_X_1024.bin");
    writeBinaryFile(dft_X, "dft_result_1024.bin");
    writeBinaryFile(fft_X, "fft_result_1024.bin");
    
    cout << "   Сохранены файлы:" << endl;
    cout << "   - signal_X_1024.bin (исходный сигнал)" << endl;
    cout << "   - dft_result_1024.bin (результат ДПФ)" << endl;
    cout << "   - fft_result_1024.bin (результат БПФ)" << endl;
    cout << "   Используй их для сравнения с numpy.fft.fft в Python" << endl;
    
    // ================== ВЫВОД СТАТИСТИКИ ==================
    cout << endl << "=== СВОДКА РЕЗУЛЬТАТОВ ===" << endl;
    cout << "Ошибка восстановления ДПФ: " << scientific << error_dft << endl;
    cout << "Ошибка восстановления БПФ: " << scientific << error_fft << endl;
    cout << "Разность ДПФ/БПФ: " << scientific << norm_diff_dft_fft << endl;
    
    if (error_dft < 1e-10 && error_fft < 1e-10 && norm_diff_dft_fft < 1e-10) {
        cout << "🎉 ВСЕ АЛГОРИТМЫ РАБОТАЮТ КОРРЕКТНО!" << endl;
    } else {
        cout << "⚠ ЕСТЬ ПРОБЛЕМЫ В РЕАЛИЗАЦИИ" << endl;
    }
    
    return 0;
}