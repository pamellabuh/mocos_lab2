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

vector<Complex> scaleFFT(const vector<Complex>& x) {
    int N = x.size();
    vector<Complex> result = fft_dif(x);
    double scale = 1.0 / sqrt(N);
    
    for (int i = 0; i < N; i++) {
        result[i] *= scale;
    }
    return result;
}

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

pair<double, double> measurePerformance(int N, int num_runs = 5) {
    vector<Complex> signal;
    
    string filename = "performance_signals/переменный_" + to_string(N) + ".bin";
    signal = readBinaryFile(filename);
    
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
    
    double time_dft = total_dft / (1000.0 * num_runs); 
    double time_fft = total_fft / (1000.0 * num_runs);
    
    cout << "N = " << N << ":\tДПФ = " << time_dft << " мс,\tБПФ = " << time_fft << " мс" << endl;
    
    return make_pair(time_dft, time_fft);
}

int main() {
    vector<int> sizes;
    vector<double> times_dft, times_fft;
    
    cout << "Измерение времени выполнения ДПФ и БПФ:" << endl;
    cout << "N\t\tДПФ (мс)\t\tБПФ (мс)" << endl;

    for (int n = 6; n <= 12; n++) {
        int N = pow(2, n);
        sizes.push_back(N);
        
        auto times = measurePerformance(N);
        times_dft.push_back(times.first);
        times_fft.push_back(times.second);
    }

    ofstream file("performance_results.csv");
    file << "N,DFT_Time_ms,FFT_Time_ms\n";
    for (size_t i = 0; i < sizes.size(); i++) {
        file << sizes[i] << "," << times_dft[i] << "," << times_fft[i] << "\n";
    }
    file.close();   
    return 0;
}