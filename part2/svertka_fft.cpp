#include <iostream>
#include <fstream>
#include <vector>
#include <complex>
#include <cmath>
#include <algorithm>
using namespace std;

typedef complex<double> Complex;
const double PI = 3.14159265358979323846;
Complex wk(int j, int k, int n) {
    double angle = -2.0 * PI * j / (1 << (n + 1 - k));
    return Complex(cos(angle), sin(angle));
}

vector<Complex> fft_dif(const vector<Complex>& x) {
    int N = x.size();
    int n = 0;
    while ((1 << n) < N) n++;
    
    vector<Complex> y = x;
    
    for (int k = 1; k <= n; k++) {
        vector<Complex> temp(N);
        int block_size = 1 << k;
        int half_block = 1 << (k - 1);
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
vector<Complex> ifft_via_fft(const vector<Complex>& A) {
    int N = A.size();
    vector<Complex> U(N);
    for (int i = 0; i < N; i++) {
        U[i] = conj(A[i]);
    }
    vector<Complex> V = fft_dif(U);
    vector<Complex> B(N);
    for (int i = 0; i < N; i++) {
        B[i] = conj(V[i]);
    }
    for (int i = 0; i < N; i++) {
        B[i] /= double(N);
    }
    return B;
}

vector<Complex> fft_convolution_exact(const vector<Complex>& x, const vector<Complex>& y) {
    size_t M = x.size();  
    size_t L = y.size();
    size_t Lu = M + L - 1;
    
    cout << "Lu : " << Lu << endl;
    size_t N_required = max(2 * L, 2 * M);
    size_t N = 1;
    while (N < N_required) {
        N <<= 1;
    }
    cout << "N : " << N << endl;
    vector<Complex> X_extended(N, Complex(0.0, 0.0));
    vector<Complex> Y_extended(N, Complex(0.0, 0.0));
    for (size_t i = 0; i < M; i++) {
        X_extended[i] = x[i];
    }
    for (size_t i = 0; i < L; i++) {
        Y_extended[i] = y[i];
    }
    vector<Complex> X_hat = fft_dif(X_extended);
    vector<Complex> Y_hat = fft_dif(Y_extended);
    vector<Complex> U_hat(N); 
    for (size_t i = 0; i < N; i++) {
        U_hat[i] = X_hat[i] * Y_hat[i];
    }
    vector<Complex> U = ifft_via_fft(U_hat);
    vector<Complex> result(Lu);
    for (size_t i = 0; i < Lu; i++) {
        result[i] = U[i];
    }
    
    return result;
}
vector<Complex> linear_convolution(const vector<Complex>& x, const vector<Complex>& y) {
    size_t M = x.size();
    size_t L = y.size();
    size_t Lu = M + L - 1;
    
    vector<Complex> u(Lu, Complex(0.0, 0.0));
    
    for (size_t n = 0; n < Lu; n++) {
        Complex sum(0.0, 0.0);
        for (size_t k = 0; k < M; k++) {
            if (n >= k && (n - k) < L) {
                sum += x[k] * y[n - k];
            }
        }
        u[n] = sum;
    }
    return u;
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

int main() {    
    try {
        vector<Complex> x = readBinaryFile("performance_signals/переменный_64.bin");
        vector<Complex> y = readBinaryFile("performance_signals/фиксированный_512.bin");
        cout << "x: " << x.size() << " точек" << endl;
        cout << "y: " << y.size() << " точек" << endl;
        vector<Complex> fft_conv = fft_convolution_exact(x, y);
        cout << "Быстрая свертка: " << fft_conv.size() << " точек" << endl;
        vector<Complex> linear_conv = linear_convolution(x, y);
        cout << "Линейная свертка: " << linear_conv.size() << " точек" << endl;
        double max_error = 0.0;
        for (size_t i = 0; i < fft_conv.size(); i++) {
            double error = abs(fft_conv[i] - linear_conv[i]);
            if (error > max_error) max_error = error;
        }
        cout << "Максимальная ошибка между методами: " << max_error << endl;
    } catch (const exception& e) {
        cerr << "Ошибка: " << e.what() << endl;
        return 1;
    }
    return 0;
}