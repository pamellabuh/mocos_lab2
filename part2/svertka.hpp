#ifndef CONV_HPP
#define CONV_HPP

#include <vector>
#include <complex>

typedef std::complex<double> Complex;
typedef std::vector<Complex> Sequence;

class Convolution {
public:
    static Sequence linear_convolution(const Sequence& x, const Sequence& y) {
        size_t M = x.size();
        size_t L = y.size();
        size_t Un = M + L - 1;
        
        Sequence u(Un, Complex(0.0, 0.0));
        
        for (size_t n = 0; n < Un; ++n) {
            Complex sum(0.0, 0.0);
            for (size_t k = 0; k < M; ++k) {
                if (n >= k && (n - k) < L) {
                    sum += x[k] * y[n - k];
                }
            }
            u[n] = sum;
        }
        return u;
    }
};

#endif