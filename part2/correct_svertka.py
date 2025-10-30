import numpy as np

def read_binary_complex(filename):
    data = np.fromfile(filename, dtype=np.float64)
    return data[0::2] + 1j * data[1::2]

def main():
    x = read_binary_complex('signal_x1.bin')
    y = read_binary_complex('signal_y1.bin')

    u_direct_cpp = read_binary_complex('convolution_direct.bin')
    u_fft_cpp = read_binary_complex('convolution_fft.bin')
    u_numpy = np.convolve(x, y, mode='full')
    error_direct_numpy = np.linalg.norm(u_direct_cpp - u_numpy)
    error_fft_numpy = np.linalg.norm(u_fft_cpp - u_numpy)
    
    print(f"Норма разности (прямая C++ и numpy): {error_direct_numpy:.6e}")
    print(f"Норма разности (БПФ C++ и numpy):    {error_fft_numpy:.6e}")

    if error_direct_numpy < 1e-10 and error_fft_numpy < 1e-10:
        print("\nнорм")
    else:
        print("\nчто то не так")

if __name__ == "__main__":
    main()