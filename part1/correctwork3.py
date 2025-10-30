import numpy as np
import struct
import matplotlib.pyplot as plt

def read_binary_file(filename):
    with open(filename, 'rb') as f:
        data = f.read()
    num_doubles = len(data) // 8 
    doubles = struct.unpack('d' * num_doubles, data)
    complex_data = []
    for i in range(0, num_doubles, 2):
        complex_data.append(complex(doubles[i], doubles[i+1]))
    
    return np.array(complex_data)

def main():
    try:

        X = read_binary_file('performance_signals/переменный_1024.bin')
        print(f"Размер сигнала: {X.shape}")
        fft_numpy = np.fft.fft(X, norm='ortho')
        fft_cpp = read_binary_file('fft_result.bin')
        dft_cpp = read_binary_file('dft_result.bin')
        diff_fft = np.linalg.norm(fft_numpy - fft_cpp)
        diff_dft = np.linalg.norm(fft_numpy - dft_cpp)
        print(f"Норма разности numpy БПФ и БПФ на с++: {diff_fft:.6e}")
        print(f"Норма разности numpy БПФ и ДПФ : {diff_dft:.6e}")        
    except FileNotFoundError as e:
        print(f"Ошибка: Файл не найден - {e}")
    except Exception as e:
        print(f"Ошибка: {e}")

if __name__ == "__main__":
    main()