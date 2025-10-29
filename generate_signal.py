import numpy as np
import os

def generate_signal_bin(filename, length):
    # Генерируем действительную часть (случайные числа)
    real_part = np.random.randn(length).astype(np.float64)
    # Генерируем мнимую часть (случайные числа)
    imag_part = np.random.randn(length).astype(np.float64)
    
    # Создаем массив для записи в формате: real, imag, real, imag, ...
    data_to_write = np.zeros(2 * length, dtype=np.float64)
    data_to_write[0::2] = real_part    # четные позиции - действительные части
    data_to_write[1::2] = imag_part    # нечетные позиции - мнимые части

    # Записываем данные в бинарный файл
    with open(filename, 'wb') as f:
        data_to_write.tofile(f)

    print(f"Сгенерирован файл {filename} с {length} элементами")

# Создаем папку для тестовых сигналов
if not os.path.exists('performance_signals'):
    os.makedirs('performance_signals')

# Фиксированный сигнал (512 точек) - для ДПФ
fixed_size = 512
fixed_file = f'performance_signals/фиксированный_{fixed_size}.bin'
generate_signal_bin(fixed_file, fixed_size)

# Сигналы с размерами - степени двойки: N = 2^n, n от 6 до 12
sizes = [2**n for n in range(6, 13)]  # 64, 128, 256, 512, 1024, 2048, 4096

for size in sizes:
    var_file = f'performance_signals/переменный_{size}.bin'
    generate_signal_bin(var_file, size)

print(f"\nСгенерировано {len(sizes) + 1} файлов в папке 'performance_signals/'")
print("Размер фиксированного сигнала:", fixed_size)
print("Размеры переменных сигналов:", sizes)
