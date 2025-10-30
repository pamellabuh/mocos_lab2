import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('performance_results.csv')

N = data['N']
dft_times = data['DFT_Time_ms']
fft_times = data['FFT_Time_ms']

plt.figure(figsize=(15, 10))
plt.subplot(2, 2, 1)
plt.loglog(N, dft_times, 'ro-', linewidth=2, markersize=8, label='ДПФ')
plt.loglog(N, fft_times, 'bo-', linewidth=2, markersize=8, label='БПФ')
plt.xlabel('Длина последовательности N')
plt.ylabel('Время выполнения (мс)')
plt.title('Зависимость времени выполнения от N\n(логарифмический масштаб)')
plt.grid(True, which="both", ls="-", alpha=0.3)
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(N, dft_times, 'ro-', linewidth=2, markersize=8, label='ДПФ')
plt.plot(N, fft_times, 'bo-', linewidth=2, markersize=8, label='БПФ')
plt.xlabel('Длина последовательности N')
plt.ylabel('Время выполнения (мс)')
plt.title('Зависимость времени выполнения от N\n(линейный масштаб)')
plt.grid(True, alpha=0.3)
plt.legend()

plt.subplot(2, 2, 3)
ratio = np.array(dft_times) / np.array(fft_times)
plt.semilogx(N, ratio, 'g^-', linewidth=2, markersize=8, label='Отношение ДПФ/БПФ')
plt.xlabel('Длина последовательности N')
plt.ylabel('Отношение времен ДПФ/БПФ')
plt.title('Отношение времени выполнения ДПФ к БПФ')
plt.grid(True, which="both", ls="-", alpha=0.3)
plt.legend()

plt.tight_layout()
plt.savefig('performance_comparison.png', dpi=300, bbox_inches='tight')
plt.show()
print("N\t\tДПФ (мс)\tБПФ (мс)\tОтношение ДПФ/БПФ")
for i in range(len(N)):
    print(f"{N[i]}\t\t{dft_times[i]:.3f}\t\t{fft_times[i]:.3f}\t\t{ratio[i]:.2f}")


