import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib

data = pd.read_csv('convolution_performance.csv')
print(data.head())

case1 = data[data['Case'] == 'Fixed_M_512']
case2 = data[data['Case'] == 'Variable_Both']

print(f"\nСлучай 1 (фиксированная M=512): {len(case1)} точек")
print(f"Случай 2 (обе переменные): {len(case2)} точек")

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

ax1.plot(case1['Size'], case1['Direct_Time'], 'ro-', linewidth=2, markersize=8, label='Прямая свертка')
ax1.plot(case1['Size'], case1['FFT_Time'], 'bo-', linewidth=2, markersize=8, label='БПФ-свертка')
ax1.set_xlabel('Длина L переменной последовательности')
ax1.set_ylabel('Время выполнения (мс)')
ax1.set_title('Случай 1: Фиксированная M=512, переменная L\n(линейный масштаб)')
ax1.grid(True, alpha=0.3)
ax1.legend()

ax2.loglog(case1['Size'], case1['Direct_Time'], 'ro-', linewidth=2, markersize=8, label='Прямая свертка')
ax2.loglog(case1['Size'], case1['FFT_Time'], 'bo-', linewidth=2, markersize=8, label='БПФ-свертка')
ax2.set_xlabel('Длина L переменной последовательности')
ax2.set_ylabel('Время выполнения (мс)')
ax2.set_title('Случай 1: Фиксированная M=512, переменная L\n(логарифмический масштаб)')
ax2.grid(True, which="both", ls="-", alpha=0.3)
ax2.legend()

ax3.plot(case2['Size'], case2['Direct_Time'], 'ro-', linewidth=2, markersize=8, label='Прямая свертка')
ax3.plot(case2['Size'], case2['FFT_Time'], 'bo-', linewidth=2, markersize=8, label='БПФ-свертка')
ax3.set_xlabel('Длина N последовательностей')
ax3.set_ylabel('Время выполнения (мс)')
ax3.set_title('Случай 2: L = M = N (обе переменные)\n(линейный масштаб)')
ax3.grid(True, alpha=0.3)
ax3.legend()

ax4.loglog(case2['Size'], case2['Direct_Time'], 'ro-', linewidth=2, markersize=8, label='Прямая свертка')
ax4.loglog(case2['Size'], case2['FFT_Time'], 'bo-', linewidth=2, markersize=8, label='БПФ-свертка')
ax4.set_xlabel('Длина N последовательностей')
ax4.set_ylabel('Время выполнения (мс)')
ax4.set_title('Случай 2: L = M = N (обе переменные)\n(логарифмический масштаб)')
ax4.grid(True, which="both", ls="-", alpha=0.3)
ax4.legend()

plt.tight_layout()
plt.savefig('convolution_performance.png', dpi=300, bbox_inches='tight')
plt.show()

print("L\tПрямая(мс)\tБПФ(мс)\tУскорение")
print("-"*50)
for _, row in case1.iterrows():
    speedup = row['Direct_Time'] / row['FFT_Time']
    print(f"{row['Size']}\t{row['Direct_Time']:.3f}\t\t{row['FFT_Time']:.3f}\t{speedup:.1f}x")

print("N\tПрямая(мс)\tБПФ(мс)\tУскорение")
print("-"*50)
for _, row in case2.iterrows():
    speedup = row['Direct_Time'] / row['FFT_Time']
    print(f"{row['Size']}\t{row['Direct_Time']:.3f}\t\t{row['FFT_Time']:.3f}\t{speedup:.1f}x")