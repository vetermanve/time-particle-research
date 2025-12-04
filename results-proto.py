"""
Скрипт для анализа результатов поиска параметров
"""

import json
import numpy as np
import matplotlib.pyplot as plt

def visualize_results(result_file):
    """Визуализирует результаты поиска"""
    
    with open(result_file, 'r') as f:
        data = json.load(f)
    
    solutions = data['solutions']
    
    if not solutions:
        print("Нет данных для визуализации")
        return
    
    # 1. График ошибки по итерациям
    iterations = [s['iteration'] for s in solutions]
    errors = [s['error'] for s in solutions]
    
    plt.figure(figsize=(15, 10))
    
    # График 1: Ошибка vs Итерация
    plt.subplot(2, 3, 1)
    plt.scatter(iterations, errors, alpha=0.5, s=10)
    plt.xlabel('Итерация')
    plt.ylabel('Ошибка')
    plt.title('Ошибка по итерациям')
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    # График 2: Распределение ошибок
    plt.subplot(2, 3, 2)
    plt.hist(errors, bins=50, alpha=0.7)
    plt.xlabel('Ошибка')
    plt.ylabel('Частота')
    plt.title('Распределение ошибок')
    plt.grid(True, alpha=0.3)
    
    # График 3: Массы протона и нейтрона
    proton_masses = [s['details']['proton_mass'] for s in solutions]
    neutron_masses = [s['details']['neutron_mass'] for s in solutions]
    
    plt.subplot(2, 3, 3)
    plt.scatter(proton_masses, neutron_masses, c=errors, alpha=0.5, cmap='viridis')
    plt.colorbar(label='Ошибка')
    plt.xlabel('Масса протона')
    plt.ylabel('Масса нейтрона')
    plt.title('Массы протона и нейтрона')
    plt.grid(True, alpha=0.3)
    
    # Добавляем целевую точку
    plt.scatter(938.272, 939.565, c='red', s=100, marker='*', label='Цель')
    plt.legend()
    
    # График 4: Частоты u и d нитей
    if solutions:
        u_freqs = [s['parameters']['freq_u'] for s in solutions]
        d_freqs = [s['parameters']['freq_d'] for s in solutions]
        
        plt.subplot(2, 3, 4)
        plt.scatter(u_freqs, d_freqs, c=errors, alpha=0.5, cmap='viridis')
        plt.colorbar(label='Ошибка')
        plt.xlabel('Частота u-нити')
        plt.ylabel('Частота d-нити')
        plt.title('Частоты нитей')
        plt.grid(True, alpha=0.3)
    
    # График 5: Сила связи vs Ошибка
    couplings = [s['parameters']['coupling'] for s in solutions]
    
    plt.subplot(2, 3, 5)
    plt.scatter(couplings, errors, alpha=0.5)
    plt.xlabel('Сила связи')
    plt.ylabel('Ошибка')
    plt.title('Зависимость ошибки от силы связи')
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    # График 6: Отношение частот d/u
    if solutions:
        ratios = [d_freqs[i]/u_freqs[i] for i in range(len(u_freqs))]
        
        plt.subplot(2, 3, 6)
        plt.hist(ratios, bins=50, alpha=0.7)
        plt.xlabel('Отношение частот d/u')
        plt.ylabel('Частота')
        plt.title('Распределение отношения частот')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('search_results_visualization.png', dpi=150)
    print("Визуализация сохранена как 'search_results_visualization.png'")
    plt.show()
    
    # Анализ корреляций
    print("\nАНАЛИЗ КОРРЕЛЯЦИЙ:")
    
    # Собираем все параметры в массив
    all_params = []
    param_names = list(solutions[0]['parameters'].keys())
    
    for s in solutions[:100]:  # Берём топ-100 решений
        params = [s['parameters'][name] for name in param_names]
        all_params.append(params)
    
    all_params = np.array(all_params)
    
    # Вычисляем корреляции с ошибкой
    print("\nКорреляция параметров с ошибкой:")
    for i, name in enumerate(param_names):
        correlation = np.corrcoef(all_params[:, i], errors[:100])[0, 1]
        print(f"  {name}: {correlation:.4f}")

# Использование:
result_file = "particle_search_20251204_174943/results.json"
visualize_results(result_file)