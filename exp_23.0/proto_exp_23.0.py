"""
МОДЕЛЬ v23 — ИСПРАВЛЕНИЕ СИСТЕМНЫХ ПРОБЛЕМ v22
Ключевые улучшения:
1. Отдельная модель для смешанных состояний (π⁰, η)
2. Электромагнитные поправки, зависящие от заряда
3. Разные эффективные массы s-кварка в мезонах и барионах
4. Сохранение идеальной точности для протона, нейтрона, π⁺
"""

import numpy as np
import json
from datetime import datetime
import os
import sys
import math

class TopologicalModelV23:
    """Топологическая модель с исправлениями системных проблем"""
    
    def __init__(self):
        self.scale = 100.0
        
        # Целевые массы (фокус на проблемных частицах)
        self.targets = {
            # Идеальные из v22 (должны остаться идеальными)
            'proton': 938.272,
            'neutron': 939.565,
            'pi+': 139.570,
            'pi-': 139.570,
            
            # Проблемные из v22 (должны быть исправлены)
            'pi0': 134.9768,    # Ошибка 13.4% в v22
            'K+': 493.677,      # Ошибка 14.7% в v22
            'K0': 497.611,      # Разница с K⁺ 4 МэВ
            'K-': 493.677,
            'eta': 547.862,     # Ошибка 15.7% в v22
            
            # Странные барионы (для проверки)
            'Lambda0': 1115.683,
            'Sigma+': 1189.37,
        }
        
        # Состав с учетом смешивания
        self.composition = {
            'proton': ['u', 'u', 'd'],
            'neutron': ['u', 'd', 'd'],
            'pi+': ['u', 'anti_d'],
            'pi-': ['d', 'anti_u'],
            
            # π⁰ — смешанное состояние
            'pi0': ['mixed_pi0'],
            
            # Каоны
            'K+': ['u', 'anti_s'],
            'K0': ['d', 'anti_s'],
            'K-': ['s', 'anti_u'],
            
            # η — сложное смешивание
            'eta': ['mixed_eta'],
            
            # Странные барионы
            'Lambda0': ['u', 'd', 's'],
            'Sigma+': ['u', 'u', 's'],
        }
        
        # Топология (обновлена для смешанных состояний)
        self.topology = {
            'proton':  {'L': 2, 'T': 0.0, 'S': 1.0, 'type': 'baryon'},
            'neutron': {'L': 1, 'T': 0.0, 'S': 0.9, 'type': 'baryon'},
            'pi+':     {'L': 3, 'T': 0.0, 'S': 1.0, 'type': 'meson'},
            'pi-':     {'L': 3, 'T': 0.0, 'S': 1.0, 'type': 'meson'},
            'pi0':     {'L': 3, 'T': 0.0, 'S': 0.95, 'type': 'meson_mixed'},  # Специальный тип
            'K+':      {'L': 3, 'T': 0.3, 'S': 1.0, 'type': 'meson_strange'},
            'K0':      {'L': 3, 'T': 0.3, 'S': 0.95, 'type': 'meson_strange'},
            'K-':      {'L': 3, 'T': 0.3, 'S': 1.0, 'type': 'meson_strange'},
            'eta':     {'L': 4, 'T': 0.5, 'S': 1.1, 'type': 'meson_mixed'},
            'Lambda0': {'L': 2, 'T': 0.4, 'S': 1.1, 'type': 'baryon_strange'},
            'Sigma+':  {'L': 2, 'T': 0.5, 'S': 1.0, 'type': 'baryon_strange'},
        }
        
        # Параметры из v22 (идеальные для протона, нейтрона, π⁺)
        self.params = {
            # Эффективные массы (из v22)
            'm_u': 2.537656,
            'm_d': 3.216655,
            'm_s_meson': 8.220566,   # s-кварк в мезонах
            'm_s_baryon': 10.0,      # s-кварк в барионах (будем оптимизировать)
            
            # Топологические коэффициенты (из v22)
            'alpha': 1.001118,
            'beta': 0.202976,
            'gamma': 0.142900,
            
            # Знаки энергии связи (из v22)
            'sign_baryon': 0.500336,
            'sign_meson': -1.385960,
            
            # Коэффициент нейтрона (из v22)
            'k_neutron': 0.749951,
            
            # НОВЫЕ ПАРАМЕТРЫ ДЛЯ ИСПРАВЛЕНИЯ ПРОБЛЕМ
            
            # 1. Смешивание для π⁰ и η
            'mixing_pi0': 0.95,       # Как в v22, но с отдельной обработкой
            'mixing_eta': 0.85,
            
            # 2. Электромагнитные поправки (критично для K⁺/K⁰)
            'delta_em_charged': 0.010,  # Поправка для заряженных частиц
            'delta_em_neutral': 0.002,  # Маленькая поправка для нейтральных
            
            # 3. Разные массы s-кварка в разных контекстах
            'lambda_s_meson': 0.743615,  # Из v22
            'lambda_s_baryon': 0.9,      # s-кварк в барионах
            
            # 4. Коррекция для разности K⁰-K⁺
            'k_K_mass_splitting': 1.05,  # Учитывает, что d тяжелее u
            
            # 5. Отдельные поправки для проблемных частиц
            'correction_pi0': 1.15,      # Коррекция массы π⁰
            'correction_eta': 1.18,      # Коррекция массы η
        }
    
    def calculate_base_mass(self, particle):
        """Базовая масса с учётом смешивания"""
        comp = self.composition[particle]
        
        # Особые случаи смешанных состояний
        if particle == 'pi0':
            # π⁰ = (uū + dđ)/√2 упрощённо
            u_part = 2 * self.params['m_u'] * 0.707  # 1/√2
            d_part = 2 * self.params['m_d'] * 0.707
            return (u_part + d_part) * self.params['correction_pi0']
        
        if particle == 'eta':
            # η = (uū + dđ - 2sš)/√6 упрощённо
            u_part = 2 * self.params['m_u'] * 0.408  # 1/√6
            d_part = 2 * self.params['m_d'] * 0.408
            s_part = 2 * self.params['m_s_meson'] * 0.408 * 2  # ×2 для -2sš
            return (u_part + d_part + s_part) * self.params['correction_eta']
        
        # Обычные частицы
        total = 0
        for quark in comp:
            if quark in ['u', 'anti_u']:
                total += self.params['m_u']
            elif quark in ['d', 'anti_d']:
                total += self.params['m_d']
            elif quark == 'anti_s':
                total += self.params['m_s_meson'] * self.params['lambda_s_meson']
            elif quark == 's':
                # s-кварк: различаем мезоны и барионы
                if 'meson' in self.topology[particle]['type']:
                    total += self.params['m_s_meson'] * self.params['lambda_s_meson']
                else:  # baryon
                    total += self.params['m_s_baryon'] * self.params['lambda_s_baryon']
        
        return total
    
    def calculate_topological_energy(self, particle):
        """Топологическая энергия с поправками"""
        topo = self.topology[particle]
        
        # Базовая энергия
        energy = (self.params['alpha'] * topo['L'] + 
                  self.params['beta'] * topo['T'] + 
                  self.params['gamma'] * topo['S'])
        
        # Коррекции для конкретных частиц
        if particle == 'neutron':
            energy *= self.params['k_neutron']
        
        # Специальные поправки для смешанных состояний
        if particle == 'pi0':
            energy *= self.params['mixing_pi0']
        elif particle == 'eta':
            energy *= self.params['mixing_eta']
        
        # Поправка для K⁰ (чтобы учесть разницу с K⁺)
        if particle == 'K0':
            energy *= self.params['k_K_mass_splitting']
        
        return energy
    
    def calculate_mass(self, particle):
        """Основная формула с электромагнитными поправками"""
        base = self.calculate_base_mass(particle)
        topo_energy = self.calculate_topological_energy(particle)
        topo = self.topology[particle]
        
        # Определяем знак
        if 'baryon' in topo['type']:
            mass = (base + self.params['sign_baryon'] * topo_energy) * self.scale
        else:  # meson
            mass = (base + self.params['sign_meson'] * topo_energy) * self.scale
        
        # ЭЛЕКТРОМАГНИТНЫЕ ПОПРАВКИ (критично!)
        if particle in ['proton', 'pi+', 'pi-', 'K+', 'K-', 'Sigma+']:
            # Заряженные частицы: положительная поправка
            mass *= (1.0 + self.params['delta_em_charged'])
        elif particle in ['neutron', 'pi0', 'K0', 'Lambda0', 'eta']:
            # Нейтральные: маленькая поправка (или отрицательная)
            mass *= (1.0 + self.params['delta_em_neutral'])
        
        # Дополнительная коррекция для K⁰
        if particle == 'K0':
            # K⁰ должен быть тяжелее K⁺ на ~4 МэВ
            # У нас m_d > m_u, поэтому K⁰ уже тяжелее
            # Но нужно точно настроить
            pass
        
        return max(mass, 1.0)
    
    def error_function(self, params_array):
        """Функция ошибки с приоритетом на исправление проблем"""
        # Параметры для оптимизации
        param_names = [
            # Основные (фиксируем из v22)
            'm_u', 'm_d', 'm_s_meson', 'm_s_baryon',
            'alpha', 'beta', 'gamma',
            'sign_baryon', 'sign_meson', 'k_neutron',
            
            # Новые для оптимизации
            'mixing_pi0', 'mixing_eta',
            'delta_em_charged', 'delta_em_neutral',
            'lambda_s_meson', 'lambda_s_baryon',
            'k_K_mass_splitting',
            'correction_pi0', 'correction_eta'
        ]
        
        for i, name in enumerate(param_names):
            self.params[name] = params_array[i]
        
        total_error = 0.0
        
        # ВЕСА (очень важны!)
        weights = {
            # Идеальные из v22 (максимальный вес для сохранения)
            'proton': 20.0,
            'neutron': 20.0,
            'pi+': 15.0,
            'pi-': 15.0,
            
            # Проблемные (большой вес для исправления)
            'pi0': 15.0,      # Было 13.4% ошибки
            'K+': 12.0,       # Было 14.7% ошибки
            'K0': 12.0,       # Критично для разности с K⁺
            'K-': 10.0,
            
            # Остальные
            'eta': 8.0,
            'Lambda0': 5.0,
            'Sigma+': 5.0,
        }
        
        # Вычисляем ошибки
        for particle, target in self.targets.items():
            mass = self.calculate_mass(particle)
            weight = weights.get(particle, 1.0)
            
            # Комбинированная ошибка
            rel_error = abs(mass - target) / target
            abs_error = abs(mass - target) / 1000.0
            
            error = weight * (rel_error + abs_error)
            total_error += error
        
        # ЖЕСТКИЕ ФИЗИЧЕСКИЕ ОГРАНИЧЕНИЯ
        
        # 1. Разность масс n-p = 1.293 ± 0.05 МэВ
        mass_n = self.calculate_mass('neutron')
        mass_p = self.calculate_mass('proton')
        diff_np = mass_n - mass_p
        if abs(diff_np - 1.293) > 0.1:
            total_error += 50.0 * abs(diff_np - 1.293)
        
        # 2. Разность K⁰ - K⁺ = 4.0 ± 0.5 МэВ
        mass_K0 = self.calculate_mass('K0')
        mass_Kp = self.calculate_mass('K+')
        diff_K = mass_K0 - mass_Kp
        if abs(diff_K - 4.0) > 1.0:
            total_error += 30.0 * abs(diff_K - 4.0)
        
        # 3. Массы π⁺ и π⁻ должны быть равны
        if abs(self.calculate_mass('pi+') - self.calculate_mass('pi-')) > 0.1:
            total_error += 100.0
        
        # 4. m_s_baryon > m_s_meson > m_d > m_u
        if self.params['m_s_baryon'] <= self.params['m_s_meson']:
            total_error += 50.0
        if self.params['m_s_meson'] <= self.params['m_d']:
            total_error += 50.0
        if self.params['m_d'] <= self.params['m_u']:
            total_error += 50.0
        
        # 5. Знаки энергии связи
        if self.params['sign_baryon'] <= 0:
            total_error += 1000.0
        if self.params['sign_meson'] >= 0:
            total_error += 1000.0
        
        return total_error

def run_v23_optimization():
    """Оптимизация v23 с фокусом на исправление проблем"""
    print("="*80)
    print("МОДЕЛЬ v23 — ИСПРАВЛЕНИЕ СИСТЕМНЫХ ПРОБЛЕМ v22")
    print("="*80)
    
    model = TopologicalModelV23()
    
    # Начальные параметры (из v22 + разумные значения новых)
    initial_params = np.array([
        # Основные из v22 (идеальные для протона, нейтрона, π⁺)
        2.537656,   # m_u
        3.216655,   # m_d
        8.220566,   # m_s_meson (из v22)
        10.0,       # m_s_baryon (оценка)
        
        1.001118,   # alpha
        0.202976,   # beta
        0.142900,   # gamma
        
        0.500336,   # sign_baryon
        -1.385960,  # sign_meson
        0.749951,   # k_neutron
        
        # Новые параметры (начальные оценки)
        0.95,       # mixing_pi0
        0.85,       # mixing_eta
        0.010,      # delta_em_charged
        0.002,      # delta_em_neutral
        0.743615,   # lambda_s_meson (из v22)
        0.9,        # lambda_s_baryon
        1.05,       # k_K_mass_splitting
        1.15,       # correction_pi0
        1.18,       # correction_eta
    ])
    
    # Границы параметров
    bounds = [
        # m_u, m_d, m_s_meson, m_s_baryon
        (2.4, 2.7), (3.1, 3.4), (7.5, 9.0), (9.0, 12.0),
        # alpha, beta, gamma
        (0.8, 1.2), (0.1, 0.3), (0.1, 0.2),
        # sign_baryon, sign_meson, k_neutron
        (0.4, 0.6), (-1.5, -1.2), (0.7, 0.8),
        # mixing_pi0, mixing_eta
        (0.9, 1.0), (0.8, 0.9),
        # delta_em_charged, delta_em_neutral
        (0.005, 0.015), (0.0, 0.005),
        # lambda_s_meson, lambda_s_baryon
        (0.7, 0.8), (0.8, 1.0),
        # k_K_mass_splitting, correction_pi0, correction_eta
        (1.0, 1.1), (1.1, 1.2), (1.1, 1.3)
    ]
    
    # Метод отжига
    import random
    import time
    
    current_params = initial_params.copy()
    current_error = model.error_function(current_params)
    
    best_params = current_params.copy()
    best_error = current_error
    
    temperature = 2.0
    cooling_rate = 0.999995
    iterations = 1000000
    
    start_time = time.time()
    
    print(f"\nНачальная ошибка: {current_error:.4f}")
    print("Запуск оптимизации...")
    
    for i in range(iterations):
        # Мутация
        new_params = current_params.copy()
        for j in range(len(new_params)):
            if random.random() < 0.25:
                min_val, max_val = bounds[j]
                mutation = random.uniform(-0.01, 0.01) * (max_val - min_val)
                new_val = current_params[j] + mutation
                new_val = max(min_val, min(max_val, new_val))
                new_params[j] = new_val
        
        # Оценка
        new_error = model.error_function(new_params)
        
        # Метрополис
        if new_error < current_error or random.random() < math.exp((current_error - new_error) / temperature):
            current_params = new_params
            current_error = new_error
        
        # Лучшее решение
        if new_error < best_error:
            best_params = new_params.copy()
            best_error = new_error
        
        # Охлаждение
        temperature *= cooling_rate
        
        if i % 100000 == 0:
            elapsed = time.time() - start_time
            progress = i / iterations * 100
            
            # Вычисляем ключевые массы
            model.error_function(best_params)
            mass_p = model.calculate_mass('proton')
            mass_n = model.calculate_mass('neutron')
            mass_pi0 = model.calculate_mass('pi0')
            mass_K0 = model.calculate_mass('K0')
            mass_Kp = model.calculate_mass('K+')
            
            diff_np = mass_n - mass_p
            diff_K = mass_K0 - mass_Kp
            
            print(f"\rИтерация {i:,}/{iterations:,} ({progress:.1f}%) | "
                  f"Ошибка: {best_error:.4f} | "
                  f"n-p: {diff_np:.3f} | "
                  f"K⁰-K⁺: {diff_K:.3f} | "
                  f"π⁰: {mass_pi0:.1f}",
                  end='', flush=True)
    
    # Финальные результаты
    elapsed = time.time() - start_time
    print(f"\n\nОптимизация завершена за {elapsed:.1f} сек")
    print(f"Лучшая ошибка: {best_error:.6f}")
    
    return model, best_params

def evaluate_v23(model, best_params):
    """Оценка результатов v23"""
    # Обновляем параметры
    param_names = [
        'm_u', 'm_d', 'm_s_meson', 'm_s_baryon',
        'alpha', 'beta', 'gamma',
        'sign_baryon', 'sign_meson', 'k_neutron',
        'mixing_pi0', 'mixing_eta',
        'delta_em_charged', 'delta_em_neutral',
        'lambda_s_meson', 'lambda_s_baryon',
        'k_K_mass_splitting',
        'correction_pi0', 'correction_eta'
    ]
    
    for i, name in enumerate(param_names):
        model.params[name] = best_params[i]
    
    # Вычисляем все массы
    results = {}
    for particle in model.targets:
        results[particle] = model.calculate_mass(particle)
    
    print("\n" + "="*80)
    print("РЕЗУЛЬТАТЫ v23")
    print("="*80)
    
    # Массы частиц
    print(f"\n{'Частица':<10} {'Расчёт':<10} {'Цель':<10} {'Ошибка %':<10} {'Статус':<6}")
    print("-"*60)
    
    total_error = 0
    for particle, target in model.targets.items():
        mass = results[particle]
        error = abs(mass - target) / target * 100
        total_error += error
        
        if error < 1.0:
            status = "✓✓✓"
        elif error < 3.0:
            status = "✓✓"
        elif error < 10.0:
            status = "✓"
        else:
            status = "⚠"
        
        print(f"{particle:<10} {mass:<10.1f} {target:<10.1f} {error:<10.2f} {status:<6}")
    
    avg_error = total_error / len(model.targets)
    print(f"\nСредняя ошибка: {avg_error:.2f}%")
    
    # Ключевые проверки
    print(f"\nКЛЮЧЕВЫЕ ПРОВЕРКИ:")
    
    diff_np = results['neutron'] - results['proton']
    print(f"  Разность n-p: {diff_np:.3f} МэВ (цель 1.293)")
    
    diff_K = results['K0'] - results['K+']
    print(f"  Разность K⁰-K⁺: {diff_K:.3f} МэВ (цель 4.0)")
    
    diff_pi = results['pi+'] - results['pi-']
    print(f"  Разность π⁺-π⁻: {diff_pi:.3f} МэВ (цель 0.0)")
    
    # Эффективные массы в МэВ
    m_u_mev = model.params['m_u'] * 100
    m_d_mev = model.params['m_d'] * 100
    m_s_meson_mev = model.params['m_s_meson'] * model.params['lambda_s_meson'] * 100
    m_s_baryon_mev = model.params['m_s_baryon'] * model.params['lambda_s_baryon'] * 100
    
    print(f"\nЭФФЕКТИВНЫЕ МАССЫ КВАРКОВ:")
    print(f"  u: {m_u_mev:.1f} МэВ")
    print(f"  d: {m_d_mev:.1f} МэВ")
    print(f"  s (мезоны): {m_s_meson_mev:.1f} МэВ")
    print(f"  s (барионы): {m_s_baryon_mev:.1f} МэВ")
    print(f"  Отношение m_d/m_u: {m_d_mev/m_u_mev:.3f}")
    
    # Сохранение результатов
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"v23_results_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    
    results_data = {
        'parameters': {name: float(best_params[i]) for i, name in enumerate(param_names)},
        'masses': {p: float(results[p]) for p in results},
        'key_tests': {
            'n_p_diff': float(diff_np),
            'K0_Kp_diff': float(diff_K),
            'pi_plus_minus_diff': float(diff_pi)
        },
        'quark_masses_mev': {
            'u': float(m_u_mev),
            'd': float(m_d_mev),
            's_meson': float(m_s_meson_mev),
            's_baryon': float(m_s_baryon_mev)
        }
    }
    
    with open(f"{results_dir}/v23_results.json", 'w') as f:
        json.dump(results_data, f, indent=2)
    
    print(f"\nРезультаты сохранены в {results_dir}/")
    
    return results

if __name__ == "__main__":
    # Запуск оптимизации
    model, best_params = run_v23_optimization()
    
    # Оценка результатов
    results = evaluate_v23(model, best_params)