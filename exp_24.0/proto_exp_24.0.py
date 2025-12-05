"""
v24 — Квантово-топологическая модель нейтральных мезонов
Основная идея: π⁰, η, η' — квантовые суперпозиции базовых состояний |uū⟩, |dđ⟩, |sš⟩
"""

import numpy as np
import json
from datetime import datetime
import os
import sys
import math
from scipy.linalg import eigh

class QuantumTopologicalModelV24:
    """Квантово-топологическая модель с квантовым смешиванием"""
    
    def __init__(self):
        self.scale = 100.0
        
        # Целевые массы (расширенный набор для смешанных состояний)
        self.targets = {
            # Идеальные из v23 (должны остаться идеальными)
            'proton': 938.272,
            'neutron': 939.565,
            'pi+': 139.570,
            'pi-': 139.570,
            'K+': 493.677,
            'K0': 497.611,
            'K-': 493.677,
            
            # Смешанные состояния (основная цель v24)
            'pi0': 134.9768,    # Критическая проблема v23: 465.9 vs 135.0
            'eta': 547.862,     # Критическая проблема v23: 1506.9 vs 547.9
            'eta_prime': 957.78,  # η' (добавляем для полной системы)
            
            # Странные барионы
            'Lambda0': 1115.683,
            'Sigma+': 1189.37,
        }
        
        # Состав частиц (физические частицы как смеси)
        self.composition = {
            'proton': ['u', 'u', 'd'],
            'neutron': ['u', 'd', 'd'],
            'pi+': ['u', 'anti_d'],
            'pi-': ['d', 'anti_u'],
            'K+': ['u', 'anti_s'],
            'K0': ['d', 'anti_s'],
            'K-': ['s', 'anti_u'],
            
            # Смешанные состояния (будут вычисляться через матрицу смешивания)
            'pi0': ['mixed'],
            'eta': ['mixed'],
            'eta_prime': ['mixed'],
            
            'Lambda0': ['u', 'd', 's'],
            'Sigma+': ['u', 'u', 's'],
        }
        
        # Базовые параметры из v23 (успешные)
        self.params = {
            # Эффективные массы (из v23)
            'm_u': 2.671,
            'm_d': 3.181,
            'm_s_meson': 6.795,
            'm_s_baryon': 7.200,
            
            # Топологические коэффициенты (из v23)
            'alpha': 1.001118,
            'beta': 0.202976,
            'gamma': 0.142900,
            
            # Знаки энергии связи (из v23)
            'sign_baryon': 0.500336,
            'sign_meson': -1.385960,
            
            # Коэффициент нейтрона (из v23)
            'k_neutron': 0.749951,
            
            # Электромагнитные поправки (из v23)
            'delta_em_charged': 0.010,
            'delta_em_neutral': 0.002,
            
            # НОВЫЕ ПАРАМЕТРЫ ДЛЯ КВАНТОВОГО СМЕШИВАНИЯ
            
            # 1. Диагональные элементы массовой матрицы
            'M_uu': 0.0,   # Масса базового состояния |uū⟩
            'M_dd': 0.0,   # Масса базового состояния |dđ⟩  
            'M_ss': 0.0,   # Масса базового состояния |sš⟩
            
            # 2. Недиагональные элементы (смешивание)
            'mixing_ud': 0.0,     # Смешивание uū ↔ dđ
            'mixing_us': 0.0,     # Смешивание uū ↔ sš
            'mixing_ds': 0.0,     # Смешивание dđ ↔ sš
            
            # 3. Углы смешивания (альтернативная параметризация)
            'theta_pi': 0.0,      # Угол смешивания для псевдоскалярных мезонов
            'theta_eta': 0.0,     # Угол для η-η' смешивания
            
            # 4. Поправки для странных барионов
            'lambda_s_baryon': 0.9,
        }
    
    def calculate_base_mass_diagonal(self, state):
        """Базовая масса для диагональных элементов (без смешивания)"""
        if state == 'uu':
            return 2 * self.params['m_u']
        elif state == 'dd':
            return 2 * self.params['m_d']
        elif state == 'ss':
            return 2 * self.params['m_s_meson']
        else:
            return 0.0
    
    def calculate_mass_matrix(self):
        """Построение массовой матрицы 3×3 в базисе |uū⟩, |dđ⟩, |sš⟩"""
        # Диагональные элементы
        M_uu = self.params['M_uu'] if self.params['M_uu'] != 0.0 else self.calculate_base_mass_diagonal('uu')
        M_dd = self.params['M_dd'] if self.params['M_dd'] != 0.0 else self.calculate_base_mass_diagonal('dd')
        M_ss = self.params['M_ss'] if self.params['M_ss'] != 0.0 else self.calculate_base_mass_diagonal('ss')
        
        # Применяем топологическую энергию (мезонный знак -)
        coupling = self.calculate_coupling_meson()
        M_uu -= coupling
        M_dd -= coupling
        M_ss -= coupling
        
        # Матрица 3×3
        matrix = np.array([
            [M_uu, self.params['mixing_ud'], self.params['mixing_us']],
            [self.params['mixing_ud'], M_dd, self.params['mixing_ds']],
            [self.params['mixing_us'], self.params['mixing_ds'], M_ss]
        ])
        
        return matrix * self.scale  # Переводим в МэВ
    
    def calculate_coupling_meson(self):
        """Топологическая энергия для мезонов (общая для всех диагональных элементов)"""
        # Базовые топологические параметры для мезонов
        L = 3.0
        T = 0.0
        S = 1.0
        
        coupling = (self.params['alpha'] * L + 
                    self.params['beta'] * T + 
                    self.params['gamma'] * S)
        
        return coupling
    
    def diagonalize_mass_matrix(self):
        """Диагонализация массовой матрицы, возвращает массы и состояния"""
        matrix = self.calculate_mass_matrix()
        
        # Диагонализация (симметричная матрица)
        eigenvalues, eigenvectors = eigh(matrix)
        
        # Сортируем по возрастанию массы
        idx = eigenvalues.argsort()
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Физическая интерпретация:
        # 0: π⁰ (самый лёгкий)
        # 1: η (средний)
        # 2: η' (самый тяжёлый)
        return eigenvalues, eigenvectors
    
    def calculate_mixed_meson_masses(self):
        """Расчёт масс смешанных мезонов через диагонализацию"""
        eigenvalues, _ = self.diagonalize_mass_matrix()
        
        # Назначаем физические частицы
        masses = {
            'pi0': eigenvalues[0],      # Самый лёгкий
            'eta': eigenvalues[1],      # Средний
            'eta_prime': eigenvalues[2] # Самый тяжёлый
        }
        
        return masses
    
    def calculate_pure_meson_mass(self, particle):
        """Масса для чистых (несмешанных) мезонов"""
        if particle == 'pi+':
            base = self.params['m_u'] + self.params['m_d']
            coupling = self.calculate_coupling_meson()
            mass = (base + self.params['sign_meson'] * coupling) * self.scale
            mass *= (1.0 + self.params['delta_em_charged'])
            return mass
        
        elif particle == 'pi-':
            # Симметрично π⁺
            base = self.params['m_u'] + self.params['m_d']
            coupling = self.calculate_coupling_meson()
            mass = (base + self.params['sign_meson'] * coupling) * self.scale
            mass *= (1.0 + self.params['delta_em_charged'])
            return mass
        
        elif particle == 'K+':
            base = self.params['m_u'] + self.params['m_s_meson']
            coupling = self.calculate_coupling_meson()
            mass = (base + self.params['sign_meson'] * coupling) * self.scale
            mass *= (1.0 + self.params['delta_em_charged'])
            return mass
        
        elif particle == 'K0':
            base = self.params['m_d'] + self.params['m_s_meson']
            coupling = self.calculate_coupling_meson()
            mass = (base + self.params['sign_meson'] * coupling) * self.scale
            mass *= (1.0 + self.params['delta_em_neutral'])
            # Дополнительная поправка для K⁰
            mass *= 1.005  # K⁰ немного тяжелее K⁺
            return mass
        
        elif particle == 'K-':
            base = self.params['m_s_meson'] + self.params['m_u']
            coupling = self.calculate_coupling_meson()
            mass = (base + self.params['sign_meson'] * coupling) * self.scale
            mass *= (1.0 + self.params['delta_em_charged'])
            return mass
        
        else:
            return 0.0
    
    def calculate_baryon_mass(self, particle):
        """Масса для барионов"""
        if particle == 'proton':
            base = 2 * self.params['m_u'] + self.params['m_d']
            L, T, S = 2.0, 0.0, 1.0  # Топология протона
            coupling = (self.params['alpha'] * L + 
                       self.params['beta'] * T + 
                       self.params['gamma'] * S)
            mass = (base + self.params['sign_baryon'] * coupling) * self.scale
            mass *= (1.0 + self.params['delta_em_charged'])
            return mass
        
        elif particle == 'neutron':
            base = self.params['m_u'] + 2 * self.params['m_d']
            L, T, S = 1.0, 0.0, 0.9  # Топология нейтрона
            coupling = (self.params['alpha'] * L + 
                       self.params['beta'] * T + 
                       self.params['gamma'] * S)
            mass = (base + self.params['sign_baryon'] * coupling) * self.scale
            mass *= self.params['k_neutron']  # Коррекция нейтрона
            mass *= (1.0 + self.params['delta_em_neutral'])
            return mass
        
        elif particle == 'Lambda0':
            base = self.params['m_u'] + self.params['m_d'] + self.params['m_s_baryon'] * self.params['lambda_s_baryon']
            L, T, S = 2.0, 0.4, 1.1  # Топология Λ⁰
            coupling = (self.params['alpha'] * L + 
                       self.params['beta'] * T + 
                       self.params['gamma'] * S)
            mass = (base + self.params['sign_baryon'] * coupling) * self.scale
            mass *= (1.0 + self.params['delta_em_neutral'])
            return mass
        
        elif particle == 'Sigma+':
            base = 2 * self.params['m_u'] + self.params['m_s_baryon'] * self.params['lambda_s_baryon']
            L, T, S = 2.0, 0.5, 1.0  # Топология Σ⁺
            coupling = (self.params['alpha'] * L + 
                       self.params['beta'] * T + 
                       self.params['gamma'] * S)
            mass = (base + self.params['sign_baryon'] * coupling) * self.scale
            mass *= (1.0 + self.params['delta_em_charged'])
            return mass
        
        else:
            return 0.0
    
    def calculate_mass(self, particle):
        """Основная функция расчёта массы"""
        if particle in ['pi0', 'eta', 'eta_prime']:
            # Смешанные состояния
            mixed_masses = self.calculate_mixed_meson_masses()
            return mixed_masses[particle]
        
        elif particle in ['pi+', 'pi-', 'K+', 'K0', 'K-']:
            # Чистые мезоны
            return self.calculate_pure_meson_mass(particle)
        
        elif particle in ['proton', 'neutron', 'Lambda0', 'Sigma+']:
            # Барионы
            return self.calculate_baryon_mass(particle)
        
        else:
            return 0.0
    
    def error_function(self, params_array):
        """Функция ошибки с приоритетом на смешанные состояния"""
        # Параметры для оптимизации
        param_names = [
            # Основные из v23 (фиксируем идеальные)
            'm_u', 'm_d', 'm_s_meson', 'm_s_baryon',
            'alpha', 'beta', 'gamma',
            'sign_baryon', 'sign_meson', 'k_neutron',
            'delta_em_charged', 'delta_em_neutral',
            'lambda_s_baryon',
            
            # Новые параметры смешивания
            'M_uu', 'M_dd', 'M_ss',
            'mixing_ud', 'mixing_us', 'mixing_ds',
        ]
        
        for i, name in enumerate(param_names):
            self.params[name] = params_array[i]
        
        total_error = 0.0
        
        # ВЕСА (приоритет на смешанные состояния)
        weights = {
            # Смешанные состояния (МАКСИМАЛЬНЫЙ ПРИОРИТЕТ)
            'pi0': 20.0,
            'eta': 15.0,
            'eta_prime': 10.0,
            
            # Идеальные из v23 (сохраняем точность)
            'proton': 10.0,
            'neutron': 10.0,
            'pi+': 8.0,
            'pi-': 8.0,
            'K+': 5.0,
            'K0': 5.0,
            'K-': 5.0,
            
            # Странные барионы
            'Lambda0': 3.0,
            'Sigma+': 3.0,
        }
        
        # Вычисляем ошибки
        for particle, target in self.targets.items():
            try:
                mass = self.calculate_mass(particle)
                weight = weights.get(particle, 1.0)
                
                # Комбинированная ошибка
                rel_error = abs(mass - target) / target
                abs_error = abs(mass - target) / 1000.0
                
                error = weight * (rel_error + abs_error)
                total_error += error
            except:
                total_error += 1000.0  # Большой штраф за ошибки
        
        # ЖЁСТКИЕ ФИЗИЧЕСКИЕ ОГРАНИЧЕНИЯ
        
        # 1. Разность масс n-p = 1.293 ± 0.1 МэВ
        try:
            mass_n = self.calculate_mass('neutron')
            mass_p = self.calculate_mass('proton')
            diff_np = mass_n - mass_p
            if abs(diff_np - 1.293) > 0.2:
                total_error += 50.0 * abs(diff_np - 1.293)
        except:
            total_error += 100.0
        
        # 2. π⁰ должен быть легче π⁺
        try:
            if self.calculate_mass('pi0') >= self.calculate_mass('pi+'):
                total_error += 100.0
        except:
            total_error += 50.0
        
        # 3. η должен быть тяжелее π⁰, но легче η'
        try:
            pi0_mass = self.calculate_mass('pi0')
            eta_mass = self.calculate_mass('eta')
            etap_mass = self.calculate_mass('eta_prime')
            
            if not (pi0_mass < eta_mass < etap_mass):
                total_error += 100.0
        except:
            total_error += 50.0
        
        # 4. Собственные значения матрицы должны быть положительными
        try:
            eigenvalues, _ = self.diagonalize_mass_matrix()
            if any(eig < 0 for eig in eigenvalues):
                total_error += 200.0
        except:
            total_error += 100.0
        
        return total_error

def run_v24_optimization():
    """Оптимизация v24 с фокусом на квантовом смешивании"""
    print("="*80)
    print("МОДЕЛЬ v24 — КВАНТОВО-ТОПОЛОГИЧЕСКАЯ МОДЕЛЬ СМЕШАННЫХ СОСТОЯНИЙ")
    print("="*80)
    
    model = QuantumTopologicalModelV24()
    
    # Начальные параметры (из v23 + начальные оценки для смешивания)
    initial_params = np.array([
        # Основные из v23
        2.671,      # m_u
        3.181,      # m_d
        6.795,      # m_s_meson
        7.200,      # m_s_baryon
        
        1.001118,   # alpha
        0.202976,   # beta
        0.142900,   # gamma
        
        0.500336,   # sign_baryon
        -1.385960,  # sign_meson
        0.749951,   # k_neutron
        
        0.010,      # delta_em_charged
        0.002,      # delta_em_neutral
        0.9,        # lambda_s_baryon
        
        # Параметры смешивания (начальные оценки)
        2.0,        # M_uu (должна давать ~135 МэВ после смешивания)
        2.0,        # M_dd
        6.0,        # M_ss (должна давать ~500-600 МэВ)
        
        0.5,        # mixing_ud (сильное смешивание uū-dđ)
        0.1,        # mixing_us (слабое смешивание uū-sš)
        0.1,        # mixing_ds (слабое смешивание dđ-sš)
    ])
    
    # Границы параметров
    bounds = [
        # Основные (узкие границы для сохранения успеха v23)
        (2.6, 2.8), (3.1, 3.3), (6.5, 7.0), (7.0, 7.5),
        (0.9, 1.1), (0.1, 0.3), (0.1, 0.2),
        (0.4, 0.6), (-1.5, -1.2), (0.7, 0.8),
        (0.005, 0.015), (0.0, 0.005), (0.8, 1.0),
        
        # Параметры смешивания (широкие границы)
        (1.0, 3.0), (1.0, 3.0), (4.0, 8.0),  # M_uu, M_dd, M_ss
        (-1.0, 1.0), (-0.5, 0.5), (-0.5, 0.5),  # mixing
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
    iterations = 1500000
    
    start_time = time.time()
    
    print(f"Начальная ошибка: {current_error:.4f}")
    print("Запуск оптимизации...")
    
    for i in range(iterations):
        # Мутация (более агрессивная для параметров смешивания)
        new_params = current_params.copy()
        
        for j in range(len(new_params)):
            if random.random() < 0.3:
                min_val, max_val = bounds[j]
                range_width = max_val - min_val
                
                # Разные шаги для разных групп параметров
                if j < 13:  # Основные параметры
                    step = range_width * 0.01
                else:  # Параметры смешивания
                    step = range_width * 0.05
                
                mutation = random.uniform(-step, step) * temperature
                new_val = current_params[j] + mutation
                
                # Жёсткие границы
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
            
            # Вычисляем ключевые массы для мониторинга
            try:
                model.error_function(best_params)
                pi0_mass = model.calculate_mass('pi0')
                eta_mass = model.calculate_mass('eta')
                etap_mass = model.calculate_mass('eta_prime')
                
                print(f"\rИтерация {i:,}/{iterations:,} ({progress:.1f}%) | "
                      f"Ошибка: {best_error:.4f} | "
                      f"π⁰: {pi0_mass:.1f} | "
                      f"η: {eta_mass:.1f} | "
                      f"η': {etap_mass:.1f}",
                      end='', flush=True)
            except:
                print(f"\rИтерация {i:,}/{iterations:,} ({progress:.1f}%) | "
                      f"Ошибка: {best_error:.4f}",
                      end='', flush=True)
    
    # Финальные результаты
    elapsed = time.time() - start_time
    print(f"\n\nОптимизация завершена за {elapsed:.1f} сек")
    print(f"Лучшая ошибка: {best_error:.6f}")
    
    return model, best_params

def evaluate_v24(model, best_params):
    """Оценка результатов v24"""
    # Обновляем параметры
    param_names = [
        'm_u', 'm_d', 'm_s_meson', 'm_s_baryon',
        'alpha', 'beta', 'gamma',
        'sign_baryon', 'sign_meson', 'k_neutron',
        'delta_em_charged', 'delta_em_neutral',
        'lambda_s_baryon',
        'M_uu', 'M_dd', 'M_ss',
        'mixing_ud', 'mixing_us', 'mixing_ds',
    ]
    
    for i, name in enumerate(param_names):
        model.params[name] = best_params[i]
    
    # Вычисляем все массы
    results = {}
    for particle in model.targets:
        results[particle] = model.calculate_mass(particle)
    
    print("\n" + "="*80)
    print("РЕЗУЛЬТАТЫ v24 — КВАНТОВАЯ ТОПОЛОГИЯ")
    print("="*80)
    
    # Массы частиц
    print(f"\n{'Частица':<12} {'Расчёт':<10} {'Цель':<10} {'Ошибка %':<10} {'Δ (МэВ)':<10} {'Статус':<6}")
    print("-"*70)
    
    total_error = 0
    for particle, target in model.targets.items():
        mass = results[particle]
        error = abs(mass - target) / target * 100
        delta = mass - target
        total_error += error
        
        if error < 1.0:
            status = "✓✓✓"
        elif error < 3.0:
            status = "✓✓"
        elif error < 10.0:
            status = "✓"
        else:
            status = "⚠"
        
        print(f"{particle:<12} {mass:<10.1f} {target:<10.1f} {error:<10.2f} {delta:<10.1f} {status:<6}")
    
    avg_error = total_error / len(model.targets)
    print(f"\nСредняя ошибка: {avg_error:.2f}%")
    
    # Анализ смешанных состояний
    print(f"\nАНАЛИЗ СМЕШАННЫХ СОСТОЯНИЙ:")
    
    try:
        eigenvalues, eigenvectors = model.diagonalize_mass_matrix()
        print(f"Собственные значения массовой матрицы:")
        print(f"  λ₁ (π⁰): {eigenvalues[0]:.1f} МэВ")
        print(f"  λ₂ (η): {eigenvalues[1]:.1f} МэВ")
        print(f"  λ₃ (η'): {eigenvalues[2]:.1f} МэВ")
        
        print(f"\nСобственные векторы (состав):")
        basis = ['uū', 'dđ', 'sš']
        for i in range(3):
            vec = eigenvectors[:, i]
            print(f"  Состояние {i+1}: ", end="")
            for j in range(3):
                if abs(vec[j]) > 0.1:
                    print(f"{basis[j]}: {vec[j]:.3f}  ", end="")
            print()
    except Exception as e:
        print(f"Ошибка при анализе: {e}")
    
    # Ключевые проверки
    print(f"\nКЛЮЧЕВЫЕ ПРОВЕРКИ:")
    
    try:
        # 1. Разность масс n-p
        diff_np = results['neutron'] - results['proton']
        np_status = "✓✓✓" if abs(diff_np - 1.293) < 0.1 else "✓✓" if abs(diff_np - 1.293) < 0.5 else "⚠"
        print(f"  {np_status} Разность n-p: {diff_np:.3f} МэВ (цель 1.293)")
        
        # 2. Разность K⁰-K⁺
        diff_K = results['K0'] - results['K+']
        print(f"  {'✓' if 3 < diff_K < 5 else '⚠'} K⁰-K⁺: {diff_K:.3f} МэВ (цель 4.0)")
        
        # 3. Иерархия смешанных мезонов
        if results['pi0'] < results['eta'] < results['eta_prime']:
            print(f"  ✓ Иерархия: π⁰ < η < η'")
        else:
            print(f"  ⚠ Иерархия нарушена!")
    except:
        print("  Ошибка при проверках")
    
    # Эффективные массы кварков в МэВ
    m_u_mev = model.params['m_u'] * 100
    m_d_mev = model.params['m_d'] * 100
    m_s_meson_mev = model.params['m_s_meson'] * 100
    m_s_baryon_mev = model.params['m_s_baryon'] * 100
    
    print(f"\nЭФФЕКТИВНЫЕ МАССЫ КВАРКОВ:")
    print(f"  u: {m_u_mev:.1f} МэВ")
    print(f"  d: {m_d_mev:.1f} МэВ")
    print(f"  s (мезоны): {m_s_meson_mev:.1f} МэВ")
    print(f"  s (барионы): {m_s_baryon_mev:.1f} МэВ")
    print(f"  Отношение m_d/m_u: {m_d_mev/m_u_mev:.3f}")
    print(f"  Отношение s/u (мезоны): {m_s_meson_mev/m_u_mev:.3f}")
    
    # Параметры смешивания
    print(f"\nПАРАМЕТРЫ СМЕШИВАНИЯ:")
    print(f"  M_uu: {model.params['M_uu']:.3f}")
    print(f"  M_dd: {model.params['M_dd']:.3f}")
    print(f"  M_ss: {model.params['M_ss']:.3f}")
    print(f"  mixing_ud: {model.params['mixing_ud']:.3f}")
    print(f"  mixing_us: {model.params['mixing_us']:.3f}")
    print(f"  mixing_ds: {model.params['mixing_ds']:.3f}")
    
    # Сохранение результатов
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"v24_results_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    
    results_data = {
        'model': 'v24_quantum_topological',
        'timestamp': datetime.now().isoformat(),
        'error': float(best_error),
        'parameters': {name: float(model.params[name]) for name in param_names},
        'masses': {p: float(results[p]) for p in results},
        'mass_matrix': {
            'eigenvalues': [float(e) for e in eigenvalues] if 'eigenvalues' in locals() else [],
            'eigenvectors': eigenvectors.tolist() if 'eigenvectors' in locals() else []
        },
        'quark_masses_mev': {
            'u': float(m_u_mev),
            'd': float(m_d_mev),
            's_meson': float(m_s_meson_mev),
            's_baryon': float(m_s_baryon_mev)
        }
    }
    
    with open(f"{results_dir}/v24_results.json", 'w') as f:
        json.dump(results_data, f, indent=2)
    
    print(f"\nРезультаты сохранены в {results_dir}/")
    
    return results

if __name__ == "__main__":
    # Запуск оптимизации
    model, best_params = run_v24_optimization()
    
    # Оценка результатов
    results = evaluate_v24(model, best_params)