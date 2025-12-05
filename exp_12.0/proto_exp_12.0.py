import numpy as np
import json
from datetime import datetime
import os

class TopologicalHadronModel:
    def __init__(self):
        # Топологические инварианты для каждой частицы (пока гипотетические)
        # L = число зацеплений, T = кручение, S = индекс синхронизации
        self.topology = {
            'proton':  {'L': 2, 'T': 0.1, 'S': 1.0},
            'neutron': {'L': 1, 'T': 0.1, 'S': 0.8},
            'pi+':     {'L': 3, 'T': 0.0, 'S': 1.0},
            'pi0':     {'L': 3, 'T': 0.0, 'S': 0.9},
            'pi-':     {'L': 3, 'T': 0.0, 'S': 1.0},
            'K+':      {'L': 3, 'T': 0.5, 'S': 1.0},
            'K0':      {'L': 3, 'T': 0.5, 'S': 0.9},
            'K-':      {'L': 3, 'T': 0.5, 'S': 1.0},
            'eta':     {'L': 4, 'T': 0.2, 'S': 1.2},
            'Lambda0': {'L': 2, 'T': 0.3, 'S': 1.1},
            'Sigma+':  {'L': 2, 'T': 0.4, 'S': 1.0},
            'Sigma0':  {'L': 2, 'T': 0.4, 'S': 0.9},
            'Sigma-':  {'L': 2, 'T': 0.4, 'S': 1.0},
            'Xi0':     {'L': 2, 'T': 0.6, 'S': 1.2},
            'Xi-':     {'L': 2, 'T': 0.6, 'S': 1.1},
            'Omega-':  {'L': 3, 'T': 0.8, 'S': 1.3}
        }
        
        # Целевые массы (МэВ)
        self.target_masses = {
            'proton': 938.272,
            'neutron': 939.565,
            'pi+': 139.570,
            'pi0': 134.9768,
            'pi-': 139.570,
            'K+': 493.677,
            'K0': 497.611,
            'K-': 493.677,
            'eta': 547.862,
            'Lambda0': 1115.683,
            'Sigma+': 1189.37,
            'Sigma0': 1192.642,
            'Sigma-': 1197.449,
            'Xi0': 1314.86,
            'Xi-': 1321.71,
            'Omega-': 1672.45
        }
        
        # Состав частиц
        self.composition = {
            'proton': ['u', 'u', 'd'],
            'neutron': ['u', 'd', 'd'],
            'pi+': ['u', 'anti_d'],
            'pi0': ['u', 'anti_u'],  # упрощённо
            'pi-': ['d', 'anti_u'],
            'K+': ['u', 'anti_s'],
            'K0': ['d', 'anti_s'],
            'K-': ['s', 'anti_u'],
            'eta': ['u', 'anti_u', 'd', 'anti_d', 's', 'anti_s'],  # смесь
            'Lambda0': ['u', 'd', 's'],
            'Sigma+': ['u', 'u', 's'],
            'Sigma0': ['u', 'd', 's'],
            'Sigma-': ['d', 'd', 's'],
            'Xi0': ['u', 's', 's'],
            'Xi-': ['d', 's', 's'],
            'Omega-': ['s', 's', 's']
        }
        
        # Начальные параметры
        self.params = {
            'm_u': 2.176,
            'm_d': 3.492,
            'm_s': 9.5,
            'alpha': 1.4,
            'beta': 0.5,
            'gamma': 0.3,
            'scale': 100.0
        }
        
        # Диапазоны для оптимизации
        self.ranges = {
            'm_u': (1.5, 3.0),
            'm_d': (2.5, 5.0),
            'm_s': (7.0, 12.0),
            'alpha': (0.5, 3.0),
            'beta': (0.0, 2.0),
            'gamma': (0.0, 2.0)
        }
    
    def effective_mass(self, quark):
        """Эффективная масса кварка"""
        if quark in ['u', 'anti_u']:
            return self.params['m_u']
        elif quark in ['d', 'anti_d']:
            return self.params['m_d']
        elif quark in ['s', 'anti_s']:
            return self.params['m_s']
        else:
            return 0.0
    
    def calculate_mass(self, particle):
        """Расчёт массы частицы"""
        comp = self.composition[particle]
        topo = self.topology[particle]
        
        # Базовая масса
        B = sum(self.effective_mass(q) for q in comp)
        
        # Топологическая энергия
        F = (self.params['alpha'] * topo['L'] + 
             self.params['beta'] * topo['T'] + 
             self.params['gamma'] * topo['S'])
        
        # Знак: + для барионов, - для мезонов
        if len(comp) == 3:  # барион
            M = (B + F) * self.params['scale']
        else:  # мезон
            M = (B - F) * self.params['scale']
        
        return max(M, 1.0)  # масса не может быть отрицательной
    
    def evaluate_all(self):
        """Оценка всех частиц"""
        results = {}
        for particle in self.target_masses:
            results[particle] = self.calculate_mass(particle)
        return results
    
    def error_function(self):
        """Функция ошибки"""
        total_error = 0.0
        results = self.evaluate_all()
        
        for particle, target in self.target_masses.items():
            calculated = results[particle]
            rel_error = abs(calculated - target) / target
            total_error += rel_error ** 2
        
        # Штрафы за нефизичность
        if self.params['m_d'] <= self.params['m_u']:
            total_error += 10.0
        if self.params['m_s'] <= self.params['m_d']:
            total_error += 10.0
        
        return total_error, results
    
    def optimize(self, iterations=500000, initial_temp=2.0, cooling_rate=0.99999):
        """Метод отжига для оптимизации параметров"""
        import random
        import math
        
        current_params = self.params.copy()
        current_error, current_results = self.error_function()
        
        best_params = current_params.copy()
        best_error = current_error
        best_results = current_results
        
        temp = initial_temp
        
        for i in range(iterations):
            # Генерация нового решения
            new_params = current_params.copy()
            for key in self.params:
                if key in self.ranges:
                    min_val, max_val = self.ranges[key]
                    step = (max_val - min_val) * 0.1 * temp
                    new_val = current_params[key] + random.uniform(-step, step)
                    # Отражающие границы
                    while new_val < min_val or new_val > max_val:
                        if new_val < min_val:
                            new_val = 2 * min_val - new_val
                        if new_val > max_val:
                            new_val = 2 * max_val - new_val
                    new_params[key] = new_val
            
            # Оценка
            self.params = new_params
            new_error, new_results = self.error_function()
            
            # Метрополис
            delta = new_error - current_error
            if delta < 0 or random.random() < math.exp(-delta / temp):
                current_params = new_params.copy()
                current_error = new_error
                current_results = new_results
            
            # Обновление лучшего
            if new_error < best_error:
                best_params = new_params.copy()
                best_error = new_error
                best_results = new_results
            
            # Охлаждение
            temp *= cooling_rate
            
            if i % 50000 == 0:
                print(f'Итерация {i}: ошибка {best_error:.6f}')
        
        self.params = best_params
        return best_params, best_error, best_results

def main():
    print("="*80)
    print("ТОПОЛОГИЧЕСКАЯ МОДЕЛЬ АДРОНОВ v12.0")
    print("="*80)
    
    model = TopologicalHadronModel()
    
    print("\nНачальные параметры:")
    for k, v in model.params.items():
        print(f"{k}: {v}")
    
    print("\nНачальная оценка:")
    init_results = model.evaluate_all()
    for p in ['proton', 'neutron', 'pi+', 'K+', 'Lambda0']:
        t = model.target_masses[p]
        c = init_results[p]
        err = abs(c - t) / t * 100
        print(f"{p}: {c:.3f} МэВ (цель {t:.3f}) - {err:.2f}%")
    
    print("\nЗапуск оптимизации...")
    best_params, best_error, best_results = model.optimize(
        iterations=200000,
        initial_temp=1.0,
        cooling_rate=0.99995
    )
    
    print("\n" + "="*80)
    print("ОПТИМИЗАЦИЯ ЗАВЕРШЕНА")
    print(f"Лучшая ошибка: {best_error:.6f}")
    
    print("\nОптимальные параметры:")
    for k, v in best_params.items():
        print(f"{k}: {v:.6f}")
    
    print("\nМассы частиц:")
    print(f"{'Частица':<10} {'Расчёт':<10} {'Цель':<10} {'Ошибка %':<8}")
    print("-"*45)
    
    particles = ['proton', 'neutron', 'pi+', 'pi0', 'pi-', 
                 'K+', 'K0', 'K-', 'eta', 'Lambda0']
    
    for p in particles:
        t = model.target_masses[p]
        c = best_results[p]
        err = abs(c - t) / t * 100
        print(f"{p:<10} {c:<10.3f} {t:<10.3f} {err:<8.3f}")
    
    # Сохранение результатов
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"topological_model_v12_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    
    results = {
        'parameters': best_params,
        'error': best_error,
        'masses': best_results,
        'topology': model.topology
    }
    
    with open(f"{results_dir}/results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nРезультаты сохранены в {results_dir}/")
    print("="*80)

if __name__ == "__main__":
    main()