"""
МОДЕЛЬ v9.3 - УПРОЩЕННЫЙ ГИБРИД
Сочетает простоту v6.1 с физичностью v9.x
"""

import numpy as np
import time
import json
from datetime import datetime
import os
import sys
from scipy.special import erf
from itertools import combinations
import math

# ============== КОНСТАНТЫ ==============

class QuantumConstants:
    COLOR_MATRICES = {
        'R': np.array([1, 0, 0]),
        'G': np.array([0, 1, 0]),
        'B': np.array([0, 0, 1]),
        'anti_R': np.array([-1, 0, 0]),
        'anti_G': np.array([0, -1, 0]),
        'anti_B': np.array([0, 0, -1])
    }
    
    @staticmethod
    def color_coherence(color1, color2):
        """Упрощенная цветовая когерентность"""
        vec1 = QuantumConstants.COLOR_MATRICES.get(color1, np.zeros(3))
        vec2 = QuantumConstants.COLOR_MATRICES.get(color2, np.zeros(3))
        dot = np.dot(vec1, vec2)
        # Для мезонов: цвет-антицвет дает -1, для барионов: разные цвета дают 0
        return 1.0 - abs(dot) / 3.0

# ============== МОДЕЛЬ КВАРКА ==============

class QuarkOscillatorSimple:
    def __init__(self, quark_type, params):
        self.type = quark_type
        self.anti = quark_type.startswith('anti_')
        self.base_type = quark_type.replace('anti_', '')
        
        # Параметры
        self.base_mass = params.get(f'base_mass_{self.base_type}', 2.2)
        self.frequency = params.get(f'freq_{self.base_type}', 1.0)
        self.amplitude = params.get(f'amp_{self.base_type}', 1.0)
        
        # Эффективная масса
        self.effective_mass = self.base_mass * self.frequency * self.amplitude
        
        # Заряд
        self.charge = 2/3 if self.base_type == 'u' else -1/3
        if self.anti:
            self.charge *= -1

# ============== МОДЕЛЬ АДРОНА v9.3 ==============

class HybridHadronV9_3:
    def __init__(self, name, composition, params):
        self.name = name
        self.composition = composition
        self.params = params
        self.is_meson = len(composition) == 2
        
        # Создаем кварки
        self.quarks = self._create_quarks()
        self._assign_colors()
        self._set_phases()
        
        # Вычисляем когерентности
        self.color_coherence = self._calculate_color_coherence()
        self.phase_coherence = self._calculate_phase_coherence()
        
        # Базовая масса
        self.base_mass = sum(q.effective_mass for q in self.quarks)
    
    def _create_quarks(self):
        quarks = []
        for q_type in self.composition:
            quark = QuarkOscillatorSimple(q_type, self.params)
            quarks.append(quark)
        return quarks
    
    def _assign_colors(self):
        if self.is_meson:
            # Мезон: кварк и антикварк
            self.quarks[0].color = 'R'
            self.quarks[1].color = 'anti_R'
        else:
            # Барион: три разных цвета
            colors = ['R', 'G', 'B']
            for i, quark in enumerate(self.quarks):
                quark.color = colors[i % 3]
    
    def _set_phases(self):
        if self.is_meson:
            # Мезоны: противоположные фазы
            self.phases = [0.0, np.pi]
        else:
            # Барионы: фазы для протона и нейтрона
            if self.name == 'proton':
                self.phases = [0.0, 0.0, np.pi/2]
            elif self.name == 'neutron':
                self.phases = [0.0, np.pi/2, np.pi/2]
            else:
                self.phases = [0.0, 0.0, 0.0]
    
    def _calculate_color_coherence(self):
        if self.is_meson:
            # Мезон: цвет-антицвет = максимальная когерентность
            return 1.0
        else:
            # Барион: три разных цвета = минимальная когерентность?
            # Но в цвето-нейтральном состоянии должна быть хорошая связь
            # Упростим: для цвето-нейтрального бариона = 1.0
            return 1.0
    
    def _calculate_phase_coherence(self):
        if self.is_meson:
            phase_diff = abs(self.phases[0] - self.phases[1])
            # Нормируем от 0 до 1
            return 1.0 - (phase_diff / np.pi)
        else:
            # Для барионов вычисляем среднюю когерентность
            coherences = []
            for i, j in combinations(range(3), 2):
                phase_diff = abs(self.phases[i] - self.phases[j])
                # Нормируем
                coherence = 1.0 - (phase_diff / np.pi)
                coherences.append(coherence)
            return np.mean(coherences)
    
    def calculate_sync_energy(self):
        """Энергия синхронизации (упрощенная)"""
        color_coupling = self.params.get('color_coupling', 1.0)
        phase_coupling = self.params.get('phase_coupling', 1.0)
        
        # Базовая энергия синхронизации
        base_sync = (self.color_coherence * color_coupling + 
                    self.phase_coherence * phase_coupling) / 2.0
        
        # Специфический коэффициент для частицы
        if self.name == 'proton':
            specific = self.params.get('coupling_proton', 1.0)
        elif self.name == 'neutron':
            specific = self.params.get('coupling_neutron', 0.3)
        elif self.name in ['pi+', 'pi-']:
            specific = self.params.get('coupling_meson_charged', 4.0)
        elif self.name == 'pi0':
            specific = self.params.get('coupling_meson_neutral', 3.5)
        else:
            specific = 1.0
        
        return base_sync * specific
    
    def calculate_mass(self):
        """Расчет массы (гибридная формула)"""
        # Базовая масса
        base = self.base_mass
        
        # Энергия синхронизации
        sync = self.calculate_sync_energy()
        
        # Формула: для барионов прибавляем, для мезонов вычитаем
        if self.is_meson:
            mass_value = (base - sync) * 100.0  # scale_factor = 100
        else:
            mass_value = (base + sync) * 100.0
        
        # Гарантируем положительность
        return max(mass_value, 1.0)

# ============== ОПТИМИЗАТОР v9.3 ==============

class HybridAnnealerV9_3:
    def __init__(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.result_dir = f"v93_results_{timestamp}"
        os.makedirs(self.result_dir, exist_ok=True)
        
        print(f"МОДЕЛЬ v9.3 - запуск {timestamp}")
        print("="*80)
        
        # Стартовые параметры из v6.1
        self.v61_params = {
            'base_mass_u': 2.203806,
            'base_mass_d': 4.583020,
            'freq_u': 0.956359,
            'freq_d': 0.868115,
            'amp_u': 1.032476,
            'amp_d': 0.877773,
            'coupling_proton': 1.613565,
            'coupling_neutron': 0.285395,
            'coupling_meson_charged': 4.273121,
            'coupling_meson_neutral': 3.8,
            'color_coupling': 1.0,
            'phase_coupling': 1.0
        }
        
        # Параметры для оптимизации (12 параметров)
        self.param_names = [
            'base_mass_u', 'base_mass_d',
            'freq_u', 'freq_d',
            'amp_u', 'amp_d',
            'color_coupling', 'phase_coupling',
            'coupling_proton', 'coupling_neutron',
            'coupling_meson_charged', 'coupling_meson_neutral'
        ]
        
        # Диапазоны
        self.ranges = {
            'base_mass_u': (1.5, 3.0),
            'base_mass_d': (3.0, 6.0),
            'freq_u': (0.7, 1.2),
            'freq_d': (0.7, 1.2),
            'amp_u': (0.8, 1.3),
            'amp_d': (0.7, 1.2),
            'color_coupling': (0.1, 2.0),
            'phase_coupling': (0.1, 2.0),
            'coupling_proton': (1.0, 2.5),
            'coupling_neutron': (0.1, 0.8),
            'coupling_meson_charged': (2.0, 6.0),
            'coupling_meson_neutral': (1.5, 5.0)
        }
        
        # Целевые частицы
        self.targets = {
            'proton': {'mass': 938.272, 'composition': ['u', 'u', 'd']},
            'neutron': {'mass': 939.565, 'composition': ['u', 'd', 'd']},
            'pi+': {'mass': 139.570, 'composition': ['u', 'anti_d']},
            'pi0': {'mass': 134.9768, 'composition': ['u', 'anti_u']},
            'pi-': {'mass': 139.570, 'composition': ['d', 'anti_u']},
        }
    
    def evaluate_particle(self, params, name):
        """Оценка одной частицы"""
        hadron = HybridHadronV9_3(name, self.targets[name]['composition'], params)
        return hadron.calculate_mass()
    
    def calculate_error(self, params):
        """Функция ошибки (упрощенная)"""
        total_error = 0.0
        
        # Вычисляем массы всех частиц
        masses = {}
        for name in self.targets:
            masses[name] = self.evaluate_particle(params, name)
        
        # Ошибки масс
        for name, target in self.targets.items():
            mass = masses[name]
            target_mass = target['mass']
            
            # Штраф за очень маленькую массу
            if mass < 10:
                total_error += 1000000.0
                continue
            
            # Относительная ошибка
            rel_error = abs(mass - target_mass) / target_mass
            total_error += rel_error ** 2
        
        # Ключевые штрафы
        # 1. Нейтрон > протона
        if masses['neutron'] <= masses['proton']:
            total_error += 1000.0
        
        # 2. coupling_neutron < coupling_proton
        if params.get('coupling_neutron', 0) >= params.get('coupling_proton', 1):
            total_error += 500.0
        
        # 3. Разность масс n-p
        mass_diff = abs((masses['neutron'] - masses['proton']) - 1.293)
        total_error += 100.0 * mass_diff
        
        return total_error, masses
    
    def run_annealing(self, iterations=200000, initial_temp=5.0, cooling_rate=0.99995):
        """Запуск отжига"""
        print(f"\nЗапуск отжига v9.3")
        print(f"Итераций: {iterations}")
        print("="*80)
        
        start_time = time.time()
        
        # Начальные параметры
        current_params = self.v61_params.copy()
        for param in self.param_names:
            if param in self.ranges:
                min_val, max_val = self.ranges[param]
                current_params[param] = np.random.uniform(min_val, max_val)
        
        current_error, current_masses = self.calculate_error(current_params)
        best_params = current_params.copy()
        best_error = current_error
        best_masses = current_masses
        
        temperature = initial_temp
        
        for i in range(1, iterations + 1):
            # Новая мутация
            new_params = current_params.copy()
            for param in self.param_names:
                if param in self.ranges:
                    min_val, max_val = self.ranges[param]
                    step = (max_val - min_val) * 0.1
                    mutation = np.random.normal(0, step) * temperature
                    new_val = current_params[param] + mutation
                    
                    # Отражающие границы
                    while new_val < min_val or new_val > max_val:
                        if new_val < min_val:
                            new_val = 2 * min_val - new_val
                        if new_val > max_val:
                            new_val = 2 * max_val - new_val
                    
                    new_params[param] = new_val
            
            new_error, new_masses = self.calculate_error(new_params)
            
            # Критерий Метрополиса
            if new_error < current_error:
                current_params = new_params
                current_error = new_error
                current_masses = new_masses
            else:
                delta = new_error - current_error
                prob = math.exp(-delta / temperature)
                if np.random.random() < prob:
                    current_params = new_params
                    current_error = new_error
                    current_masses = new_masses
            
            # Обновление лучшего
            if new_error < best_error:
                best_params = new_params.copy()
                best_error = new_error
                best_masses = new_masses
            
            # Охлаждение
            temperature *= cooling_rate
            
            # Прогресс
            if i % 10000 == 0:
                elapsed = time.time() - start_time
                speed = i / elapsed
                print(f"Итерация {i:6d}/{iterations} | "
                      f"Ошибка: {current_error:.3f} (лучшая: {best_error:.3f}) | "
                      f"Скорость: {speed:.1f} итер/сек")
        
        # Финальный отчет
        elapsed = time.time() - start_time
        print("\n" + "="*80)
        print("ОТЖИГ ЗАВЕРШЕН")
        print(f"Время: {elapsed:.1f} сек")
        print(f"Лучшая ошибка: {best_error:.6f}")
        print("\nЛУЧШИЕ ПАРАМЕТРЫ:")
        for param in self.param_names:
            print(f"  {param:25}: {best_params[param]:10.6f}")
        
        print("\nМАССЫ ЧАСТИЦ:")
        for name in self.targets:
            mass = best_masses[name]
            target = self.targets[name]['mass']
            error_pct = abs(mass - target) / target * 100
            status = "✓" if error_pct < 0.1 else "⚠" if error_pct < 1.0 else "✗"
            print(f"  {status} {name:6}: {mass:8.3f} МэВ (цель {target:7.3f}) - {error_pct:5.2f}%")
        
        # Разность масс n-p
        diff = best_masses['neutron'] - best_masses['proton']
        print(f"\nРазность масс n-p: {diff:.3f} МэВ (цель 1.293 МэВ)")
        
        # Сохранение результатов
        self.save_results(best_params, best_error, best_masses)
        
        return best_params, best_error, best_masses
    
    def save_results(self, params, error, masses):
        """Сохранение результатов"""
        results = {
            'model': 'v9.3',
            'timestamp': datetime.now().isoformat(),
            'error': error,
            'parameters': params,
            'masses': masses
        }
        
        filename = os.path.join(self.result_dir, "final_results.json")
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2, default=self.json_serializer)
        
        print(f"\nРезультаты сохранены в: {filename}")

    def json_serializer(self, obj):
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return str(obj)

# ============== ЗАПУСК ==============

def main():
    print("МОДЕЛЬ v9.3 - УПРОЩЕННЫЙ ГИБРИД v6.1 + v9.x")
    print("="*80)
    
    annealer = HybridAnnealerV9_3()
    
    # Быстрая оптимизация
    best_params, best_error, best_masses = annealer.run_annealing(
        iterations=200000,
        initial_temp=5.0,
        cooling_rate=0.99995
    )
    
    return best_params

if __name__ == "__main__":
    main()