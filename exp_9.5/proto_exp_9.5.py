"""
МОДЕЛЬ v9.5 - ФИЗИЧЕСКАЯ ИНТЕРПРЕТАЦИЯ v6.1
Формула v6.1 + вычисление coupling через физические когерентности
"""

import numpy as np
import time
import json
from datetime import datetime
import os
import sys
import math

class QuantumConstantsV95:
    """Квантовые константы и цветовая алгебра"""
    
    # Цветовые матрицы (упрощенные)
    COLOR_DOT_PRODUCTS = {
        ('R', 'anti_R'): -1.0,    # Кварк-антикварк максимальная связь
        ('R', 'R'): 1.0,          # Одинаковые цвета
        ('R', 'G'): 0.0,          # Разные цвета
        ('R', 'B'): 0.0,
        ('G', 'G'): 1.0,
        ('B', 'B'): 1.0,
        ('anti_R', 'anti_R'): 1.0,
    }
    
    @staticmethod
    def get_color_dot(color1, color2):
        """Скалярное произведение цветов"""
        key = (color1, color2)
        if key in QuantumConstantsV95.COLOR_DOT_PRODUCTS:
            return QuantumConstantsV95.COLOR_DOT_PRODUCTS[key]
        
        # Для всех остальных комбинаций
        return 0.0

class QuarkV95:
    """Кварк в модели v9.5"""
    
    def __init__(self, quark_type, params):
        self.type = quark_type
        self.anti = quark_type.startswith('anti_')
        self.base_type = quark_type.replace('anti_', '')
        
        # Физические параметры
        self.base_mass = params.get(f'base_mass_{self.base_type}', 2.2)
        self.frequency = params.get(f'freq_{self.base_type}', 1.0)
        self.amplitude = params.get(f'amp_{self.base_type}', 1.0)
        
        # Эффективная масса
        self.effective_mass = self.base_mass * self.frequency * self.amplitude
        
        # Заряд
        if self.base_type == 'u':
            self.charge = 2/3
        elif self.base_type == 'd':
            self.charge = -1/3
        else:
            self.charge = 0
            
        if self.anti:
            self.charge *= -1
        
        # Цвет
        self.color = None
        self.phase = 0.0

class HadronV95:
    """Адрон в модели v9.5"""
    
    def __init__(self, name, composition, params):
        self.name = name
        self.composition = composition
        self.params = params
        self.is_meson = len(composition) == 2
        
        # Создаем кварки
        self.quarks = self._create_quarks()
        self._assign_colors_and_phases()
        
        # Вычисляем когерентности
        self.color_coherence = self._calculate_color_coherence()
        self.phase_coherence = self._calculate_phase_coherence()
        
        # Базовая масса
        self.base_mass = sum(q.effective_mass for q in self.quarks)
        
        # Вычисляем coupling из физических параметров
        self.coupling = self._calculate_coupling()
    
    def _create_quarks(self):
        """Создание кварков"""
        quarks = []
        for q_type in self.composition:
            quark = QuarkV95(q_type, self.params)
            quarks.append(quark)
        return quarks
    
    def _assign_colors_and_phases(self):
        """Назначение цветов и фаз"""
        if self.is_meson:
            # Мезон: кварк и антикварк
            self.quarks[0].color = 'R'
            self.quarks[1].color = 'anti_R'
            self.quarks[0].phase = 0.0
            self.quarks[1].phase = np.pi  # Противоположная фаза
        else:
            # Барион: три разных цвета
            colors = ['R', 'G', 'B']
            for i, quark in enumerate(self.quarks):
                quark.color = colors[i % 3]
            
            # Фазы в зависимости от частицы
            if self.name == 'proton':  # uud
                self.quarks[0].phase = 0.0    # u
                self.quarks[1].phase = 0.0    # u
                self.quarks[2].phase = np.pi/2  # d
            elif self.name == 'neutron':  # udd
                self.quarks[0].phase = 0.0      # u
                self.quarks[1].phase = np.pi/2  # d
                self.quarks[2].phase = np.pi/2  # d
            else:
                for quark in self.quarks:
                    quark.phase = 0.0
    
    def _calculate_color_coherence(self):
        """Цветовая когерентность"""
        if self.is_meson:
            # Мезон: кварк-антикварк = сильная связь
            return 1.0
        else:
            # Барион: три разных цвета = цвето-нейтральная конфигурация
            # Упрощенно: хорошая когерентность
            return 0.8
    
    def _calculate_phase_coherence(self):
        """Фазовая когерентность"""
        phases = [q.phase for q in self.quarks]
        
        if len(phases) == 2:  # Мезоны
            diff = abs(phases[0] - phases[1])
            # Нормируем: противоположные фазы (π) = максимальная когерентность (1.0)
            return 1.0 - (diff / np.pi)
        else:  # Барионы
            # Средняя попарная когерентность
            coherences = []
            for i in range(len(phases)):
                for j in range(i+1, len(phases)):
                    diff = abs(phases[i] - phases[j])
                    coherence = 1.0 - (diff / np.pi)
                    coherences.append(coherence)
            return np.mean(coherences)
    
    def _calculate_coupling(self):
        """Вычисление coupling из физических параметров"""
        # Базовые коэффициенты связи
        color_strength = self.params.get('color_strength', 1.0)
        phase_strength = self.params.get('phase_strength', 1.0)
        
        # Комбинация когерентностей
        coherence = (color_strength * self.color_coherence + 
                    phase_strength * self.phase_coherence) / (color_strength + phase_strength)
        
        # Базовый coupling для типа частицы
        if self.name == 'proton':
            base_coupling = self.params.get('base_coupling_proton', 1.6)
        elif self.name == 'neutron':
            base_coupling = self.params.get('base_coupling_neutron', 0.3)
        elif self.name in ['pi+', 'pi-']:
            base_coupling = self.params.get('base_coupling_meson_charged', 4.0)
        elif self.name == 'pi0':
            base_coupling = self.params.get('base_coupling_meson_neutral', 3.5)
        else:
            base_coupling = 1.0
        
        # Итоговый coupling = базовый * когерентность
        return base_coupling * coherence
    
    def calculate_mass(self):
        """Расчет массы (формула v6.1)"""
        if self.is_meson:
            mass_value = (self.base_mass - self.coupling) * 100.0
        else:
            mass_value = (self.base_mass + self.coupling) * 100.0
        
        return max(mass_value, 1.0)

class AnnealerV95:
    """Оптимизатор для v9.5"""
    
    def __init__(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.result_dir = f"v95_results_{timestamp}"
        os.makedirs(self.result_dir, exist_ok=True)
        
        print(f"МОДЕЛЬ v9.5 - ФИЗИЧЕСКАЯ ИНТЕРПРЕТАЦИЯ v6.1")
        print("="*80)
        
        # Начальные параметры из v6.1
        self.start_params = {
            # Кварковые параметры
            'base_mass_u': 2.203806,
            'base_mass_d': 4.583020,
            'freq_u': 0.956359,
            'freq_d': 0.868115,
            'amp_u': 1.032476,
            'amp_d': 0.877773,
            
            # Физические параметры связи
            'color_strength': 1.0,
            'phase_strength': 1.0,
            
            # Базовые coupling (как в v6.1)
            'base_coupling_proton': 1.613565,
            'base_coupling_neutron': 0.285395,
            'base_coupling_meson_charged': 4.273121,
            'base_coupling_meson_neutral': 3.8,
        }
        
        self.param_names = list(self.start_params.keys())
        
        # Диапазоны параметров
        self.ranges = {
            'base_mass_u': (1.5, 3.0),
            'base_mass_d': (3.0, 6.0),
            'freq_u': (0.7, 1.2),
            'freq_d': (0.7, 1.2),
            'amp_u': (0.8, 1.3),
            'amp_d': (0.7, 1.2),
            'color_strength': (0.5, 2.0),
            'phase_strength': (0.5, 2.0),
            'base_coupling_proton': (1.0, 2.5),
            'base_coupling_neutron': (0.1, 0.8),
            'base_coupling_meson_charged': (3.0, 6.0),
            'base_coupling_meson_neutral': (2.5, 5.0),
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
        hadron = HadronV95(name, self.targets[name]['composition'], params)
        return hadron.calculate_mass()
    
    def calculate_error(self, params):
        """Функция ошибки"""
        total_error = 0.0
        masses = {}
        
        # Вычисляем массы
        for name in self.targets:
            masses[name] = self.evaluate_particle(params, name)
        
        # Штрафы за отклонения
        for name, target in self.targets.items():
            mass = masses[name]
            target_mass = target['mass']
            
            # Сильный штраф за отрицательные или очень маленькие массы
            if mass < 10:
                total_error += 1000000.0
                continue
            
            # Квадратичная ошибка
            rel_error = abs(mass - target_mass) / target_mass
            total_error += rel_error ** 2
        
        # Физические ограничения
        if masses['neutron'] <= masses['proton']:
            total_error += 1000.0
        
        if params.get('base_coupling_neutron', 0) >= params.get('base_coupling_proton', 1):
            total_error += 500.0
        
        # Разность масс n-p (очень важна!)
        mass_diff = abs((masses['neutron'] - masses['proton']) - 1.293)
        total_error += 200.0 * mass_diff
        
        # Проверка на разумность параметров
        u_eff = params['base_mass_u'] * params['freq_u'] * params['amp_u'] * 100
        d_eff = params['base_mass_d'] * params['freq_d'] * params['amp_d'] * 100
        ratio = d_eff / u_eff if u_eff > 0 else 1.0
        
        if ratio < 1.3 or ratio > 2.2:
            total_error += 100.0 * abs(ratio - 1.6)
        
        return total_error, masses
    
    def run_annealing(self, iterations=200000, initial_temp=5.0, cooling_rate=0.99995):
        """Запуск отжига"""
        print(f"\nЗапуск отжига v9.5")
        print(f"Итераций: {iterations}")
        print("="*80)
        
        start_time = time.time()
        
        # Инициализация параметров
        current_params = self.start_params.copy()
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
            # Мутация параметров
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
            
            # Оценка новой точки
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
            
            # Обновление лучшего результата
            if new_error < best_error:
                best_params = new_params.copy()
                best_error = new_error
                best_masses = new_masses
            
            # Охлаждение
            temperature *= cooling_rate
            
            # Вывод прогресса
            if i % 20000 == 0:
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
        
        # Расчет эффективных масс
        u_eff = best_params['base_mass_u'] * best_params['freq_u'] * best_params['amp_u'] * 100
        d_eff = best_params['base_mass_d'] * best_params['freq_d'] * best_params['amp_d'] * 100
        
        print(f"\nЭффективные массы кварков:")
        print(f"  u-кварк: {u_eff:.2f} МэВ")
        print(f"  d-кварк: {d_eff:.2f} МэВ")
        print(f"  Отношение m_d/m_u: {d_eff/u_eff:.3f}")
        
        print(f"\nФизические параметры:")
        print(f"  color_strength: {best_params['color_strength']:.4f}")
        print(f"  phase_strength: {best_params['phase_strength']:.4f}")
        
        print("\nМАССЫ ЧАСТИЦ:")
        total_error = 0
        for name in self.targets:
            mass = best_masses[name]
            target = self.targets[name]['mass']
            error_pct = abs(mass - target) / target * 100
            total_error += error_pct
            status = "✓" if error_pct < 0.1 else "⚠" if error_pct < 1.0 else "✗"
            print(f"  {status} {name:6}: {mass:8.3f} МэВ (цель {target:7.3f}) - {error_pct:5.2f}%")
        
        avg_error = total_error / len(self.targets)
        print(f"\nСредняя ошибка: {avg_error:.2f}%")
        
        # Разность масс n-p
        diff = best_masses['neutron'] - best_masses['proton']
        print(f"\nРазность масс n-p: {diff:.3f} МэВ (цель 1.293 МэВ)")
        
        # Сохранение результатов
        self.save_results(best_params, best_error, best_masses)
        
        return best_params, best_error, best_masses
    
    def save_results(self, params, error, masses):
        """Сохранение результатов"""
        results = {
            'model': 'v9.5',
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

def main():
    """Основная функция"""
    print("МОДЕЛЬ v9.5 - ФИЗИЧЕСКАЯ ИНТЕРПРЕТАЦИЯ v6.1")
    print("="*80)
    
    annealer = AnnealerV95()
    
    # Оптимизация
    best_params, best_error, best_masses = annealer.run_annealing(
        iterations=200000,
        initial_temp=5.0,
        cooling_rate=0.99995
    )
    
    return best_params

if __name__ == "__main__":
    main()