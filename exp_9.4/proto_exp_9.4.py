"""
МОДЕЛЬ v9.4 - ГИБРИД С МАСШТАБИРОВАНИЕМ
Добавляем масштабные коэффициенты K_base и K_sync
"""

import numpy as np
import time
import json
from datetime import datetime
import os
import sys
import math

# ============== КЛАСС КВАРКА ==============

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
            
        # Цвет (будет установлен позже)
        self.color = None

# ============== МОДЕЛЬ АДРОНА v9.4 ==============

class HybridHadronV9_4:
    def __init__(self, name, composition, params):
        self.name = name
        self.composition = composition
        self.params = params
        self.is_meson = len(composition) == 2
        
        # Масштабные коэффициенты
        self.K_base = params.get('K_base', 1.0)
        self.K_sync = params.get('K_sync', 1.0)
        
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
            self.quarks[0].color = 'R'
            self.quarks[1].color = 'anti_R'
        else:
            colors = ['R', 'G', 'B']
            for i, quark in enumerate(self.quarks):
                quark.color = colors[i % 3]
    
    def _set_phases(self):
        if self.is_meson:
            self.phases = [0.0, np.pi]
        else:
            if self.name == 'proton':
                self.phases = [0.0, 0.0, np.pi/2]
            elif self.name == 'neutron':
                self.phases = [0.0, np.pi/2, np.pi/2]
            else:
                self.phases = [0.0, 0.0, 0.0]
    
    def _calculate_color_coherence(self):
        return 1.0  # Упрощенно
    
    def _calculate_phase_coherence(self):
        if self.is_meson:
            phase_diff = abs(self.phases[0] - self.phases[1])
            return 1.0 - (phase_diff / np.pi)
        else:
            coherences = []
            for i in range(len(self.phases)):
                for j in range(i+1, len(self.phases)):
                    phase_diff = abs(self.phases[i] - self.phases[j])
                    coherence = 1.0 - (phase_diff / np.pi)
                    coherences.append(coherence)
            return np.mean(coherences)
    
    def calculate_sync_energy(self):
        color_coupling = self.params.get('color_coupling', 1.0)
        phase_coupling = self.params.get('phase_coupling', 1.0)
        
        # Базовая энергия синхронизации
        base_sync = (self.color_coherence * color_coupling + 
                    self.phase_coherence * phase_coupling) / 2.0
        
        # Специфический коэффициент
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
        """Новая формула с масштабированием"""
        base = self.base_mass * self.K_base
        sync = self.calculate_sync_energy() * self.K_sync
        
        if self.is_meson:
            mass_value = (base - sync) * 100.0
        else:
            mass_value = (base + sync) * 100.0
        
        return max(mass_value, 1.0)

# ============== ОПТИМИЗАТОР v9.4 ==============

class HybridAnnealerV9_4:
    def __init__(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.result_dir = f"v94_results_{timestamp}"
        os.makedirs(self.result_dir, exist_ok=True)
        
        print(f"МОДЕЛЬ v9.4 - запуск {timestamp}")
        print("="*80)
        
        # Начальные параметры из v9.3
        self.start_params = {
            'base_mass_u': 1.654982,
            'base_mass_d': 3.792685,
            'freq_u': 0.917813,
            'freq_d': 1.032635,
            'amp_u': 0.891071,
            'amp_d': 0.706689,
            'color_coupling': 1.274902,
            'phase_coupling': 1.768462,
            'coupling_proton': 1.771594,
            'coupling_neutron': 0.629494,
            'coupling_meson_charged': 4.828026,
            'coupling_meson_neutral': 1.922282,
            'K_base': 1.3,  # Новые параметры
            'K_sync': 1.5
        }
        
        # Параметры для оптимизации (14 параметров)
        self.param_names = [
            'base_mass_u', 'base_mass_d',
            'freq_u', 'freq_d',
            'amp_u', 'amp_d',
            'color_coupling', 'phase_coupling',
            'coupling_proton', 'coupling_neutron',
            'coupling_meson_charged', 'coupling_meson_neutral',
            'K_base', 'K_sync'
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
            'coupling_meson_neutral': (1.5, 5.0),
            'K_base': (1.0, 1.5),  # Диапазоны для масштабов
            'K_sync': (0.5, 2.0)
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
        hadron = HybridHadronV9_4(name, self.targets[name]['composition'], params)
        return hadron.calculate_mass()
    
    def calculate_error(self, params):
        total_error = 0.0
        masses = {}
        
        for name in self.targets:
            masses[name] = self.evaluate_particle(params, name)
        
        # Ошибки масс
        for name, target in self.targets.items():
            mass = masses[name]
            target_mass = target['mass']
            
            if mass < 10:
                total_error += 1000000.0
                continue
            
            rel_error = abs(mass - target_mass) / target_mass
            total_error += rel_error ** 2
        
        # Физические ограничения
        if masses['neutron'] <= masses['proton']:
            total_error += 1000.0
        
        if params.get('coupling_neutron', 0) >= params.get('coupling_proton', 1):
            total_error += 500.0
        
        # Разность масс n-p (важный штраф)
        mass_diff = abs((masses['neutron'] - masses['proton']) - 1.293)
        total_error += 200.0 * mass_diff
        
        return total_error, masses
    
    def run_annealing(self, iterations=100000, initial_temp=3.0, cooling_rate=0.9999):
        print(f"\nЗапуск отжига v9.4")
        print(f"Итераций: {iterations}")
        print("="*80)
        
        start_time = time.time()
        
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
            # Мутация
            new_params = current_params.copy()
            for param in self.param_names:
                if param in self.ranges:
                    min_val, max_val = self.ranges[param]
                    step = (max_val - min_val) * 0.08
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
            
            # Метрополис
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
        
        # Расчет эффективных масс
        u_eff = best_params['base_mass_u'] * best_params['freq_u'] * best_params['amp_u'] * 100
        d_eff = best_params['base_mass_d'] * best_params['freq_d'] * best_params['amp_d'] * 100
        
        print(f"\nЭффективные массы кварков:")
        print(f"  u-кварк: {u_eff:.2f} МэВ")
        print(f"  d-кварк: {d_eff:.2f} МэВ")
        print(f"  Отношение m_d/m_u: {d_eff/u_eff:.3f}")
        
        print(f"\nМасштабные коэффициенты:")
        print(f"  K_base: {best_params['K_base']:.4f}")
        print(f"  K_sync: {best_params['K_sync']:.4f}")
        
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
        
        # Сохранение
        self.save_results(best_params, best_error, best_masses)
        
        return best_params, best_error, best_masses
    
    def save_results(self, params, error, masses):
        results = {
            'model': 'v9.4',
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
    print("МОДЕЛЬ v9.4 - ГИБРИД С МАСШТАБИРОВАНИЕМ")
    print("="*80)
    
    annealer = HybridAnnealerV9_4()
    
    # Быстрая оптимизация
    best_params, best_error, best_masses = annealer.run_annealing(
        iterations=100000,
        initial_temp=3.0,
        cooling_rate=0.9999
    )
    
    return best_params

if __name__ == "__main__":
    main()