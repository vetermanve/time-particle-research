"""
МОДЕЛЬ v9.1 - ОДНОПРОЦЕССНАЯ ОПТИМИЗИРОВАННАЯ ВЕРСИЯ
"""

import numpy as np
import time
import json
from datetime import datetime
import os
from scipy.special import erf
from itertools import combinations
from functools import lru_cache
import hashlib
import math

# ============== ФИЗИЧЕСКИЕ КОНСТАНТЫ ==============

class QuantumConstants:
    COLOR_MATRICES = {
        'R': np.array([1, 0, 0], dtype=np.float32),
        'G': np.array([0, 1, 0], dtype=np.float32), 
        'B': np.array([0, 0, 1], dtype=np.float32),
        'anti_R': np.array([-1, 0, 0], dtype=np.float32),
        'anti_G': np.array([0, -1, 0], dtype=np.float32),
        'anti_B': np.array([0, 0, -1], dtype=np.float32)
    }
    
    SPIN_UP = np.array([1, 0], dtype=np.float32)
    SPIN_DOWN = np.array([0, 1], dtype=np.float32)
    
    QUARK_CHARGES = {
        'u': 2/3, 'd': -1/3
    }
    
    @staticmethod
    def color_coherence(color1, color2):
        vec1 = QuantumConstants.COLOR_MATRICES.get(color1, np.zeros(3, dtype=np.float32))
        vec2 = QuantumConstants.COLOR_MATRICES.get(color2, np.zeros(3, dtype=np.float32))
        dot = np.dot(vec1, vec2)
        return np.exp(-abs(dot)).item()

# ============== МОДЕЛЬ КВАРКА (УПРОЩЕННАЯ) ==============

class QuarkOscillatorSimple:
    
    def __init__(self, quark_type, params):
        self.type = quark_type
        self.anti = quark_type.startswith('anti_')
        self.base_type = quark_type.replace('anti_', '')
        
        # Базовые параметры
        self.base_mass = params.get(f'base_mass_{self.base_type}', 2.2)
        self.frequency = params.get(f'freq_{self.base_type}', 1.0)
        self.amplitude = params.get(f'amp_{self.base_type}', 1.0)
        self.effective_mass_val = self.base_mass * self.frequency * self.amplitude
        
        # Заряд
        self.charge = QuantumConstants.QUARK_CHARGES[self.base_type]
        if self.anti:
            self.charge *= -1

# ============== ГИБРИДНАЯ МОДЕЛЬ АДРОНА (ОПТИМИЗИРОВАННАЯ) ==============

class HybridHadronResonatorSimple:
    
    # Глобальные кэши
    _color_coherence_cache = {}
    _phase_coherence_cache = {}
    _quark_cache = {}
    
    def __init__(self, name, composition, params, use_cache=True):
        self.name = name
        self.composition = composition
        self.params = params
        self.scale = params.get('scale_factor', 100.0)
        self.is_meson = len(composition) == 2
        self.is_neutral_meson = name in ['pi0']
        self.use_cache = use_cache
        
        # Создаем кварки
        self.quarks = self._create_quarks_fast()
        self._assign_colors_fast()
        self._set_phases_fast()
        
        # Предвычисленные значения
        self.color_coherence_val = self._calculate_color_coherence_fast()
        self.phase_coherence_val = self._calculate_phase_coherence_fast()
    
    def _create_quarks_fast(self):
        quarks = []
        for q_type in self.composition:
            # Используем кэш для кварков
            quark_key = (q_type, hash(str(self.params)))
            if quark_key in self._quark_cache:
                quark = self._quark_cache[quark_key]
            else:
                quark = QuarkOscillatorSimple(q_type, self.params)
                self._quark_cache[quark_key] = quark
            quarks.append(quark)
        return quarks
    
    def _assign_colors_fast(self):
        if self.is_meson:
            if 'anti' in self.quarks[0].type:
                self.quarks[0].color = 'anti_R'
                self.quarks[1].color = 'R'
            else:
                self.quarks[0].color = 'R'
                self.quarks[1].color = 'anti_R'
        else:
            colors = ['R', 'G', 'B']
            if any('anti' in q.type for q in self.quarks):
                colors = ['anti_R', 'anti_G', 'anti_B']
            np.random.shuffle(colors)
            for i, quark in enumerate(self.quarks):
                quark.color = colors[i]
    
    def _set_phases_fast(self):
        if self.is_meson:
            self.phases = np.array([0.0, np.pi], dtype=np.float32)
        else:
            if self.name == 'proton':
                self.phases = np.array([0.0, 0.0, np.pi/2], dtype=np.float32)
            elif self.name == 'neutron':
                self.phases = np.array([0.0, np.pi/2, np.pi/2], dtype=np.float32)
            else:
                self.phases = np.zeros(len(self.quarks), dtype=np.float32)
    
    def _calculate_color_coherence_fast(self):
        if self.use_cache:
            color_key = tuple(sorted([q.color for q in self.quarks]))
            if color_key in self._color_coherence_cache:
                return self._color_coherence_cache[color_key]
        
        if self.is_meson:
            result = QuantumConstants.color_coherence(
                self.quarks[0].color, self.quarks[1].color)
        else:
            coherences = []
            for i, j in combinations(range(3), 2):
                coh = QuantumConstants.color_coherence(
                    self.quarks[i].color, self.quarks[j].color)
                coherences.append(coh)
            result = np.mean(coherences).item()
        
        if self.use_cache:
            self._color_coherence_cache[color_key] = result
        return result
    
    def _calculate_phase_coherence_fast(self):
        if self.use_cache:
            phase_key = tuple(self.phases.tolist())
            if phase_key in self._phase_coherence_cache:
                return self._phase_coherence_cache[phase_key]
        
        if self.is_meson:
            phase_diff = abs(self.phases[0] - self.phases[1]) % (2*np.pi)
            phase_diff = min(phase_diff, 2*np.pi - phase_diff)
            coherence = np.cos(phase_diff + np.pi)
            result = (coherence + 1) / 2
        else:
            coherences = []
            for i, j in combinations(range(3), 2):
                phase_diff = abs(self.phases[i] - self.phases[j]) % (2*np.pi)
                phase_diff = min(phase_diff, 2*np.pi - phase_diff)
                coherence = np.cos(phase_diff)
                coherences.append((coherence + 1) / 2)
            result = np.mean(coherences).item()
        
        if self.use_cache:
            self._phase_coherence_cache[phase_key] = result
        return result
    
    def calculate_base_interaction_energy(self):
        """Оптимизированная базовая энергия взаимодействия"""
        color_energy = self.params.get('color_coupling', 1.0) * self.color_coherence_val
        phase_energy = self.params.get('phase_coupling', 1.0) * self.phase_coherence_val
        
        mass_factor = np.mean([q.effective_mass_val for q in self.quarks]).item()
        base_energy = (color_energy + phase_energy) * mass_factor
        
        if self.is_meson:
            return -base_energy
        else:
            return base_energy
    
    def calculate_specific_coupling(self):
        """Специфический коэффициент связи из v6.1"""
        if self.name == 'proton':
            return self.params.get('coupling_proton', 1.0)
        elif self.name == 'neutron':
            return self.params.get('coupling_neutron', 0.3)
        elif self.name in ['pi+', 'pi-']:
            return self.params.get('coupling_meson_charged', 4.0)
        elif self.name == 'pi0':
            return self.params.get('coupling_meson_neutral', 3.5)
        else:
            return 1.0
    
    def calculate_interaction_energy(self):
        """Оптимизированная гибридная энергия"""
        base_energy = self.calculate_base_interaction_energy()
        specific_factor = self.calculate_specific_coupling()
        
        return base_energy * specific_factor
    
    def calculate_mass(self):
        base_mass = sum(q.effective_mass_val for q in self.quarks)
        interaction = self.calculate_interaction_energy()
        
        total = base_mass + interaction
        
        quantum_fluctuations = self.params.get('quantum_noise', 0.001)
        scale = abs(quantum_fluctuations * total)
        noise = np.random.normal(0, scale)
        
        return (total + noise) * self.scale

# ============== ВЫЧИСЛИТЕЛЬНЫЙ КЛАСС С КЭШИРОВАНИЕМ ==============

class HybridCalculator:
    """Класс для вычислений с кэшированием"""
    
    def __init__(self, param_names, ranges, targets):
        self.param_names = param_names
        self.ranges = ranges
        self.targets = targets
        
        # Кэши
        self._param_prep_cache = {}
        self._particle_eval_cache = {}
        
    @lru_cache(maxsize=1000)
    def prepare_params_cached(self, params_tuple):
        """Кэшированная подготовка параметров"""
        params = dict(zip(self.param_names, params_tuple))
        params['color_coupling_meson'] = params['color_coupling'] * params['meson_coupling_scale']
        params['phase_coupling_meson'] = params['phase_coupling'] * params['meson_coupling_scale']
        params['color_coupling_baryon'] = params['color_coupling'] * params['baryon_coupling_scale']
        params['phase_coupling_baryon'] = params['phase_coupling'] * params['baryon_coupling_scale']
        return params
    
    def prepare_params(self, params):
        """Подготовка параметров"""
        params_tuple = tuple(params.get(name, 0) for name in self.param_names)
        return self.prepare_params_cached(params_tuple)
    
    def evaluate_particle_fast(self, params, particle_name, composition, is_meson):
        """Быстрая оценка частицы"""
        # Ключ для кэша
        cache_key = (hash(str(params)), particle_name)
        if cache_key in self._particle_eval_cache:
            return self._particle_eval_cache[cache_key]
        
        part_params = self.prepare_params(params)
        
        if is_meson:
            part_params['color_coupling'] = part_params.get('color_coupling_meson', part_params['color_coupling'])
            part_params['phase_coupling'] = part_params.get('phase_coupling_meson', part_params['phase_coupling'])
        else:
            part_params['color_coupling'] = part_params.get('color_coupling_baryon', part_params['color_coupling'])
            part_params['phase_coupling'] = part_params.get('phase_coupling_baryon', part_params['phase_coupling'])
        
        hadron = HybridHadronResonatorSimple(particle_name, composition, part_params, use_cache=True)
        mass = hadron.calculate_mass()
        charge = sum(q.charge for q in hadron.quarks)
        
        result = (mass, charge)
        self._particle_eval_cache[cache_key] = result
        return result
    
    def evaluate_all_particles_fast(self, params):
        """Быстрая оценка всех частиц"""
        results = {}
        for name, target in self.targets.items():
            is_meson = len(target['composition']) == 2
            mass, charge = self.evaluate_particle_fast(params, name, target['composition'], is_meson)
            results[f'{name}_mass'] = mass
            results[f'{name}_charge'] = charge
        
        # Вычисляем эффективные массы
        m_u = params['base_mass_u'] * params['freq_u'] * params['amp_u'] * params['scale_factor']
        m_d = params['base_mass_d'] * params['freq_d'] * params['amp_d'] * params['scale_factor']
        results['m_u_eff'] = m_u
        results['m_d_eff'] = m_d
        results['ratio_d_u'] = m_d / m_u if m_u > 0 else 1
        
        # Энергии связи
        results['E_proton'] = params.get('coupling_proton', 1.0)
        results['E_neutron'] = params.get('coupling_neutron', 0.3)
        results['E_meson_charged'] = params.get('coupling_meson_charged', 4.0)
        results['E_meson_neutral'] = params.get('coupling_meson_neutral', 3.5)
        results['ratio_neutron_proton'] = results['E_neutron'] / results['E_proton'] if results['E_proton'] > 0 else 0
        
        return results
    
    def calculate_error_fast(self, params):
        """Быстрая функция ошибки"""
        results = self.evaluate_all_particles_fast(params)
        total_error = 0.0
        
        weights = {
            'proton': 40.0, 'neutron': 40.0,
            'pi+': 25.0, 'pi0': 30.0, 'pi-': 25.0
        }
        
        for name, target in self.targets.items():
            mass = results[f'{name}_mass']
            target_mass = target['mass']
            
            if mass <= 0:
                total_error += 10000.0
                continue
                
            rel_error = abs(mass - target_mass) / target_mass
            total_error += weights[name] * (rel_error ** 2)
            
            if rel_error > 0.3:
                total_error += weights[name] * 10.0 * (rel_error - 0.3)
        
        # Проверки зарядов
        for name, target in self.targets.items():
            if abs(results[f'{name}_charge'] - target['charge']) > 0.001:
                total_error += 1000.0
        
        # Физические ограничения
        if results['neutron_mass'] < results['proton_mass']:
            diff = results['proton_mass'] - results['neutron_mass']
            total_error += 500.0 * diff
        
        ratio_d_u = results['ratio_d_u']
        if ratio_d_u < 1.3 or ratio_d_u > 2.0:
            penalty = abs(ratio_d_u - 1.6) * 20.0
            total_error += penalty
        
        if params.get('meson_coupling_scale', 1) < params.get('baryon_coupling_scale', 1):
            total_error += 200.0
        
        if params.get('coupling_neutron', 0) > params.get('coupling_proton', 1):
            total_error += 300.0
        
        if params.get('coupling_meson_neutral', 0) > params.get('coupling_meson_charged', 4):
            total_error += 200.0
        
        mass_diff = abs((results['neutron_mass'] - results['proton_mass']) - 1.293)
        total_error += 100.0 * mass_diff
        
        return total_error, results

# ============== ОДНОПРОЦЕССНЫЙ ОТЖИГ ==============

class HybridAnnealerSingle:
    
    def __init__(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.result_dir = f"v91_single_{timestamp}"
        os.makedirs(self.result_dir, exist_ok=True)
        
        # Параметры из v6.1
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
            'scale_factor': 100.0
        }
        
        self.param_names = [
            'base_mass_u', 'base_mass_d',
            'freq_u', 'freq_d',
            'amp_u', 'amp_d',
            'color_coupling', 'phase_coupling',
            'meson_coupling_scale', 'baryon_coupling_scale',
            'coupling_proton', 'coupling_neutron',
            'coupling_meson_charged', 'coupling_meson_neutral',
            'scale_factor'
        ]
        
        self.base_params = self.v61_params.copy()
        
        self.ranges = {
            'base_mass_u': (1.5, 3.0),
            'base_mass_d': (3.0, 6.0),
            'freq_u': (0.7, 1.2),
            'freq_d': (0.7, 1.2),
            'amp_u': (0.8, 1.3),
            'amp_d': (0.7, 1.2),
            'color_coupling': (0.5, 3.0),
            'phase_coupling': (0.5, 2.0),
            'meson_coupling_scale': (2.0, 6.0),
            'baryon_coupling_scale': (0.5, 2.0),
            'coupling_proton': (1.0, 2.5),
            'coupling_neutron': (0.1, 0.8),
            'coupling_meson_charged': (3.0, 5.0),
            'coupling_meson_neutral': (2.5, 4.5),
            'scale_factor': (90.0, 110.0)
        }
        
        self.targets = {
            'proton': {'mass': 938.272, 'charge': 1.0, 'composition': ['u', 'u', 'd']},
            'neutron': {'mass': 939.565, 'charge': 0.0, 'composition': ['u', 'd', 'd']},
            'pi+': {'mass': 139.570, 'charge': 1.0, 'composition': ['u', 'anti_d']},
            'pi0': {'mass': 134.9768, 'charge': 0.0, 'composition': ['u', 'anti_u']},
            'pi-': {'mass': 139.570, 'charge': -1.0, 'composition': ['d', 'anti_u']},
        }
        
        # Создаем калькулятор
        self.calculator = HybridCalculator(self.param_names, self.ranges, self.targets)
        
        # Статистика
        self.start_time = None
        self.iterations_done = 0
        self.total_iterations = 0
    
    def save_progress(self, iteration, total_iterations, error, temperature, params, is_checkpoint=False):
        """Сохранить прогресс"""
        if is_checkpoint:
            filename = f"checkpoint_{iteration:07d}.json"
        else:
            filename = "progress.json"
        
        progress_file = os.path.join(self.result_dir, filename)
        
        progress_data = {
            'iteration': iteration,
            'total_iterations': total_iterations,
            'progress_percent': (iteration / total_iterations) * 100,
            'current_error': error,
            'temperature': temperature,
            'params': params,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(progress_file, 'w') as f:
            json.dump(progress_data, f, indent=2)
        
        # Также сохраняем сводку
        if iteration % 10000 == 0 or iteration == total_iterations:
            self._save_summary(iteration, error, params)
    
    def _save_summary(self, iteration, error, params):
        """Сохранить сводку"""
        summary_file = os.path.join(self.result_dir, "summary.txt")
        
        with open(summary_file, 'a') as f:
            f.write(f"\n=== Итерация {iteration} ===\n")
            f.write(f"Ошибка: {error:.3f}\n")
            f.write(f"Параметры:\n")
            for name in self.param_names:
                f.write(f"  {name}: {params.get(name, 0):.6f}\n")
            f.write(f"Время: {datetime.now().strftime('%H:%M:%S')}\n")
    
    def print_progress(self, iteration, total_iterations, error, temperature, best_error):
        """Вывести прогресс на экран"""
        elapsed = time.time() - self.start_time
        progress_percent = (iteration / total_iterations) * 100
        iters_per_sec = iteration / elapsed if elapsed > 0 else 0
        
        # Оставшееся время
        if iteration > 0:
            remaining = elapsed / iteration * (total_iterations - iteration)
            remaining_str = f"{remaining:.0f} сек"
            if remaining > 60:
                remaining_str = f"{remaining/60:.1f} мин"
            if remaining > 3600:
                remaining_str = f"{remaining/3600:.1f} час"
        else:
            remaining_str = "N/A"
        
        print(f"\rПрогресс: {progress_percent:.1f}% ({iteration:,}/{total_iterations:,}) | "
              f"Ошибка: {error:.3f} (лучшая: {best_error:.3f}) | "
              f"Температура: {temperature:.3f} | "
              f"Скорость: {iters_per_sec:.1f} итер/сек | "
              f"Осталось: {remaining_str}", end="")
    
    def run_annealing(self, total_iterations=200000, initial_temp=8.0, checkpoint_interval=20000):
        """Запустить однопроцессный отжиг"""
        print("="*80)
        print("ГИБРИДНАЯ МОДЕЛЬ v9.1 - ОДНОПРОЦЕССНАЯ ВЕРСИЯ")
        print(f"Итераций: {total_iterations:,}")
        print("Особенности: кэширование, оптимизированные вычисления")
        print("="*80)
        
        self.start_time = time.time()
        self.total_iterations = total_iterations
        
        # Инициализация параметров
        current_params = self.base_params.copy()
        for param in self.param_names:
            if param in self.ranges:
                min_val, max_val = self.ranges[param]
                current_params[param] = np.random.uniform(min_val, max_val)
        
        # Начальная оценка
        current_error, current_results = self.calculator.calculate_error_fast(current_params)
        best_params = current_params.copy()
        best_error = current_error
        best_results = current_results
        
        temperature = initial_temp
        cooling_rate = 0.99997
        update_interval = max(100, total_iterations // 100)
        
        print(f"\nНачальная ошибка: {current_error:.3f}")
        print(f"Начальная температура: {temperature:.3f}")
        print(f"Сохранение чекпоинтов каждые {checkpoint_interval:,} итераций\n")
        
        try:
            for iteration in range(total_iterations):
                # Генерация новых параметров
                new_params = current_params.copy()
                
                for param in self.param_names:
                    if param in self.ranges:
                        min_val, max_val = self.ranges[param]
                        current_val = current_params[param]
                        
                        step = (max_val - min_val) * 0.05
                        mutation = np.random.normal(0, step) * temperature
                        
                        # Направленные мутации для физических ограничений
                        if param == 'coupling_neutron' and current_val > 0.5:
                            mutation -= 0.2 * step
                        elif param == 'coupling_proton' and current_val < 1.3:
                            mutation += 0.2 * step
                        elif param == 'coupling_meson_neutral' and current_val > new_params.get('coupling_meson_charged', 4):
                            mutation -= 0.3 * step
                        
                        new_val = current_val + mutation
                        
                        # Отражающие границы
                        while new_val < min_val or new_val > max_val:
                            if new_val < min_val:
                                new_val = 2 * min_val - new_val
                            if new_val > max_val:
                                new_val = 2 * max_val - new_val
                        
                        new_params[param] = new_val
                
                # Оценка новых параметров
                new_error, new_results = self.calculator.calculate_error_fast(new_params)
                
                # Принятие решения
                delta = new_error - current_error
                
                if delta < 0:
                    current_params = new_params
                    current_error = new_error
                    current_results = new_results
                else:
                    prob = np.exp(-delta / temperature)
                    if np.random.random() < prob:
                        current_params = new_params
                        current_error = new_error
                        current_results = new_results
                
                # Обновление лучшего результата
                if new_error < best_error:
                    best_params = new_params.copy()
                    best_error = new_error
                    best_results = new_results
                
                temperature *= cooling_rate
                
                # Вывод прогресса
                if iteration % update_interval == 0:
                    self.print_progress(iteration, total_iterations, current_error, temperature, best_error)
                    self.save_progress(iteration, total_iterations, current_error, temperature, current_params)
                
                # Сохранение чекпоинта
                if iteration % checkpoint_interval == 0 and iteration > 0:
                    self.save_progress(iteration, total_iterations, best_error, temperature, best_params, is_checkpoint=True)
                
                self.iterations_done = iteration + 1
            
            # Финальный вывод
            print("\n" + "="*80)
            print("ОТЖИГ ЗАВЕРШЕН")
            
        except KeyboardInterrupt:
            print("\n\nОтжиг прерван пользователем")
        
        elapsed = time.time() - self.start_time
        
        print(f"Всего итераций: {self.iterations_done:,}")
        print(f"Общее время: {elapsed:.1f} сек")
        print(f"Скорость: {self.iterations_done/elapsed:.1f} итер/сек")
        print(f"Лучшая ошибка: {best_error:.3f}")
        print("="*80)
        
        # Сохраняем финальные результаты
        self.save_final_results(best_params, best_error, best_results)
        self.print_final_report(best_params, best_error, best_results)
        
        return best_params, best_error, best_results
    
    def save_final_results(self, params, error, results):
        """Сохранить финальные результаты"""
        final_data = {
            'model': 'v9.1_single',
            'timestamp': datetime.now().isoformat(),
            'total_iterations': self.total_iterations,
            'iterations_done': self.iterations_done,
            'final_error': error,
            'params': params,
            'results': results
        }
        
        with open(os.path.join(self.result_dir, "final_results.json"), 'w') as f:
            json.dump(final_data, f, indent=2, default=self.json_serializer)
    
    def print_final_report(self, params, error, results):
        """Вывести финальный отчет"""
        print("\n" + "="*80)
        print("ФИНАЛЬНЫЙ ОТЧЕТ v9.1")
        print("="*80)
        
        print(f"\nКЛЮЧЕВЫЕ ПАРАМЕТРЫ:")
        groups = {
            'Массы кварков': ['base_mass_u', 'base_mass_d'],
            'Частоты': ['freq_u', 'freq_d'],
            'Амплитуды': ['amp_u', 'amp_d'],
            'Физические coupling': ['color_coupling', 'phase_coupling'],
            'Специфические coupling': ['coupling_proton', 'coupling_neutron',
                                      'coupling_meson_charged', 'coupling_meson_neutral']
        }
        
        for group_name, param_list in groups.items():
            print(f"  {group_name}:")
            for param in param_list:
                if param in params:
                    print(f"    {param}: {params[param]:.6f}")
        
        print(f"\nЭФФЕКТИВНЫЕ МАССЫ КВАРКОВ (МэВ):")
        print(f"  u: {results['m_u_eff']:.2f}")
        print(f"  d: {results['m_d_eff']:.2f}")
        print(f"  m_d/m_u: {results['ratio_d_u']:.2f}")
        
        print(f"\nЭНЕРГИИ СВЯЗИ:")
        print(f"  Протон: {results['E_proton']:.3f}")
        print(f"  Нейтрон: {results['E_neutron']:.3f}")
        print(f"  Отношение n/p: {results['ratio_neutron_proton']:.3f}")
        
        print(f"\nМАССЫ ЧАСТИЦ:")
        total_error = 0
        for name in self.targets.keys():
            mass = results[f'{name}_mass']
            target = self.targets[name]['mass']
            error = abs(mass - target) / target * 100
            total_error += error
            print(f"  {name}: {mass:.1f} МэВ (цель {target:.1f}) - {error:.2f}%")
        
        avg_error = total_error / len(self.targets)
        print(f"\nСредняя ошибка: {avg_error:.2f}%")
        
        diff = results['neutron_mass'] - results['proton_mass']
        print(f"\nРАЗНОСТЬ МАСС n-p:")
        print(f"  Модель: {diff:.3f} МэВ")
        print(f"  Эксперимент: 1.293 МэВ")
        print(f"  Отклонение: {abs(diff-1.293):.3f} МэВ")
        
        print(f"\nРезультаты сохранены в: {self.result_dir}")
        print("="*80)
    
    def json_serializer(self, obj):
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return str(obj)

# ============== ЗАПУСК ==============

def main():
    print("="*80)
    print("ГИБРИДНАЯ МОДЕЛЬ v9.1 - ОДНОПРОЦЕССНАЯ ВЕРСИЯ")
    print("="*80)
    
    print("\nОСОБЕННОСТИ:")
    print("  • Один процесс - нет проблем с синхронизацией")
    print("  • Кэширование вычислений для ускорения")
    print("  • Предвычисленные значения и оптимизации")
    print("  • Сохранение прогресса и чекпоинтов")
    print("  • Возможность прерывания и продолжения")
    
    print("\nПАРАМЕТРЫ ОПТИМИЗАЦИИ:")
    print("  15 параметров, 200,000 итераций")
    print("  Начальная температура: 8.0")
    print("  Скорость охлаждения: 0.99997")
    
    try:
        annealer = HybridAnnealerSingle()
        best_params, best_error, best_results = annealer.run_annealing(
            total_iterations=200000,
            initial_temp=8.0,
            checkpoint_interval=20000
        )
        
    except KeyboardInterrupt:
        print("\n\nВычисление прервано пользователем")
        print("Промежуточные результаты сохранены")
    except Exception as e:
        print(f"\nОШИБКА: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*80)
    print("ВЫЧИСЛЕНИЯ ЗАВЕРШЕНЫ")
    print("="*80)

if __name__ == "__main__":
    main()