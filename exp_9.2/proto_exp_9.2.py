"""
МОДЕЛЬ v9.2 - ОПТИМИЗИРОВАННАЯ ОДНОПРОЦЕССНАЯ ВЕРСИЯ
Исправлены проблемы с отрицательными массами мезонов
"""

import numpy as np
import time
import json
from datetime import datetime
import os
import sys
from scipy.special import erf
from itertools import combinations
from functools import lru_cache
import math

# ============== НАСТРОЙКИ ЛОГИРОВАНИЯ ==============

class Logger:
    """Класс для логирования в файл и консоль"""
    
    def __init__(self, log_dir):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(log_dir, f"log_{timestamp}.txt")
        self.console = sys.stdout
        
        # Открываем файл для записи
        self.file = open(self.log_file, 'w', encoding='utf-8')
    
    def write(self, message):
        """Записать сообщение в лог и консоль"""
        # В консоль
        self.console.write(message)
        self.console.flush()
        
        # В файл
        self.file.write(message)
        self.file.flush()
    
    def flush(self):
        """Сбросить буферы"""
        self.console.flush()
        self.file.flush()
    
    def close(self):
        """Закрыть лог"""
        self.file.close()

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
        """Вычисление цветовой когерентности"""
        vec1 = QuantumConstants.COLOR_MATRICES.get(color1, np.zeros(3, dtype=np.float32))
        vec2 = QuantumConstants.COLOR_MATRICES.get(color2, np.zeros(3, dtype=np.float32))
        dot = np.dot(vec1, vec2)
        return np.exp(-abs(dot)).item()

# ============== МОДЕЛЬ КВАРКА ==============

class QuarkOscillatorSimple:
    
    def __init__(self, quark_type, params):
        self.type = quark_type
        self.anti = quark_type.startswith('anti_')
        self.base_type = quark_type.replace('anti_', '')
        
        # Базовые параметры
        self.base_mass = params.get(f'base_mass_{self.base_type}', 2.2)
        self.frequency = params.get(f'freq_{self.base_type}', 1.0)
        self.amplitude = params.get(f'amp_{self.base_type}', 1.0)
        
        # Эффективная масса
        self.effective_mass_val = self.base_mass * self.frequency * self.amplitude
        
        # Заряд
        self.charge = QuantumConstants.QUARK_CHARGES.get(self.base_type, 0)
        if self.anti:
            self.charge *= -1

# ============== ГИБРИДНАЯ МОДЕЛЬ АДРОНА ==============

class HybridHadronResonatorV9_2:
    
    # Глобальные кэши
    _color_coherence_cache = {}
    _phase_coherence_cache = {}
    
    def __init__(self, name, composition, params, use_cache=True):
        self.name = name
        self.composition = composition
        self.params = params
        self.scale = params.get('scale_factor', 100.0)
        self.is_meson = len(composition) == 2
        self.use_cache = use_cache
        
        # Создаем кварки
        self.quarks = self._create_quarks_fast()
        self._assign_colors_fast()
        self._set_phases_fast()
        
        # Вычисляем когерентности
        self.color_coherence_val = self._calculate_color_coherence_fast()
        self.phase_coherence_val = self._calculate_phase_coherence_fast()
        
        # Вычисляем базовую массу
        self.base_mass_val = sum(q.effective_mass_val for q in self.quarks)
    
    def _create_quarks_fast(self):
        """Быстрое создание кварков"""
        quarks = []
        for q_type in self.composition:
            quark = QuarkOscillatorSimple(q_type, self.params)
            quarks.append(quark)
        return quarks
    
    def _assign_colors_fast(self):
        """Быстрое назначение цветов"""
        if self.is_meson:
            # Для мезонов: кварк и антикварк
            if 'anti' in self.quarks[0].type:
                self.quarks[0].color = 'anti_R'
                self.quarks[1].color = 'R'
            else:
                self.quarks[0].color = 'R'
                self.quarks[1].color = 'anti_R'
        else:
            # Для барионов: три разных цвета
            colors = ['R', 'G', 'B']
            if any('anti' in q.type for q in self.quarks):
                colors = ['anti_R', 'anti_G', 'anti_B']
            np.random.shuffle(colors)
            for i, quark in enumerate(self.quarks):
                quark.color = colors[i]
    
    def _set_phases_fast(self):
        """Установка фазовых соотношений"""
        if self.is_meson:
            # Для мезонов: противоположные фазы
            self.phases = np.array([0.0, np.pi], dtype=np.float32)
        else:
            # Для барионов: специфичные конфигурации
            if self.name == 'proton':
                self.phases = np.array([0.0, 0.0, np.pi/2], dtype=np.float32)
            elif self.name == 'neutron':
                self.phases = np.array([0.0, np.pi/2, np.pi/2], dtype=np.float32)
            else:
                self.phases = np.zeros(len(self.quarks), dtype=np.float32)
    
    def _calculate_color_coherence_fast(self):
        """Быстрый расчет цветовой когерентности"""
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
        """Быстрый расчет фазовой когерентности"""
        if self.use_cache:
            phase_key = tuple(self.phases.tolist())
            if phase_key in self._phase_coherence_cache:
                return self._phase_coherence_cache[phase_key]
        
        if self.is_meson:
            phase_diff = abs(self.phases[0] - self.phases[1]) % (2*np.pi)
            phase_diff = min(phase_diff, 2*np.pi - phase_diff)
            coherence = np.cos(phase_diff + np.pi)  # Для мезонов +π
            result = (coherence + 1) / 2
        else:
            coherences = []
            for i, j in combinations(range(3), 2):
                phase_diff = abs(self.phases[i] - self.phases[j]) % (2*np.pi)
                phase_diff = min(phase_diff, 2*np.pi - phase_diff)
                coherence = np.cos(phase_diff)  # Для барионов без π
                coherences.append((coherence + 1) / 2)
            result = np.mean(coherences).item()
        
        if self.use_cache:
            self._phase_coherence_cache[phase_key] = result
        return result
    
    def calculate_base_interaction_energy(self):
        """Базовая энергия взаимодействия"""
        # Коэффициенты связи
        color_coupling = self.params.get('color_coupling', 1.0)
        phase_coupling = self.params.get('phase_coupling', 1.0)
        
        # Масштабирование для мезонов/барионов
        if self.is_meson:
            meson_scale = self.params.get('meson_coupling_scale', 1.0)
            color_coupling *= meson_scale
            phase_coupling *= meson_scale
        else:
            baryon_scale = self.params.get('baryon_coupling_scale', 1.0)
            color_coupling *= baryon_scale
            phase_coupling *= baryon_scale
        
        # Энергии
        color_energy = color_coupling * self.color_coherence_val
        phase_energy = phase_coupling * self.phase_coherence_val
        
        # Фактор массы (средняя эффективная масса кварков)
        mass_factor = np.mean([q.effective_mass_val for q in self.quarks]).item()
        
        # Базовая энергия
        base_energy = (color_energy + phase_energy) * mass_factor
        
        # Для мезонов энергия связи уменьшает массу, для барионов - увеличивает
        if self.is_meson:
            return -abs(base_energy)  # Гарантируем отрицательность для мезонов
        else:
            return abs(base_energy)   # Гарантируем положительность для барионов
    
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
        """Полная энергия взаимодействия"""
        base_energy = self.calculate_base_interaction_energy()
        specific_factor = self.calculate_specific_coupling()
        
        return base_energy * specific_factor
    
    def calculate_mass(self):
        """Расчет массы частицы"""
        # Базовая масса
        base_mass = self.base_mass_val
        
        # Энергия взаимодействия
        interaction = self.calculate_interaction_energy()
        
        # Общая энергия (для барионов добавляем, для мезонов вычитаем)
        if self.is_meson:
            total = base_mass + interaction  # interaction отрицательна для мезонов
        else:
            total = base_mass + interaction  # interaction положительна для барионов
        
        # Квантовые флуктуации (маленькие)
        quantum_fluctuations = self.params.get('quantum_noise', 0.001)
        scale = abs(quantum_fluctuations * total)
        noise = np.random.normal(0, scale)
        
        # Итоговая масса
        final_mass = (total + noise) * self.scale
        
        # Гарантируем положительность массы
        return max(final_mass, 1.0)  # Минимум 1.0 МэВ

# ============== ВЫЧИСЛИТЕЛЬНЫЙ КЛАСС ==============

class HybridCalculatorV9_2:
    """Класс для вычислений v9.2"""
    
    def __init__(self, param_names, ranges, targets):
        self.param_names = param_names
        self.ranges = ranges
        self.targets = targets
        
        # Кэш
        self._particle_eval_cache = {}
    
    def prepare_params(self, params):
        """Подготовка параметров"""
        # Создаем копию параметров с дополнительными вычисленными значениями
        prepared = params.copy()
        
        # Масштабирование для мезонов и барионов
        color_coupling = params.get('color_coupling', 1.0)
        phase_coupling = params.get('phase_coupling', 1.0)
        
        meson_scale = params.get('meson_coupling_scale', 1.0)
        baryon_scale = params.get('baryon_coupling_scale', 1.0)
        
        prepared['color_coupling_meson'] = color_coupling * meson_scale
        prepared['phase_coupling_meson'] = phase_coupling * meson_scale
        prepared['color_coupling_baryon'] = color_coupling * baryon_scale
        prepared['phase_coupling_baryon'] = phase_coupling * baryon_scale
        
        return prepared
    
    def evaluate_particle(self, params, particle_name, composition, is_meson):
        """Оценка одной частицы"""
        cache_key = (particle_name, hash(str(sorted(params.items()))))
        
        if cache_key in self._particle_eval_cache:
            return self._particle_eval_cache[cache_key]
        
        prepared_params = self.prepare_params(params)
        hadron = HybridHadronResonatorV9_2(particle_name, composition, prepared_params, use_cache=True)
        
        mass = hadron.calculate_mass()
        charge = sum(q.charge for q in hadron.quarks)
        
        result = (mass, charge)
        self._particle_eval_cache[cache_key] = result
        return result
    
    def evaluate_all_particles(self, params):
        """Оценка всех частиц"""
        results = {}
        
        for name, target in self.targets.items():
            is_meson = len(target['composition']) == 2
            mass, charge = self.evaluate_particle(params, name, target['composition'], is_meson)
            results[f'{name}_mass'] = mass
            results[f'{name}_charge'] = charge
        
        # Вычисляем эффективные массы кварков
        m_u_eff = params['base_mass_u'] * params['freq_u'] * params['amp_u'] * params['scale_factor']
        m_d_eff = params['base_mass_d'] * params['freq_d'] * params['amp_d'] * params['scale_factor']
        
        results['m_u_eff'] = m_u_eff
        results['m_d_eff'] = m_d_eff
        results['ratio_d_u'] = m_d_eff / m_u_eff if m_u_eff > 0 else 1.0
        
        # Энергии связи
        results['E_proton'] = params.get('coupling_proton', 1.0)
        results['E_neutron'] = params.get('coupling_neutron', 0.3)
        results['E_meson_charged'] = params.get('coupling_meson_charged', 4.0)
        results['E_meson_neutral'] = params.get('coupling_meson_neutral', 3.5)
        results['ratio_neutron_proton'] = results['E_neutron'] / results['E_proton'] if results['E_proton'] > 0 else 0.0
        
        return results
    
    def calculate_error(self, params):
        """Расчет ошибки модели"""
        results = self.evaluate_all_particles(params)
        total_error = 0.0
        
        # Веса для разных частиц
        weights = {
            'proton': 40.0,    # Высокий вес для протона
            'neutron': 40.0,   # Высокий вес для нейтрона
            'pi+': 50.0,       # Высокий вес для заряженных пионов
            'pi0': 60.0,       # Самый высокий вес для нейтрального пиона (проблемный)
            'pi-': 50.0        # Высокий вес для заряженных пионов
        }
        
        # Ошибки масс
        for name, target in self.targets.items():
            mass = results[f'{name}_mass']
            target_mass = target['mass']
            
            # Штраф за отрицательную или очень маленькую массу
            if mass <= 10.0:  # Масса меньше 10 МэВ
                total_error += 1000000.0  # Очень большой штраф
                continue
            
            # Относительная ошибка
            rel_error = abs(mass - target_mass) / target_mass
            
            # Квадратичная ошибка с весом
            total_error += weights[name] * (rel_error ** 2)
            
            # Дополнительный штраф за большие отклонения
            if rel_error > 0.5:  # Ошибка больше 50%
                total_error += weights[name] * 100.0 * (rel_error - 0.5)
        
        # Штрафы за нарушение физических ограничений
        
        # 1. Нейтрон должен быть тяжелее протона
        if results['neutron_mass'] <= results['proton_mass']:
            total_error += 1000.0
        
        # 2. Отношение масс d/u должно быть в разумных пределах
        ratio_d_u = results['ratio_d_u']
        if ratio_d_u < 1.3 or ratio_d_u > 2.2:
            penalty = abs(ratio_d_u - 1.6) * 100.0
            total_error += penalty
        
        # 3. coupling нейтрона должен быть меньше coupling протона
        if params.get('coupling_neutron', 0) > params.get('coupling_proton', 1):
            total_error += 500.0
        
        # 4. coupling нейтрального мезона должен быть меньше заряженного
        if params.get('coupling_meson_neutral', 0) > params.get('coupling_meson_charged', 4):
            total_error += 300.0
        
        # 5. Правильная разность масс n-p (1.293 МэВ)
        mass_diff = abs((results['neutron_mass'] - results['proton_mass']) - 1.293)
        total_error += 200.0 * mass_diff
        
        # 6. Штраф за слишком большие энергии связи мезонов
        if abs(results['E_meson_charged']) > 10.0:
            total_error += 500.0
        
        return total_error, results

# ============== ОДНОПРОЦЕССНЫЙ ОТЖИГ ==============

class HybridAnnealerV9_2:
    
    def __init__(self):
        # Создаем директорию для результатов
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.result_dir = f"v92_results_{timestamp}"
        os.makedirs(self.result_dir, exist_ok=True)
        
        # Создаем логгер
        self.logger = Logger(self.result_dir)
        sys.stdout = self.logger
        
        print(f"МОДЕЛЬ v9.2 - запуск {timestamp}")
        print(f"Директория результатов: {self.result_dir}")
        print("="*80)
        
        # Параметры из v6.1 (как отправная точка)
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
        
        # Имена параметров
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
        
        # Диапазоны параметров
        self.ranges = {
            'base_mass_u': (1.5, 3.0),
            'base_mass_d': (3.0, 6.0),
            'freq_u': (0.7, 1.2),
            'freq_d': (0.7, 1.2),
            'amp_u': (0.8, 1.3),
            'amp_d': (0.7, 1.2),
            'color_coupling': (0.5, 3.0),
            'phase_coupling': (0.5, 2.0),
            'meson_coupling_scale': (0.5, 2.0),  # Уменьшен для мезонов
            'baryon_coupling_scale': (0.5, 2.0),
            'coupling_proton': (1.0, 2.5),
            'coupling_neutron': (0.1, 0.8),
            'coupling_meson_charged': (1.0, 5.0),  # Уменьшен диапазон
            'coupling_meson_neutral': (0.5, 4.0),  # Уменьшен диапазон
            'scale_factor': (90.0, 110.0)
        }
        
        # Целевые частицы
        self.targets = {
            'proton': {'mass': 938.272, 'charge': 1.0, 'composition': ['u', 'u', 'd']},
            'neutron': {'mass': 939.565, 'charge': 0.0, 'composition': ['u', 'd', 'd']},
            'pi+': {'mass': 139.570, 'charge': 1.0, 'composition': ['u', 'anti_d']},
            'pi0': {'mass': 134.9768, 'charge': 0.0, 'composition': ['u', 'anti_u']},
            'pi-': {'mass': 139.570, 'charge': -1.0, 'composition': ['d', 'anti_u']},
        }
        
        # Создаем калькулятор
        self.calculator = HybridCalculatorV9_2(self.param_names, self.ranges, self.targets)
        
        # Статистика
        self.start_time = None
        self.iterations_done = 0
    
    def save_progress(self, iteration, total_iterations, error, temperature, params, best_error, best_params):
        """Сохранение прогресса"""
        progress_file = os.path.join(self.result_dir, "progress.json")
        
        progress_data = {
            'iteration': iteration,
            'total_iterations': total_iterations,
            'progress_percent': (iteration / total_iterations) * 100,
            'current_error': error,
            'best_error': best_error,
            'temperature': temperature,
            'current_params': params,
            'best_params': best_params,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(progress_file, 'w') as f:
            json.dump(progress_data, f, indent=2, default=self.json_serializer)
        
        # Чекпоинт каждые 10000 итераций
        if iteration % 10000 == 0:
            checkpoint_file = os.path.join(self.result_dir, f"checkpoint_{iteration:07d}.json")
            with open(checkpoint_file, 'w') as f:
                json.dump(progress_data, f, indent=2, default=self.json_serializer)
    
    def print_progress(self, iteration, total_iterations, current_error, temperature, best_error, elapsed):
        """Вывод прогресса на экран"""
        progress_percent = (iteration / total_iterations) * 100
        iters_per_sec = iteration / elapsed if elapsed > 0 else 0
        
        # Оставшееся время
        if iteration > 0:
            remaining = elapsed / iteration * (total_iterations - iteration)
            if remaining < 60:
                remaining_str = f"{remaining:.0f}сек"
            elif remaining < 3600:
                remaining_str = f"{remaining/60:.1f}мин"
            else:
                remaining_str = f"{remaining/3600:.1f}час"
        else:
            remaining_str = "---"
        
        print(f"\rПрогресс: {progress_percent:5.1f}% | "
              f"Итерация: {iteration:7,d}/{total_iterations:,d} | "
              f"Ошибка: {current_error:8.3f} (лучшая: {best_error:8.3f}) | "
              f"Темп: {temperature:6.4f} | "
              f"Скорость: {iters_per_sec:5.1f} итер/сек | "
              f"Осталось: {remaining_str:>7}", end="")
    
    def run_annealing(self, total_iterations=500000, initial_temp=10.0, cooling_rate=0.99998):
        """Запуск отжига"""
        print("\n" + "="*80)
        print("ЗАПУСК ОТЖИГА v9.2")
        print("="*80)
        print(f"Всего итераций: {total_iterations:,}")
        print(f"Начальная температура: {initial_temp}")
        print(f"Скорость охлаждения: {cooling_rate}")
        print("="*80)
        
        self.start_time = time.time()
        
        # Инициализация параметров
        current_params = self.v61_params.copy()
        for param in self.param_names:
            if param in self.ranges:
                min_val, max_val = self.ranges[param]
                current_params[param] = np.random.uniform(min_val, max_val)
        
        # Начальная оценка
        current_error, current_results = self.calculator.calculate_error(current_params)
        best_params = current_params.copy()
        best_error = current_error
        best_results = current_results
        
        temperature = initial_temp
        update_interval = max(100, total_iterations // 1000)
        
        print(f"\nНачальная ошибка: {current_error:.3f}")
        print(f"Начальная температура: {temperature:.3f}")
        
        try:
            for iteration in range(1, total_iterations + 1):
                # Генерация новых параметров
                new_params = current_params.copy()
                
                for param in self.param_names:
                    if param in self.ranges:
                        min_val, max_val = self.ranges[param]
                        current_val = current_params[param]
                        
                        # Адаптивный шаг мутации
                        step = (max_val - min_val) * 0.05
                        mutation = np.random.normal(0, step) * temperature
                        
                        # Направленные мутации для физических ограничений
                        if param == 'coupling_neutron' and current_val > 0.5:
                            mutation -= 0.1 * step
                        elif param == 'coupling_proton' and current_val < 1.3:
                            mutation += 0.1 * step
                        elif param == 'coupling_meson_neutral' and current_val > new_params.get('coupling_meson_charged', 4):
                            mutation -= 0.2 * step
                        elif param == 'meson_coupling_scale' and current_val > 1.5:
                            mutation -= 0.1 * step
                        
                        new_val = current_val + mutation
                        
                        # Отражающие границы
                        while new_val < min_val or new_val > max_val:
                            if new_val < min_val:
                                new_val = 2 * min_val - new_val
                            if new_val > max_val:
                                new_val = 2 * max_val - new_val
                        
                        new_params[param] = new_val
                
                # Оценка новых параметров
                new_error, new_results = self.calculator.calculate_error(new_params)
                
                # Принятие решения (метрополис)
                delta = new_error - current_error
                
                if delta < 0:
                    # Принимаем улучшение
                    current_params = new_params
                    current_error = new_error
                    current_results = new_results
                else:
                    # Принимаем с вероятностью exp(-Δ/T)
                    prob = math.exp(-delta / temperature)
                    if np.random.random() < prob:
                        current_params = new_params
                        current_error = new_error
                        current_results = new_results
                
                # Обновление лучшего результата
                if new_error < best_error:
                    best_params = new_params.copy()
                    best_error = new_error
                    best_results = new_results
                
                # Охлаждение
                temperature *= cooling_rate
                
                # Вывод прогресса
                if iteration % update_interval == 0:
                    elapsed = time.time() - self.start_time
                    self.print_progress(iteration, total_iterations, current_error, 
                                      temperature, best_error, elapsed)
                    
                    # Сохранение прогресса
                    self.save_progress(iteration, total_iterations, current_error,
                                     temperature, current_params, best_error, best_params)
                
                self.iterations_done = iteration
            
            # Завершение
            print("\n" + "="*80)
            
        except KeyboardInterrupt:
            print("\n\nОтжиг прерван пользователем")
            print("Сохранение текущих результатов...")
        
        elapsed = time.time() - self.start_time
        
        print(f"\nОТЖИГ ЗАВЕРШЕН")
        print(f"Всего итераций: {self.iterations_done:,}")
        print(f"Общее время: {elapsed:.1f} сек")
        print(f"Скорость: {self.iterations_done/elapsed:.1f} итер/сек")
        print(f"Лучшая ошибка: {best_error:.3f}")
        print("="*80)
        
        # Сохранение финальных результатов
        self.save_final_results(best_params, best_error, best_results)
        
        # Печать финального отчета
        self.print_final_report(best_params, best_error, best_results)
        
        # Восстановление стандартного вывода
        sys.stdout = self.logger.console
        self.logger.close()
        
        return best_params, best_error, best_results
    
    def save_final_results(self, params, error, results):
        """Сохранение финальных результатов"""
        final_data = {
            'model': 'v9.2',
            'timestamp': datetime.now().isoformat(),
            'total_iterations': self.iterations_done,
            'final_error': error,
            'parameters': params,
            'results': results
        }
        
        with open(os.path.join(self.result_dir, "final_results.json"), 'w') as f:
            json.dump(final_data, f, indent=2, default=self.json_serializer)
    
    def print_final_report(self, params, error, results):
        """Печать финального отчета"""
        print("\n" + "="*80)
        print("ФИНАЛЬНЫЙ ОТЧЕТ v9.2")
        print("="*80)
        
        # Ключевые параметры
        print(f"\nКЛЮЧЕВЫЕ ПАРАМЕТРЫ:")
        
        param_groups = {
            'Массы кварков': ['base_mass_u', 'base_mass_d'],
            'Частоты': ['freq_u', 'freq_d'],
            'Амплитуды': ['amp_u', 'amp_d'],
            'Общие коэффициенты': ['color_coupling', 'phase_coupling'],
            'Масштабы связи': ['meson_coupling_scale', 'baryon_coupling_scale'],
            'Специфические коэффициенты': ['coupling_proton', 'coupling_neutron',
                                         'coupling_meson_charged', 'coupling_meson_neutral'],
            'Масштаб': ['scale_factor']
        }
        
        for group_name, param_list in param_groups.items():
            print(f"\n  {group_name}:")
            for param in param_list:
                if param in params:
                    print(f"    {param:25}: {params[param]:10.6f}")
        
        # Эффективные массы кварков
        print(f"\nЭФФЕКТИВНЫЕ МАССЫ КВАРКОВ:")
        print(f"  u-кварк: {results['m_u_eff']:8.2f} МэВ")
        print(f"  d-кварк: {results['m_d_eff']:8.2f} МэВ")
        print(f"  Отношение m_d/m_u: {results['ratio_d_u']:6.3f}")
        
        # Энергии связи
        print(f"\nЭНЕРГИИ СВЯЗИ:")
        print(f"  Протон:          {results['E_proton']:8.3f}")
        print(f"  Нейтрон:         {results['E_neutron']:8.3f}")
        print(f"  Заряженный мезон: {results['E_meson_charged']:8.3f}")
        print(f"  Нейтральный мезон: {results['E_meson_neutral']:8.3f}")
        print(f"  Отношение n/p:   {results['ratio_neutron_proton']:8.3f}")
        
        # Массы частиц
        print(f"\nМАССЫ ЧАСТИЦ:")
        total_error_percent = 0
        particle_count = 0
        
        for name in self.targets.keys():
            mass = results[f'{name}_mass']
            target = self.targets[name]['mass']
            error_percent = abs(mass - target) / target * 100
            total_error_percent += error_percent
            particle_count += 1
            
            status = "✓" if error_percent < 1.0 else "⚠" if error_percent < 10.0 else "✗"
            print(f"  {status} {name:5}: {mass:8.2f} МэВ (цель {target:7.2f}) - ошибка {error_percent:6.2f}%")
        
        avg_error_percent = total_error_percent / particle_count
        print(f"\n  Средняя ошибка: {avg_error_percent:6.2f}%")
        
        # Разность масс n-p
        diff = results['neutron_mass'] - results['proton_mass']
        diff_error = abs(diff - 1.293)
        diff_status = "✓" if diff_error < 0.1 else "⚠" if diff_error < 1.0 else "✗"
        
        print(f"\nРАЗНОСТЬ МАСС n-p:")
        print(f"  {diff_status} Модель: {diff:6.3f} МэВ")
        print(f"    Цель:    1.293 МэВ")
        print(f"    Отклонение: {diff_error:6.3f} МэВ")
        
        # Физические ограничения
        print(f"\nФИЗИЧЕСКИЕ ОГРАНИЧЕНИЯ:")
        
        constraints = [
            ("Нейтрон > протона", results['neutron_mass'] > results['proton_mass']),
            ("coupling_neutron < coupling_proton", params['coupling_neutron'] < params['coupling_proton']),
            ("coupling_meson_neutral < coupling_meson_charged", 
             params['coupling_meson_neutral'] < params['coupling_meson_charged']),
            ("1.3 < m_d/m_u < 2.2", 1.3 < results['ratio_d_u'] < 2.2),
            ("Все массы положительны", all(results[f'{name}_mass'] > 0 for name in self.targets.keys()))
        ]
        
        for desc, condition in constraints:
            status = "✓" if condition else "✗"
            print(f"  {status} {desc}")
        
        # Директория результатов
        print(f"\nРезультаты сохранены в: {self.result_dir}")
        print("="*80)
    
    def json_serializer(self, obj):
        """Сериализатор для JSON"""
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return str(obj)

# ============== ЗАПУСК ==============

def main():
    """Основная функция"""
    try:
        # Создаем оптимизатор
        annealer = HybridAnnealerV9_2()
        
        # Запускаем отжиг
        best_params, best_error, best_results = annealer.run_annealing(
            total_iterations=500000,    # 500,000 итераций
            initial_temp=10.0,          # Начальная температура
            cooling_rate=0.99998        # Скорость охлаждения
        )
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\nПрограмма прервана пользователем")
        return 1
    except Exception as e:
        print(f"\nКРИТИЧЕСКАЯ ОШИБКА: {e}")
        import traceback
        traceback.print_exc()
        return 2

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)