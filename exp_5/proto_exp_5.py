"""
Модель синхронизации осциллирующих нитей v5.0
Фокус на протоне, нейтроне и пионе с раздельными механизмами
"""

import numpy as np
import json
import time
import os
import sys
from datetime import datetime

# ================= КОНФИГУРАЦИЯ v5.0 =================
CONFIG = {
    # Целевые свойства частиц (в МэВ)
    'target_particles': {
        # БАРИОНЫ
        'proton': {
            'mass': 938.272,
            'charge': 1.0,
            'spin': 0.5,
            'composition': ['u', 'u', 'd'],
            'is_meson': False
        },
        'neutron': {
            'mass': 939.565,
            'charge': 0.0,
            'spin': 0.5,
            'composition': ['u', 'd', 'd'],
            'is_meson': False
        },
        # МЕЗОН
        'pi+': {
            'mass': 139.57,
            'charge': 1.0,
            'spin': 0.0,
            'composition': ['u', 'anti_d'],
            'is_meson': True
        }
    },
    
    # Свойства кварков (в МэВ до масштабирования)
    'type_properties': {
        'u': {'charge': 2/3, 'base_mass': 2.3},
        'd': {'charge': -1/3, 'base_mass': 4.8},
        'anti_u': {'charge': -2/3, 'base_mass': 2.3},
        'anti_d': {'charge': 1/3, 'base_mass': 4.8}
    },
    
    # Диапазоны перебора параметров (на основе предыдущих успехов)
    'param_ranges': {
        'frequency': {
            'u': {'min': 0.90, 'max': 1.10, 'step': 0.001},
            'd': {'min': 0.90, 'max': 1.10, 'step': 0.001}
        },
        'amplitude': {
            'u': {'min': 0.95, 'max': 1.05, 'step': 0.001},
            'd': {'min': 0.90, 'max': 1.00, 'step': 0.001}
        },
        # Разные силы связи для барионов и мезонов
        'coupling_baryon': {
            'min': 0.5,
            'max': 3.0,
            'step': 0.01
        },
        'coupling_meson': {
            'min': 1.0,
            'max': 5.0,
            'step': 0.05
        },
        'phase_shift': {
            'min': np.pi * 0.9,
            'max': np.pi * 1.1,
            'step': 0.01
        }
    },
    
    # Параметры поиска
    'search': {
        'max_iterations': 100000,
        'save_interval': 25000,
        'min_error': 0.01,
        'max_solutions': 50,
        'scale_factor': 100.0,
        'temperature': 0.2,
        'cooling_rate': 0.99999
    }
}

# ================= УНИВИЕРСАЛЬНАЯ МОДЕЛЬ v5.0 =================
class UniversalParticleModel:
    def __init__(self, composition, params, config, is_meson=False):
        self.composition = composition
        self.config = config
        self.is_meson = is_meson
        self.thread_count = len(composition)
        
        # Извлекаем параметры
        self.frequencies = []
        self.amplitudes = []
        
        for quark in composition:
            base_type = quark.replace('anti_', '')
            self.frequencies.append(params[f'freq_{base_type}'])
            self.amplitudes.append(params[f'amp_{base_type}'])
        
        # Сила связи в зависимости от типа
        self.coupling = params['coupling_meson'] if is_meson else params['coupling_baryon']
        
        # Фазы
        if is_meson:
            # Для мезонов: кварк и антикварк в противофазе
            self.phases = [0.0, params.get('phase_shift', np.pi)]
        else:
            # Для барионов: оптимизированные фазы
            base_phase = np.random.uniform(0, 2*np.pi)
            if composition == ['u', 'u', 'd']:  # Протон
                self.phases = [base_phase, base_phase, base_phase + np.pi/2]
            elif composition == ['u', 'd', 'd']:  # Нейтрон
                self.phases = [base_phase, base_phase + np.pi/2, base_phase + np.pi/2]
            else:
                self.phases = [base_phase, base_phase + np.pi/2, base_phase + np.pi]
        
        self.type_properties = config['type_properties']
        self.scale_factor = config['search']['scale_factor']
    
    def calculate_charge(self):
        """Вычисляет суммарный заряд"""
        total = sum(self.type_properties[q]['charge'] for q in self.composition)
        return round(total, 10)
    
    def calculate_base_mass(self):
        """Базовая масса без синхронизации"""
        total = 0
        for i, quark_type in enumerate(self.composition):
            base_mass = self.type_properties[quark_type]['base_mass']
            total += base_mass * self.amplitudes[i] * self.frequencies[i]
        return total
    
    def calculate_synchronization_energy(self):
        """Вычисляет энергию синхронизации (всегда положительную)"""
        # 1. Частотная синхронизация
        freq_coherence = 0
        pairs = 0
        
        for i in range(self.thread_count):
            for j in range(i+1, self.thread_count):
                ratio = self.frequencies[i] / self.frequencies[j]
                # Ищем ближайшее простое отношение
                simple_ratio = self._find_simple_ratio(ratio, max_denominator=5)
                coherence = 1.0 - abs(ratio - simple_ratio) / simple_ratio
                freq_coherence += max(0, coherence)
                pairs += 1
        
        freq_coherence = freq_coherence / pairs if pairs > 0 else 0.5
        
        # 2. Фазовая синхронизация
        phase_coherence = 0
        for i in range(self.thread_count):
            for j in range(i+1, self.thread_count):
                diff = abs(self.phases[i] - self.phases[j]) % (2*np.pi)
                diff = min(diff, 2*np.pi - diff)
                
                if self.is_meson:
                    # Для мезонов максимум при разности π
                    phase_coherence += np.cos(diff + np.pi)  # = -cos(diff)
                else:
                    # Для барионов максимум при одинаковых фазах
                    phase_coherence += np.cos(diff)
        
        max_pairs = self.thread_count * (self.thread_count - 1) / 2
        phase_coherence = (phase_coherence / max_pairs + 1) / 2 if max_pairs > 0 else 0.5
        
        # 3. Симметрия состава
        symmetry = 1.0
        if not self.is_meson:  # Для барионов
            if self.composition == ['u', 'u', 'd'] or self.composition == ['u', 'd', 'd']:
                symmetry = 1.1
        else:  # Для мезонов
            symmetry = 1.0
        
        # 4. Общая энергия синхронизации (ВСЕГДА ПОЛОЖИТЕЛЬНАЯ)
        sync_energy = self.coupling * (0.6 * freq_coherence + 0.4 * phase_coherence) * symmetry
        
        return sync_energy
    
    def _find_simple_ratio(self, ratio, max_denominator=5):
        """Находит ближайшее простое отношение частот"""
        best_error = float('inf')
        best_ratio = ratio
        
        for den in range(1, max_denominator + 1):
            for num in range(1, max_denominator + 1):
                simple_ratio = num / den
                error = abs(ratio - simple_ratio)
                if error < best_error:
                    best_error = error
                    best_ratio = simple_ratio
        
        return best_ratio
    
    def calculate_total_mass(self):
        """Общая масса частицы"""
        base_mass = self.calculate_base_mass()
        sync_energy = self.calculate_synchronization_energy()
        
        # КЛЮЧЕВОЕ ИЗМЕНЕНИЕ: РАЗНЫЕ ЗНАКИ ДЛЯ БАРИОНОВ И МЕЗОНОВ
        if self.is_meson:
            # Для мезонов: базовая масса МИНУС энергия связи
            total_mass = base_mass - sync_energy
        else:
            # Для барионов: базовая масса ПЛЮС энергия связи
            total_mass = base_mass + sync_energy
        
        # Масштабируем до МэВ
        return total_mass * self.scale_factor
    
    def calculate_spin(self):
        """Вычисляет спин частицы"""
        if self.is_meson:
            return 0.0
        else:
            return 0.5

# ================= ИНТЕЛЛЕКТУАЛЬНЫЙ ПОИСК v5.0 =================
class IntelligentParameterSearchV5:
    def __init__(self, config):
        self.config = config
        self.solutions = []
        self.best_solution = None
        self.iteration = 0
        self.start_time = time.time()
        self.temperature = config['search']['temperature']
        
        # Создаём директорию для результатов
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.result_dir = f"particle_search_v5_{timestamp}"
        os.makedirs(self.result_dir, exist_ok=True)
        
        # Сохраняем конфигурацию
        with open(f"{self.result_dir}/config.json", 'w') as f:
            json.dump(config, f, indent=2, default=self._json_serializer)
        
        # Начальные параметры (из предыдущих успехов)
        self.current_params = {
            'freq_u': 0.98,
            'freq_d': 0.95,
            'amp_u': 1.00,
            'amp_d': 0.92,
            'coupling_baryon': 1.5,
            'coupling_meson': 3.0,
            'phase_shift': np.pi
        }
        
        print(f"Результаты будут сохранены в: {self.result_dir}")
        print(f"Целевые частицы: {list(config['target_particles'].keys())}")
    
    def _json_serializer(self, obj):
        """Сериализатор для numpy типов"""
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return str(obj)
    
    def generate_parameters(self, method='smart'):
        """Генерирует параметры разными методами"""
        ranges = self.config['param_ranges']
        
        if method == 'smart' and self.best_solution and self.iteration > 5000:
            # Умная генерация на основе лучшего решения
            best_params = self.best_solution['parameters']
            new_params = {}
            temp_factor = max(0.1, self.temperature)
            
            for key in best_params.keys():
                if 'freq' in key:
                    quark = key.split('_')[1]
                    min_val = ranges['frequency'][quark]['min']
                    max_val = ranges['frequency'][quark]['max']
                    std = 0.01 * temp_factor
                    new_params[key] = np.clip(best_params[key] + np.random.normal(0, std), min_val, max_val)
                
                elif 'amp' in key:
                    quark = key.split('_')[1]
                    min_val = ranges['amplitude'][quark]['min']
                    max_val = ranges['amplitude'][quark]['max']
                    std = 0.005 * temp_factor
                    new_params[key] = np.clip(best_params[key] + np.random.normal(0, std), min_val, max_val)
                
                elif key == 'coupling_baryon':
                    min_val = ranges['coupling_baryon']['min']
                    max_val = ranges['coupling_baryon']['max']
                    std = 0.02 * temp_factor
                    new_params[key] = np.clip(best_params[key] + np.random.normal(0, std), min_val, max_val)
                
                elif key == 'coupling_meson':
                    min_val = ranges['coupling_meson']['min']
                    max_val = ranges['coupling_meson']['max']
                    std = 0.1 * temp_factor
                    new_params[key] = np.clip(best_params[key] + np.random.normal(0, std), min_val, max_val)
                
                elif key == 'phase_shift':
                    min_val = ranges['phase_shift']['min']
                    max_val = ranges['phase_shift']['max']
                    std = 0.05 * temp_factor
                    new_params[key] = np.clip(best_params[key] + np.random.normal(0, std), min_val, max_val)
            
            return new_params
        
        else:
            # Случайная генерация
            params = {}
            
            for quark in ['u', 'd']:
                freq_range = ranges['frequency'][quark]
                amp_range = ranges['amplitude'][quark]
                
                params[f'freq_{quark}'] = np.random.uniform(freq_range['min'], freq_range['max'])
                params[f'amp_{quark}'] = np.random.uniform(amp_range['min'], amp_range['max'])
            
            params['coupling_baryon'] = np.random.uniform(
                ranges['coupling_baryon']['min'], ranges['coupling_baryon']['max'])
            params['coupling_meson'] = np.random.uniform(
                ranges['coupling_meson']['min'], ranges['coupling_meson']['max'])
            params['phase_shift'] = np.random.uniform(
                ranges['phase_shift']['min'], ranges['phase_shift']['max'])
            
            return params
    
    def create_particles(self, params):
        """Создаёт все три частицы"""
        particles = {}
        
        for name, target in self.config['target_particles'].items():
            particles[name] = UniversalParticleModel(
                composition=target['composition'],
                params=params,
                config=self.config,
                is_meson=target['is_meson']
            )
        
        return particles
    
    def calculate_error(self, particles):
        """Вычисляет взвешенную ошибку"""
        total_error = 0
        details = {}
        
        for name, particle in particles.items():
            target = self.config['target_particles'][name]
            
            # Рассчитываем свойства
            mass = particle.calculate_total_mass()
            charge = particle.calculate_charge()
            spin = particle.calculate_spin()
            
            # Вычисляем ошибки
            mass_error = abs(mass - target['mass']) / target['mass']
            charge_error = abs(charge - target['charge'])
            spin_error = abs(spin - target['spin'])
            
            # Взвешивание (барионы важнее сейчас)
            weights = {'mass': 3.0, 'charge': 5.0, 'spin': 0.5}
            if name in ['proton', 'neutron']:
                weights['mass'] = 4.0  # Особо важны массы нуклонов
            
            error = (weights['mass'] * mass_error + 
                    weights['charge'] * charge_error + 
                    weights['spin'] * spin_error)
            
            total_error += error
            
            details[name] = {
                'mass': mass,
                'charge': charge,
                'spin': spin,
                'mass_error': mass_error,
                'charge_error': charge_error,
                'spin_error': spin_error,
                'particle_error': error
            }
        
        return total_error, details
    
    def save_solution(self, params, error, details, iteration):
        """Сохраняет решение"""
        solution = {
            'iteration': iteration,
            'parameters': params,
            'error': error,
            'details': details,
            'temperature': self.temperature,
            'timestamp': time.time()
        }
        
        self.solutions.append(solution)
        
        # Проверяем, является ли это лучшим решением
        if self.best_solution is None or error < self.best_solution['error']:
            self.best_solution = solution
            
            print(f"\n{'='*70}")
            print(f"НОВОЕ ЛУЧШЕЕ РЕШЕНИЕ (итерация {iteration})")
            print(f"Общая ошибка: {error:.6f}")
            
            for name in ['proton', 'neutron', 'pi+']:
                if name in details:
                    d = details[name]
                    target = self.config['target_particles'][name]
                    mass_err_percent = d['mass_error'] * 100
                    print(f"{name}: {d['mass']:.3f} МэВ (цель {target['mass']}, ошибка {mass_err_percent:.2f}%)")
            
            print(f"{'='*70}")
        
        # Сортируем и обрезаем
        self.solutions.sort(key=lambda x: x['error'])
        max_solutions = self.config['search']['max_solutions']
        if len(self.solutions) > max_solutions:
            self.solutions = self.solutions[:max_solutions]
    
    def save_checkpoint(self, force=False):
        """Сохраняет контрольную точку"""
        if not force and self.iteration % self.config['search']['save_interval'] != 0:
            return
        
        checkpoint = {
            'iteration': self.iteration,
            'solutions': self.solutions,
            'best_solution': self.best_solution,
            'temperature': self.temperature,
            'elapsed_time': time.time() - self.start_time
        }
        
        checkpoint_file = f"{self.result_dir}/checkpoint_{self.iteration:08d}.json"
        latest_file = f"{self.result_dir}/latest_checkpoint.json"
        
        try:
            with open(checkpoint_file, 'w') as f:
                json.dump(checkpoint, f, indent=2, default=self._json_serializer)
            
            with open(latest_file, 'w') as f:
                json.dump(checkpoint, f, indent=2, default=self._json_serializer)
            
            print(f"\n  Контрольная точка сохранена: {checkpoint_file}")
        except Exception as e:
            print(f"  Ошибка сохранения: {e}")
    
    def run(self):
        """Запускает основной цикл поиска"""
        print("="*70)
        print("МОДЕЛЬ СИНХРОНИЗАЦИИ НИТЕЙ v5.0")
        print("Протон, нейтрон и пион с раздельными механизмами")
        print("="*70)
        
        max_iter = self.config['search']['max_iterations']
        cooling_rate = self.config['search']['cooling_rate']
        min_error = self.config['search']['min_error']
        
        try:
            while self.iteration < max_iter:
                # Выбор метода генерации
                if self.iteration < 10000 or np.random.random() < 0.3:
                    method = 'random'
                else:
                    method = 'smart'
                
                # Генерация параметров
                params = self.generate_parameters(method)
                
                # Создание частиц
                particles = self.create_particles(params)
                
                # Вычисление ошибки
                error, details = self.calculate_error(particles)
                
                # Принимаем ли это решение?
                accept = False
                if self.best_solution is None:
                    accept = True
                elif error < self.best_solution['error']:
                    accept = True
                else:
                    # Имитация отжига
                    prob = np.exp((self.best_solution['error'] - error) / self.temperature)
                    if np.random.random() < prob:
                        accept = True
                
                if accept:
                    self.save_solution(params, error, details, self.iteration)
                
                # Уменьшаем температуру
                self.temperature *= cooling_rate
                
                # Увеличиваем счётчик
                self.iteration += 1
                
                # Вывод прогресса
                if self.iteration % 1000 == 0:
                    elapsed = time.time() - self.start_time
                    if self.best_solution:
                        best_err = self.best_solution['error']
                        proton_mass = self.best_solution['details'].get('proton', {}).get('mass', 0)
                        pi_mass = self.best_solution['details'].get('pi+', {}).get('mass', 0)
                    else:
                        best_err = float('inf')
                        proton_mass = 0
                        pi_mass = 0
                    
                    print(f"\rИтерация: {self.iteration:,} | "
                          f"Ошибка: {best_err:.4f} | "
                          f"Протон: {proton_mass:.0f} МэВ | "
                          f"Пион: {pi_mass:.0f} МэВ | "
                          f"Время: {elapsed:.1f} сек", end='')
                
                # Сохранение контрольной точки
                if self.iteration % 10000 == 0:
                    self.save_checkpoint()
                
                # Проверка критерия остановки
                if error < min_error:
                    print(f"\n\nДостигнута минимальная ошибка {min_error}!")
                    break
        
        except KeyboardInterrupt:
            print("\n\nПоиск прерван пользователем")
        
        finally:
            # Финальное сохранение
            self.save_checkpoint(force=True)
            
            # Финальный вывод
            print(f"\n\n{'='*70}")
            print("ПОИСК ЗАВЕРШЁН")
            print(f"Всего итераций: {self.iteration:,}")
            print(f"Время выполнения: {time.time() - self.start_time:.2f} сек")
            print(f"Директория с результатами: {self.result_dir}")
            
            if self.best_solution:
                self._print_final_summary()
    
    def _print_final_summary(self):
        """Выводит финальную сводку"""
        best = self.best_solution
        details = best['details']
        params = best['parameters']
        
        print(f"\n{'='*70}")
        print("ЛУЧШЕЕ РЕШЕНИЕ:")
        print(f"Итерация: {best['iteration']:,}")
        print(f"Общая ошибка: {best['error']:.6f}")
        
        print(f"\nДЕТАЛЬНЫЕ РЕЗУЛЬТАТЫ:")
        print(f"{'Частица':<10} {'Масса (МэВ)':<15} {'Цель (МэВ)':<15} {'Ошибка (%)':<12} {'Заряд':<8} {'Спин':<8}")
        print("-" * 70)
        
        for name in ['proton', 'neutron', 'pi+']:
            if name in details:
                d = details[name]
                target = self.config['target_particles'][name]
                mass_err_percent = d['mass_error'] * 100
                print(f"{name:<10} {d['mass']:<15.3f} {target['mass']:<15.3f} "
                      f"{mass_err_percent:<12.3f} {d['charge']:<8.1f} {d['spin']:<8.1f}")
        
        print(f"\nПАРАМЕТРЫ МОДЕЛИ:")
        for key, value in params.items():
            print(f"  {key}: {value:.6f}")
        
        # Эффективные массы кварков
        print(f"\nЭФФЕКТИВНЫЕ МАССЫ КВАРКОВ (до синхронизации):")
        for quark in ['u', 'd']:
            base_mass = self.config['type_properties'][quark]['base_mass']
            freq = params[f'freq_{quark}']
            amp = params[f'amp_{quark}']
            effective = base_mass * freq * amp * self.config['search']['scale_factor']
            print(f"  {quark}-кварк: {effective:.1f} МэВ")
        
        print(f"\nОТНОШЕНИЕ МАСС d/u: {params['freq_d'] * params['amp_d'] * 4.8 / (params['freq_u'] * params['amp_u'] * 2.3):.3f}")
        
        # Энергии связи
        print(f"\nЭНЕРГИИ СВЯЗИ (в единицах модели):")
        particles = self.create_particles(params)
        for name, particle in particles.items():
            base = particle.calculate_base_mass()
            sync = particle.calculate_synchronization_energy()
            total = particle.calculate_total_mass()
            sign = "-" if particle.is_meson else "+"
            print(f"  {name}: база={base:.3f}, связь={sync:.3f}, масса=база{sign}связь={total:.3f}×100 МэВ")

# ================= ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ =================
def visualize_results(result_dir):
    """Создаёт визуализацию результатов"""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Для визуализации установите matplotlib: pip install matplotlib")
        return
    
    latest_file = f"{result_dir}/latest_checkpoint.json"
    
    if not os.path.exists(latest_file):
        print(f"Файл результатов не найден: {latest_file}")
        return
    
    with open(latest_file, 'r') as f:
        data = json.load(f)
    
    if 'best_solution' not in data:
        print("Нет данных для визуализации")
        return
    
    best = data['best_solution']
    details = best['details']
    
    # Создаём график
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Сравнение масс
    ax1 = axes[0, 0]
    particles = ['proton', 'neutron', 'pi+']
    masses = [details[p]['mass'] for p in particles]
    targets = [CONFIG['target_particles'][p]['mass'] for p in particles]
    
    x = np.arange(len(particles))
    width = 0.35
    
    ax1.bar(x - width/2, masses, width, label='Расчёт', alpha=0.7, color='blue')
    ax1.bar(x + width/2, targets, width, label='Эксперимент', alpha=0.7, color='red')
    
    ax1.set_xlabel('Частицы')
    ax1.set_ylabel('Масса (МэВ)')
    ax1.set_title('Сравнение расчётных и экспериментальных масс')
    ax1.set_xticks(x)
    ax1.set_xticklabels(['Протон', 'Нейтрон', 'Пион π⁺'])
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Ошибки масс
    ax2 = axes[0, 1]
    errors = [details[p]['mass_error'] * 100 for p in particles]
    colors = ['green' if e < 1 else 'orange' if e < 5 else 'red' for e in errors]
    
    bars = ax2.bar(particles, errors, color=colors)
    ax2.axhline(1, color='green', linestyle='--', alpha=0.5, label='1% ошибка')
    ax2.axhline(5, color='orange', linestyle='--', alpha=0.5, label='5% ошибка')
    ax2.set_xlabel('Частицы')
    ax2.set_ylabel('Ошибка массы (%)')
    ax2.set_title('Относительные ошибки масс')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Добавляем значения на столбцы
    for bar, error in zip(bars, errors):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{error:.2f}%', ha='center', va='bottom')
    
    # 3. Эффективные массы кварков
    ax3 = axes[1, 0]
    params = best['parameters']
    quark_masses = [
        CONFIG['type_properties']['u']['base_mass'] * params['amp_u'] * params['freq_u'] * 100,
        CONFIG['type_properties']['d']['base_mass'] * params['amp_d'] * params['freq_d'] * 100
    ]
    
    ax3.bar(['u-кварк', 'd-кварк'], quark_masses, color=['lightblue', 'lightgreen'])
    ax3.set_xlabel('Кварк')
    ax3.set_ylabel('Эффективная масса (МэВ)')
    ax3.set_title('Эффективные массы кварков')
    ax3.grid(True, alpha=0.3)
    
    # Добавляем значения
    for i, mass in enumerate(quark_masses):
        ax3.text(i, mass, f'{mass:.1f} МэВ', ha='center', va='bottom')
    
    # 4. Силы связи
    ax4 = axes[1, 1]
    couplings = {
        'coupling_baryon': params['coupling_baryon'],
        'coupling_meson': params['coupling_meson']
    }
    
    ax4.bar(couplings.keys(), couplings.values(), color=['purple', 'orange'])
    ax4.set_xlabel('Тип связи')
    ax4.set_ylabel('Значение')
    ax4.set_title('Силы связи для барионов и мезонов')
    ax4.grid(True, alpha=0.3)
    
    # Добавляем значения
    for i, (key, value) in enumerate(couplings.items()):
        ax4.text(i, value, f'{value:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    # Сохраняем график
    plot_file = f"{result_dir}/results_visualization.png"
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nВизуализация сохранена в: {plot_file}")

# ================= ОСНОВНОЙ БЛОК =================
if __name__ == "__main__":
    print("\n" + "="*70)
    print("МОДЕЛЬ СИНХРОНИЗАЦИИ НИТЕЙ v5.0")
    print("Протон, нейтрон и пион с раздельными механизмами")
    print("="*70)
    
    # Создаём и запускаем поиск
    search = IntelligentParameterSearchV5(CONFIG)
    search.run()
    
    # Создаём визуализацию
    visualize_results(search.result_dir)
    
    print("\n" + "="*70)
    print("ГОТОВО! v5.0 фокусируется на трёх ключевых частицах")
    print("с раздельными знаками энергии синхронизации.")
    print("="*70)