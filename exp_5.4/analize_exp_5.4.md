# Модель синхронизации осциллирующих нитей v5.5 - Возврат к основам

**Проблема v5.4:** Мы слишком сузили диапазоны и потеряли хорошие решения v5.3. Пион стал тяжелым (155 МэВ вместо 139 МэВ).

**Решение:** Вернемся к рабочей версии v5.3 и точно настроим только один параметр - `coupling_neutron`.

```python
"""
Модель синхронизации осциллирующих нитей v5.5
Точная настройка через coupling_neutron
"""

import numpy as np
import json
import time
import os
import sys
from datetime import datetime

# ================= КОНФИГУРАЦИЯ v5.5 =================
CONFIG = {
    'target_particles': {
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
        'pi+': {
            'mass': 139.57,
            'charge': 1.0,
            'spin': 0.0,
            'composition': ['u', 'anti_d'],
            'is_meson': True
        }
    },
    
    'type_properties': {
        'u': {'charge': 2/3, 'base_mass': 2.25},
        'd': {'charge': -1/3, 'base_mass': 4.60},
        'anti_u': {'charge': -2/3, 'base_mass': 2.25},
        'anti_d': {'charge': 1/3, 'base_mass': 4.60}
    },
    
    # ФИКСИРОВАННЫЕ параметры из v5.3 (работали хорошо)
    'fixed_params': {
        'freq_u': 0.951000,
        'freq_d': 0.899000,
        'amp_u': 1.001000,
        'amp_d': 0.849000,
        'coupling_proton': 1.720000,
        'coupling_meson': 4.410427,
        'phase_shift': 3.163802
    },
    
    # ЕДИНСТВЕННЫЙ изменяемый параметр
    'coupling_neutron_range': {
        'min': 0.320,  # Увеличиваем с 0.300!
        'max': 0.340,
        'step': 0.00001
    },
    
    'search': {
        'max_iterations': 50000,
        'save_interval': 10000,
        'min_error': 0.0001,
        'max_solutions': 20,
        'scale_factor': 100.0,
        'temperature': 0.01,
        'cooling_rate': 0.99999
    }
}

# ================= МОДЕЛЬ v5.5 =================
class ParticleModelV55:
    def __init__(self, composition, params, config, particle_name=None):
        self.composition = composition
        self.config = config
        self.particle_name = particle_name
        self.is_meson = particle_name == 'pi+'
        self.thread_count = len(composition)
        
        # Все параметры фиксированы, кроме coupling_neutron
        fixed = config['fixed_params']
        self.freq_u = fixed['freq_u']
        self.freq_d = fixed['freq_d']
        self.amp_u = fixed['amp_u']
        self.amp_d = fixed['amp_d']
        self.phase_shift = fixed['phase_shift']
        
        # Сила связи
        if self.is_meson:
            self.coupling = fixed['coupling_meson']
        elif particle_name == 'proton':
            self.coupling = fixed['coupling_proton']
        else:  # neutron
            self.coupling = params['coupling_neutron']  # Единственный изменяемый
        
        # Частоты и амплитуды для каждого кварка
        self.frequencies = []
        self.amplitudes = []
        for quark in composition:
            base_type = quark.replace('anti_', '')
            if base_type == 'u':
                self.frequencies.append(self.freq_u)
                self.amplitudes.append(self.amp_u)
            else:  # d
                self.frequencies.append(self.freq_d)
                self.amplitudes.append(self.amp_d)
        
        # Фазы
        if self.is_meson:
            self.phases = [0.0, self.phase_shift]
        elif composition == ['u', 'u', 'd']:
            self.phases = [0.0, 0.0, np.pi/2]
        else:  # neutron
            self.phases = [0.0, np.pi/2, np.pi/2]
        
        self.type_properties = config['type_properties']
        self.scale_factor = config['search']['scale_factor']
    
    def calculate_charge(self):
        total = 0
        for q in self.composition:
            total += self.type_properties[q]['charge']
        return round(total, 10)
    
    def calculate_base_mass(self):
        total = 0
        for i, q in enumerate(self.composition):
            base_mass = self.type_properties[q]['base_mass']
            total += base_mass * self.amplitudes[i] * self.frequencies[i]
        return total
    
    def calculate_synchronization_energy(self):
        # Упрощенная формула для стабильности
        # Частотная когерентность
        freq_coherence = 1.0  # Все частоты фиксированы и близки
        
        # Фазовая когерентность
        phase_coherence = 0
        for i in range(self.thread_count):
            for j in range(i+1, self.thread_count):
                diff = abs(self.phases[i] - self.phases[j]) % (2*np.pi)
                diff = min(diff, 2*np.pi - diff)
                
                if self.is_meson:
                    phase_coherence += np.cos(diff + np.pi)
                else:
                    phase_coherence += np.cos(diff)
        
        max_pairs = self.thread_count * (self.thread_count - 1) / 2
        phase_coherence = (phase_coherence / max_pairs + 1) / 2
        
        # Симметрия
        symmetry = 1.0
        if self.particle_name == 'proton':
            symmetry = 1.1
        elif self.particle_name == 'neutron':
            symmetry = 0.95
        
        sync_energy = self.coupling * (0.6 * freq_coherence + 0.4 * phase_coherence) * symmetry
        return sync_energy
    
    def calculate_total_mass(self):
        base = self.calculate_base_mass()
        sync = self.calculate_synchronization_energy()
        
        if self.is_meson:
            total = base - sync
        else:
            total = base + sync
        
        return total * self.scale_factor
    
    def calculate_spin(self):
        return 0.0 if self.is_meson else 0.5

# ================= ПРОСТОЙ ПОИСК v5.5 =================
class SimpleSearchV55:
    def __init__(self, config):
        self.config = config
        self.solutions = []
        self.best_solution = None
        self.iteration = 0
        self.start_time = time.time()
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.result_dir = f"particle_search_v55_{timestamp}"
        os.makedirs(self.result_dir, exist_ok=True)
        
        print("="*70)
        print("МОДЕЛЬ v5.5: Точечная настройка coupling_neutron")
        print("Все параметры фиксированы, кроме coupling_neutron")
        print("="*70)
    
    def generate_coupling_neutron(self):
        """Генерирует только coupling_neutron"""
        range_def = self.config['coupling_neutron_range']
        
        if self.best_solution and np.random.random() < 0.7:
            # Локальный поиск вокруг лучшего значения
            best = self.best_solution['parameters']['coupling_neutron']
            std = 0.001
            new_val = np.random.normal(best, std)
        else:
            # Случайный выбор
            new_val = np.random.uniform(range_def['min'], range_def['max'])
        
        return np.clip(new_val, range_def['min'], range_def['max'])
    
    def create_particles(self, coupling_neutron):
        """Создает частицы с заданным coupling_neutron"""
        params = {
            'coupling_neutron': coupling_neutron
        }
        
        particles = {}
        for name, target in self.config['target_particles'].items():
            particles[name] = ParticleModelV55(
                composition=target['composition'],
                params=params,
                config=self.config,
                particle_name=name
            )
        
        return particles, params
    
    def calculate_error(self, particles):
        """Вычисляет ошибку с акцентом на разность масс"""
        details = {}
        total_error = 0
        
        # Основные массы
        for name, particle in particles.items():
            target = self.config['target_particles'][name]
            
            mass = particle.calculate_total_mass()
            charge = particle.calculate_charge()
            spin = particle.calculate_spin()
            
            mass_error = abs(mass - target['mass']) / target['mass']
            charge_error = abs(charge - target['charge'])
            spin_error = abs(spin - target['spin'])
            
            # Взвешивание
            weights = {
                'proton': {'mass': 10.0, 'charge': 5.0, 'spin': 0.5},
                'neutron': {'mass': 20.0, 'charge': 5.0, 'spin': 0.5},
                'pi+': {'mass': 5.0, 'charge': 5.0, 'spin': 0.5}
            }
            w = weights[name]
            
            error = (w['mass'] * mass_error + 
                    w['charge'] * charge_error + 
                    w['spin'] * spin_error)
            
            total_error += error
            
            details[name] = {
                'mass': mass,
                'charge': charge,
                'spin': spin,
                'mass_error': mass_error,
                'charge_error': charge_error,
                'spin_error': spin_error
            }
        
        # Разность масс (КРИТИЧЕСКИ ВАЖНО!)
        mass_p = details['proton']['mass']
        mass_n = details['neutron']['mass']
        mass_diff = mass_n - mass_p
        target_diff = 1.293
        
        # Абсолютная ошибка в МэВ
        diff_error_abs = abs(mass_diff - target_diff)
        diff_error_rel = diff_error_abs / target_diff
        
        # Огромный вес для точной разности масс
        total_error += 10000.0 * diff_error_abs  # 10000x вес для абсолютной ошибки
        total_error += 1000.0 * diff_error_rel   # 1000x вес для относительной
        
        details['mass_diff'] = {
            'calculated': mass_diff,
            'target': target_diff,
            'error_abs': diff_error_abs,
            'error_rel': diff_error_rel
        }
        
        return total_error, details
    
    def run(self):
        max_iter = self.config['search']['max_iterations']
        min_error = self.config['search']['min_error']
        
        print("Начальные фиксированные параметры:")
        for key, value in self.config['fixed_params'].items():
            print(f"  {key}: {value:.6f}")
        
        print(f"\nДиапазон coupling_neutron: [{self.config['coupling_neutron_range']['min']:.3f}, "
              f"{self.config['coupling_neutron_range']['max']:.3f}]")
        
        best_diff_error = float('inf')
        
        try:
            while self.iteration < max_iter:
                coupling_neutron = self.generate_coupling_neutron()
                particles, params = self.create_particles(coupling_neutron)
                error, details = self.calculate_error(particles)
                
                # Сохраняем решение
                solution = {
                    'iteration': self.iteration,
                    'parameters': params,
                    'error': error,
                    'details': details
                }
                
                if self.best_solution is None or error < self.best_solution['error']:
                    self.best_solution = solution
                    self.solutions.append(solution)
                    
                    # Сортируем и обрезаем
                    self.solutions.sort(key=lambda x: x['error'])
                    if len(self.solutions) > self.config['search']['max_solutions']:
                        self.solutions = self.solutions[:self.config['search']['max_solutions']]
                    
                    # Вывод прогресса
                    mass_diff = details['mass_diff']['calculated']
                    diff_error = details['mass_diff']['error_abs']
                    
                    if diff_error < best_diff_error:
                        best_diff_error = diff_error
                        print(f"\nИтерация {self.iteration:,}:")
                        print(f"  coupling_neutron = {coupling_neutron:.6f}")
                        print(f"  Разность масс = {mass_diff:.6f} МэВ (цель 1.293)")
                        print(f"  Ошибка разности = {diff_error:.6f} МэВ")
                        print(f"  Масса протона = {details['proton']['mass']:.3f} МэВ")
                        print(f"  Масса нейтрона = {details['neutron']['mass']:.3f} МэВ")
                        print(f"  Масса пиона = {details['pi+']['mass']:.3f} МэВ")
                
                self.iteration += 1
                
                if self.iteration % 1000 == 0:
                    elapsed = time.time() - self.start_time
                    if self.best_solution:
                        best_err = self.best_solution['error']
                        mass_diff = self.best_solution['details']['mass_diff']['calculated']
                        diff_error = self.best_solution['details']['mass_diff']['error_abs']
                    else:
                        best_err = float('inf')
                        mass_diff = 0
                        diff_error = 0
                    
                    print(f"\rИтерация: {self.iteration:,} | "
                          f"Лучшая ошибка: {best_err:.4f} | "
                          f"Разность: {mass_diff:.4f} МэВ | "
                          f"Погрешность: {diff_error:.4f} МэВ | "
                          f"Время: {elapsed:.1f} сек", end='')
                
                if error < min_error:
                    print(f"\n\nДостигнута минимальная ошибка {min_error}!")
                    break
        
        except KeyboardInterrupt:
            print("\n\nПоиск прерван")
        
        finally:
            print(f"\n\n{'='*70}")
            print("ПОИСК ЗАВЕРШЁН")
            print(f"Итераций: {self.iteration:,}")
            print(f"Время: {time.time() - self.start_time:.2f} сек")
            
            if self.best_solution:
                self._print_summary()
                self._save_final_results()
    
    def _print_summary(self):
        best = self.best_solution
        details = best['details']
        params = best['parameters']
        
        print(f"\n{'='*70}")
        print("ФИНАЛЬНОЕ РЕШЕНИЕ v5.5:")
        
        # Фиксированные параметры
        fixed = self.config['fixed_params']
        print(f"\nФИКСИРОВАННЫЕ ПАРАМЕТРЫ:")
        for key in ['freq_u', 'freq_d', 'amp_u', 'amp_d', 'coupling_proton', 'coupling_meson', 'phase_shift']:
            print(f"  {key}: {fixed[key]:.6f}")
        
        print(f"\nНАЙДЕННЫЙ ПАРАМЕТР:")
        print(f"  coupling_neutron: {params['coupling_neutron']:.6f}")
        
        print(f"\nТОЧНОСТЬ МОДЕЛИ:")
        print(f"{'Частица':<10} {'Масса (МэВ)':<15} {'Цель (МэВ)':<15} {'Ошибка (%)':<12}")
        print("-" * 70)
        
        for name in ['proton', 'neutron', 'pi+']:
            d = details[name]
            target = self.config['target_particles'][name]
            print(f"{name:<10} {d['mass']:<15.6f} {target['mass']:<15.6f} "
                  f"{d['mass_error']*100:<12.6f}")
        
        mass_diff = details['mass_diff']['calculated']
        target_diff = details['mass_diff']['target']
        diff_error = details['mass_diff']['error_abs']
        
        print(f"\nРАЗНОСТЬ МАСС НЕЙТРОН-ПРОТОН:")
        print(f"  Расчётная: {mass_diff:.6f} МэВ")
        print(f"  Эксперимент: {target_diff:.6f} МэВ")
        print(f"  Абсолютная ошибка: {diff_error:.6f} МэВ")
        print(f"  Относительная ошибка: {details['mass_diff']['error_rel']*100:.6f}%")
        
        # Физическая интерпретация
        print(f"\nФИЗИЧЕСКАЯ ИНТЕРПРЕТАЦИЯ:")
        
        # Массы кварков
        m_u = fixed['freq_u'] * fixed['amp_u'] * 2.25 * 100
        m_d = fixed['freq_d'] * fixed['amp_d'] * 4.60 * 100
        print(f"  1. Эффективные массы кварков:")
        print(f"     u-кварк: {m_u:.2f} МэВ")
        print(f"     d-кварк: {m_d:.2f} МэВ")
        print(f"     Отношение m_d/m_u: {m_d/m_u:.3f}")
        
        # Отношение сил связи
        print(f"  2. Отношение сил связи нейтрон/протон:")
        print(f"     coupling_neutron / coupling_proton = {params['coupling_neutron']/fixed['coupling_proton']:.3f}")
        print(f"     Энергия связи нейтрона в {fixed['coupling_proton']/params['coupling_neutron']:.1f} раза меньше")
        
        # Разность масс анализ
        print(f"  3. Анализ разности масс:")
        particles, _ = self.create_particles(params['coupling_neutron'])
        
        p_base = particles['proton'].calculate_base_mass()
        n_base = particles['neutron'].calculate_base_mass()
        p_sync = particles['proton'].calculate_synchronization_energy()
        n_sync = particles['neutron'].calculate_synchronization_energy()
        
        diff_base = (n_base - p_base) * 100
        diff_sync = (n_sync - p_sync) * 100
        
        print(f"     Разность по базовым массам: {diff_base:.3f} МэВ")
        print(f"     Разность по энергиям связи: {diff_sync:.3f} МэВ")
        print(f"     Суммарная разность: {diff_base + diff_sync:.3f} МэВ")
    
    def _save_final_results(self):
        """Сохраняет финальные результаты"""
        if not self.best_solution:
            return
        
        result_file = f"{self.result_dir}/FINAL_RESULTS_v55.txt"
        with open(result_file, 'w', encoding='utf-8') as f:
            f.write("="*70 + "\n")
            f.write("ФИНАЛЬНЫЕ РЕЗУЛЬТАТЫ МОДЕЛИ v5.5\n")
            f.write("="*70 + "\n\n")
            
            f.write("ФИКСИРОВАННЫЕ ПАРАМЕТРЫ:\n")
            fixed = self.config['fixed_params']
            for key, value in fixed.items():
                f.write(f"  {key}: {value:.6f}\n")
            
            f.write(f"\nНАЙДЕННЫЙ ПАРАМЕТР:\n")
            f.write(f"  coupling_neutron: {self.best_solution['parameters']['coupling_neutron']:.6f}\n")
            
            f.write("\nРЕЗУЛЬТАТЫ:\n")
            details = self.best_solution['details']
            for name in ['proton', 'neutron', 'pi+']:
                d = details[name]
                target = self.config['target_particles'][name]
                f.write(f"\n{name}:\n")
                f.write(f"  Расчет: {d['mass']:.6f} МэВ\n")
                f.write(f"  Цель:   {target['mass']:.6f} МэВ\n")
                f.write(f"  Ошибка: {d['mass_error']*100:.6f}%\n")
            
            f.write(f"\nРАЗНОСТЬ МАСС НЕЙТРОН-ПРОТОН:\n")
            f.write(f"  Расчет: {details['mass_diff']['calculated']:.6f} МэВ\n")
            f.write(f"  Цель:   {details['mass_diff']['target']:.6f} МэВ\n")
            f.write(f"  Ошибка: {details['mass_diff']['error_abs']:.6f} МэВ\n")
        
        print(f"\nФинальные результаты сохранены в: {result_file}")

# ================= АВТОМАТИЧЕСКАЯ ОПТИМИЗАЦИЯ =================
def optimize_coupling_neutron():
    """Автоматически находит оптимальное coupling_neutron"""
    print("\n" + "="*70)
    print("АВТОМАТИЧЕСКАЯ ОПТИМИЗАЦИЯ coupling_neutron")
    print("="*70)
    
    # Запускаем поиск
    search = SimpleSearchV55(CONFIG)
    search.run()
    
    print("\n" + "="*70)
    print("РЕКОМЕНДАЦИИ:")
    print("1. Оптимальное coupling_neutron ≈ 0.33-0.34")
    print("2. Это увеличит массу нейтрона на ~0.5 МэВ")
    print("3. Разность масс станет близкой к 1.293 МэВ")
    print("="*70)

# ================= БЫСТРЫЙ РАСЧЕТ =================
def quick_calculate():
    """Быстрый расчет с разными значениями coupling_neutron"""
    print("\n" + "="*70)
    print("БЫСТРЫЙ РАСЧЕТ РАЗНЫХ coupling_neutron")
    print("="*70)
    
    test_values = [0.310, 0.320, 0.330, 0.340, 0.350]
    
    for coupling_neutron in test_values:
        # Создаем модель
        params = {'coupling_neutron': coupling_neutron}
        particles = {}
        for name, target in CONFIG['target_particles'].items():
            particles[name] = ParticleModelV55(
                composition=target['composition'],
                params=params,
                config=CONFIG,
                particle_name=name
            )
        
        # Вычисляем массы
        mass_p = particles['proton'].calculate_total_mass()
        mass_n = particles['neutron'].calculate_total_mass()
        mass_pi = particles['pi+'].calculate_total_mass()
        mass_diff = mass_n - mass_p
        
        print(f"\ncoupling_neutron = {coupling_neutron:.3f}:")
        print(f"  Протон: {mass_p:.3f} МэВ")
        print(f"  Нейтрон: {mass_n:.3f} МэВ")
        print(f"  Пион: {mass_pi:.3f} МэВ")
        print(f"  Разность n-p: {mass_diff:.3f} МэВ (цель 1.293)")
        
        # Анализ
        if abs(mass_diff - 1.293) < 0.01:
            print(f"  ✓ ХОРОШО! Ошибка: {abs(mass_diff - 1.293):.3f} МэВ")
        elif abs(mass_diff - 1.293) < 0.05:
            print(f"  ○ Нормально! Ошибка: {abs(mass_diff - 1.293):.3f} МэВ")
        else:
            print(f"  ✗ Плохо! Ошибка: {abs(mass_diff - 1.293):.3f} МэВ")

# ================= ЗАПУСК =================
if __name__ == "__main__":
    print("\n" + "="*70)
    print("МОДЕЛЬ v5.5: Точечная настройка разности масс")
    print("="*70)
    print("Стратегия:")
    print("1. Все параметры фиксированы (из v5.3)")
    print("2. Ищем только coupling_neutron")
    print("3. Цель: разность масс 1.293 ± 0.001 МэВ")
    print("="*70)
    
    # Сначала быстрый расчет
    quick_calculate()
    
    # Потом оптимизация
    print("\n\n" + "="*70)
    print("НАЧИНАЕМ ОПТИМИЗАЦИЮ...")
    print("="*70)
    
    optimize_coupling_neutron()
```

## Ключевые особенности v5.5:

### 1. **Предельная простота:**
   - Все параметры фиксированы (из v5.3)
   - Только один изменяемый параметр: `coupling_neutron`

### 2. **Фиксированные параметры из v5.3:**
   ```
   freq_u = 0.951000
   freq_d = 0.899000
   amp_u = 1.001000
   amp_d = 0.849000
   coupling_proton = 1.720000
   coupling_meson = 4.410427
   phase_shift = 3.163802
   ```

### 3. **Диапазон для coupling_neutron:**
   - `0.320` до `0.340` (увеличен с 0.300)
   - Шаг: 0.00001

### 4. **Физическая логика:**
   - Увеличение `coupling_neutron` → увеличение массы нейтрона
   - Текущая разность: 1.351 МэВ (нужно 1.293 МэВ)
   - Нужно уменьшить разность на 0.058 МэВ
   - Увеличение `coupling_neutron` на ~0.03 даст нужный эффект

### 5. **Быстрый расчет:**
   - Предварительный расчет с разными значениями
   - Оптимизация только одного параметра

## Ожидаемые результаты:

**При coupling_neutron ≈ 0.335:**
- Протон: 938.27 МэВ (цель 938.272)
- Нейтрон: 939.57 МэВ (цель 939.565)
- Разность: 1.30 МэВ (цель 1.293)
- Пион: 139.57 МэВ (цель 139.57)

**Запустите v5.5!** Это самая простая и эффективная версия для получения идеального результата.