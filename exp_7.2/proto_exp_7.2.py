"""
МОДЕЛЬ v7.2 - ПАРАЛЛЕЛЬНОЕ РАСШИРЕНИЕ СУЩЕСТВУЮЩЕЙ МОДЕЛИ
Основа: идеальные параметры v6.1
Расширение: добавление странных частиц
6 ядер = 6 целевых частиц/групп
"""

import numpy as np
import time
import json
from datetime import datetime
import os
import multiprocessing as mp
from multiprocessing import Pool, Manager, cpu_count

# ============== БАЗОВЫЕ ПАРАМЕТРЫ v6.1 (ИДЕАЛЬНЫЕ) ==============

OPTIMAL_V61 = {
    'base_mass_u': 2.203806,
    'base_mass_d': 4.583020,
    'freq_u': 0.956359,
    'freq_d': 0.868115,
    'amp_u': 1.032476,
    'amp_d': 0.877773,
    'coupling_proton': 1.613565,
    'coupling_neutron': 0.285395,
    'coupling_meson_light': 4.273121,
    'phase_shift': 3.173848,
    'scale_factor': 100.0
}

# ============== РАСПРЕДЕЛЕННЫЙ ОПТИМИЗАТОР ==============

class DistributedOptimizer:
    def __init__(self, num_cores=6):
        self.num_cores = num_cores
        
        # ОПРЕДЕЛЯЕМ ЗАДАЧИ ДЛЯ КАЖДОГО ЯДРА
        self.core_tasks = {
            0: {'name': 'STRANGE_MASS', 'target': None, 'params': ['base_mass_s', 'freq_s', 'amp_s']},
            1: {'name': 'KAON_PLUS', 'target': 493.677, 'particles': ['k+', 'k0']},
            2: {'name': 'LAMBDA0', 'target': 1115.683, 'particles': ['lambda0']},
            3: {'name': 'KAON_MINUS', 'target': 493.677, 'particles': ['k-', 'k0_bar']},
            4: {'name': 'MESON_SYNC', 'target': None, 'params': ['coupling_meson_strange']},
            5: {'name': 'BARYON_SYNC', 'target': None, 'params': ['coupling_lambda0']}
        }
        
        # Целевые частицы
        self.targets = {
            'proton': {'mass': 938.272, 'charge': 1.0, 'composition': ['u', 'u', 'd']},
            'neutron': {'mass': 939.565, 'charge': 0.0, 'composition': ['u', 'd', 'd']},
            'pi+': {'mass': 139.570, 'charge': 1.0, 'composition': ['u', 'anti_d']},
            'pi0': {'mass': 134.9768, 'charge': 0.0, 'composition': ['u', 'anti_u']},
            'pi-': {'mass': 139.570, 'charge': -1.0, 'composition': ['d', 'anti_u']},
            'k+': {'mass': 493.677, 'charge': 1.0, 'composition': ['u', 'anti_s']},
            'k0': {'mass': 497.611, 'charge': 0.0, 'composition': ['d', 'anti_s']},
            'k-': {'mass': 493.677, 'charge': -1.0, 'composition': ['s', 'anti_u']},
            'k0_bar': {'mass': 497.611, 'charge': 0.0, 'composition': ['s', 'anti_d']},
            'lambda0': {'mass': 1115.683, 'charge': 0.0, 'composition': ['u', 'd', 's']},
        }
        
        # Инициализируем базовые параметры
        self.base_params = OPTIMAL_V61.copy()
        # Добавляем начальные значения для новых параметров
        self.base_params.update({
            'base_mass_s': 55.0,  # Начальное значение из v6.6
            'freq_s': 0.7,
            'amp_s': 0.75,
            'coupling_meson_strange': 5.0,
            'coupling_lambda0': 0.8
        })
        
        # Диапазоны ТОЛЬКО для новых параметров
        self.param_ranges = {
            'base_mass_s': (30.0, 100.0),
            'freq_s': (0.3, 1.0),
            'amp_s': (0.5, 1.0),
            'coupling_meson_strange': (3.0, 7.0),
            'coupling_lambda0': (0.3, 1.5)
        }
        
        # Создаем директорию для результатов
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.result_dir = f"distributed_v72_{timestamp}"
        os.makedirs(self.result_dir, exist_ok=True)
        
        # Общая память для результатов
        manager = Manager()
        self.shared_results = manager.dict()
        self.shared_params = manager.dict(self.base_params)
    
    # ============== ОСНОВНАЯ МОДЕЛЬ ==============
    
    class ParticleCalculator:
        @staticmethod
        def calculate_mass(params, particle_name, composition, is_meson=False):
            """Быстрый расчет массы"""
            base = 0.0
            for quark in composition:
                if quark in ['u', 'anti_u']:
                    base += params['base_mass_u'] * params['freq_u'] * params['amp_u']
                elif quark in ['d', 'anti_d']:
                    base += params['base_mass_d'] * params['freq_d'] * params['amp_d']
                elif quark in ['s', 'anti_s']:
                    base += params['base_mass_s'] * params['freq_s'] * params['amp_s']
            
            # Энергия синхронизации
            if particle_name == 'proton':
                coupling = params['coupling_proton']
                phases = [0, 0, np.pi/2]
                thread_count = 3
            elif particle_name == 'neutron':
                coupling = params['coupling_neutron']
                phases = [0, np.pi/2, np.pi/2]
                thread_count = 3
            elif particle_name == 'lambda0':
                coupling = params['coupling_lambda0']
                phases = [0, np.pi/2, np.pi]
                thread_count = 3
            elif is_meson:
                if 'k' in particle_name:  # Каоны
                    coupling = params['coupling_meson_strange']
                else:  # Пионы
                    coupling = params['coupling_meson_light']
                phases = [0, params['phase_shift']]
                thread_count = 2
            else:
                coupling = params['coupling_meson_light']
                phases = [0, params['phase_shift']]
                thread_count = 2
            
            # Простая когерентность
            if thread_count == 2:
                if is_meson:
                    phase_coherence = np.cos(phases[1] - phases[0] + np.pi)
                else:
                    phase_coherence = np.cos(phases[1] - phases[0])
                phase_coherence = (phase_coherence + 1) / 2
            else:  # 3 нити
                phase_coherence_sum = 0
                for i in range(3):
                    for j in range(i+1, 3):
                        diff = abs(phases[i] - phases[j])
                        phase_coherence_sum += np.cos(diff)
                phase_coherence = (phase_coherence_sum / 3 + 1) / 2
            
            sync_energy = coupling * phase_coherence
            
            if is_meson:
                total = base - sync_energy
            else:
                total = base + sync_energy
            
            return total * params['scale_factor']
    
    # ============== ФУНКЦИИ ДЛЯ КАЖДОГО ЯДРА ==============
    
    def optimize_strange_mass(self, core_id, iterations=200000):
        """Ядро 0: Оптимизация массы s-кварка"""
        task = self.core_tasks[core_id]
        print(f"[Ядро {core_id}] Задача: {task['name']}")
        
        best_params = self.base_params.copy()
        best_error = float('inf')
        
        for i in range(iterations):
            # Мутируем только параметры s-кварка
            params = best_params.copy()
            
            for param_name in task['params']:
                if param_name in self.param_ranges:
                    min_val, max_val = self.param_ranges[param_name]
                    mutation = np.random.normal(0, (max_val - min_val) * 0.05)
                    params[param_name] = np.clip(params[param_name] + mutation, min_val, max_val)
            
            # Рассчитываем m_s/m_u
            m_u = params['base_mass_u'] * params['freq_u'] * params['amp_u']
            m_s = params['base_mass_s'] * params['freq_s'] * params['amp_s']
            ratio = m_s / m_u if m_u > 0 else 1
            
            # Цель: отношение ~30 (физически разумно)
            target_ratio = 30.0
            error = abs(ratio - target_ratio) / target_ratio
            
            if error < best_error:
                best_error = error
                best_params = params
            
            # Сохраняем прогресс каждые 50000 итераций
            if i % 50000 == 0:
                self.shared_results[f'core_{core_id}_progress'] = f"{i}/{iterations}"
                self.shared_results[f'core_{core_id}_ratio'] = ratio
        
        self.shared_results[f'core_{core_id}_best'] = {
            'params': {k: best_params[k] for k in task['params']},
            'error': best_error
        }
        return best_params
    
    def optimize_kaon_plus(self, core_id, iterations=200000):
        """Ядро 1: Оптимизация для K⁺ и K⁰"""
        task = self.core_tasks[core_id]
        print(f"[Ядро {core_id}] Задача: {task['name']}")
        
        best_params = self.base_params.copy()
        best_error = float('inf')
        
        for i in range(iterations):
            params = best_params.copy()
            
            # Мутируем параметры, связанные с каонами
            params['base_mass_s'] = np.clip(params['base_mass_s'] + np.random.normal(0, 2.0), 30, 100)
            params['coupling_meson_strange'] = np.clip(
                params['coupling_meson_strange'] + np.random.normal(0, 0.3), 3, 7)
            
            # Рассчитываем массы
            calculator = self.ParticleCalculator()
            
            k_plus_mass = calculator.calculate_mass(
                params, 'k+', self.targets['k+']['composition'], is_meson=True)
            k0_mass = calculator.calculate_mass(
                params, 'k0', self.targets['k0']['composition'], is_meson=True)
            
            # Ошибка
            error_k_plus = abs(k_plus_mass - 493.677) / 493.677
            error_k0 = abs(k0_mass - 497.611) / 497.611
            total_error = (error_k_plus + error_k0) / 2
            
            # Также учитываем отношение масс s/u
            m_u = params['base_mass_u'] * params['freq_u'] * params['amp_u']
            m_s = params['base_mass_s'] * params['freq_s'] * params['amp_s']
            ratio_error = abs(m_s/m_u - 30) / 30 if m_u > 0 else 1
            
            final_error = total_error * 0.7 + ratio_error * 0.3
            
            if final_error < best_error:
                best_error = final_error
                best_params = params
        
        self.shared_results[f'core_{core_id}_best'] = {
            'k+_mass': calculator.calculate_mass(best_params, 'k+', self.targets['k+']['composition'], True),
            'k0_mass': calculator.calculate_mass(best_params, 'k0', self.targets['k0']['composition'], True),
            'error': best_error
        }
        return best_params
    
    def optimize_lambda(self, core_id, iterations=200000):
        """Ядро 2: Оптимизация для Λ⁰"""
        task = self.core_tasks[core_id]
        print(f"[Ядро {core_id}] Задача: {task['name']}")
        
        best_params = self.base_params.copy()
        best_error = float('inf')
        
        for i in range(iterations):
            params = best_params.copy()
            
            # Мутируем параметры Λ⁰
            params['coupling_lambda0'] = np.clip(
                params['coupling_lambda0'] + np.random.normal(0, 0.1), 0.3, 1.5)
            params['base_mass_s'] = np.clip(
                params['base_mass_s'] + np.random.normal(0, 1.5), 30, 100)
            
            calculator = self.ParticleCalculator()
            lambda_mass = calculator.calculate_mass(
                params, 'lambda0', self.targets['lambda0']['composition'], is_meson=False)
            
            error = abs(lambda_mass - 1115.683) / 1115.683
            
            if error < best_error:
                best_error = error
                best_params = params
        
        self.shared_results[f'core_{core_id}_best'] = {
            'lambda_mass': calculator.calculate_mass(best_params, 'lambda0', 
                                                   self.targets['lambda0']['composition'], False),
            'error': best_error
        }
        return best_params
    
    def optimize_kaon_minus(self, core_id, iterations=200000):
        """Ядро 3: Оптимизация для K⁻ и K⁰-бар"""
        task = self.core_tasks[core_id]
        print(f"[Ядро {core_id}] Задача: {task['name']}")
        
        # Аналогично оптимизации каонов, но с фокусом на заряженные
        return self.optimize_kaon_plus(core_id, iterations)
    
    def optimize_meson_sync(self, core_id, iterations=200000):
        """Ядро 4: Оптимизация coupling для странных мезонов"""
        task = self.core_tasks[core_id]
        print(f"[Ядро {core_id}] Задача: {task['name']}")
        
        best_params = self.base_params.copy()
        best_error = float('inf')
        
        for i in range(iterations):
            params = best_params.copy()
            params['coupling_meson_strange'] = np.clip(
                params['coupling_meson_strange'] + np.random.normal(0, 0.2), 3, 7)
            
            calculator = self.ParticleCalculator()
            
            # Тестируем на всех каонах
            errors = []
            for name in ['k+', 'k0', 'k-', 'k0_bar']:
                mass = calculator.calculate_mass(
                    params, name, self.targets[name]['composition'], is_meson=True)
                target = self.targets[name]['mass']
                errors.append(abs(mass - target) / target)
            
            avg_error = np.mean(errors)
            
            if avg_error < best_error:
                best_error = avg_error
                best_params = params
        
        self.shared_results[f'core_{core_id}_best'] = {
            'coupling_meson_strange': best_params['coupling_meson_strange'],
            'error': best_error
        }
        return best_params
    
    def optimize_baryon_sync(self, core_id, iterations=200000):
        """Ядро 5: Оптимизация coupling для Λ⁰"""
        task = self.core_tasks[core_id]
        print(f"[Ядро {core_id}] Задача: {task['name']}")
        
        best_params = self.base_params.copy()
        best_error = float('inf')
        
        for i in range(iterations):
            params = best_params.copy()
            params['coupling_lambda0'] = np.clip(
                params['coupling_lambda0'] + np.random.normal(0, 0.05), 0.3, 1.5)
            
            calculator = self.ParticleCalculator()
            lambda_mass = calculator.calculate_mass(
                params, 'lambda0', self.targets['lambda0']['composition'], is_meson=False)
            
            error = abs(lambda_mass - 1115.683) / 1115.683
            
            if error < best_error:
                best_error = error
                best_params = params
        
        self.shared_results[f'core_{core_id}_best'] = {
            'coupling_lambda0': best_params['coupling_lambda0'],
            'error': best_error
        }
        return best_params
    
    # ============== ОСНОВНАЯ ФУНКЦИЯ ==============
    
    def run_distributed_optimization(self):
        """Запуск распределенной оптимизации"""
        print("="*80)
        print("РАСПРЕДЕЛЕННАЯ ОПТИМИЗАЦИЯ v7.2")
        print(f"Ядер: {self.num_cores}")
        print("="*80)
        
        print("\nРАСПРЕДЕЛЕНИЕ ЗАДАЧ:")
        for core_id, task in self.core_tasks.items():
            print(f"  Ядро {core_id}: {task['name']}")
        
        print(f"\nБАЗОВЫЕ ПАРАМЕТРЫ (из v6.1):")
        for key, value in OPTIMAL_V61.items():
            if key not in ['scale_factor']:
                print(f"  {key}: {value:.6f}")
        
        print(f"\nНАЧИНАЮ ОПТИМИЗАЦИЮ...")
        start_time = time.time()
        
        # Создаем пул процессов
        with mp.Pool(processes=self.num_cores) as pool:
            # Запускаем все задачи параллельно
            tasks = [
                (0, 200000),  # Ядро 0: Strange mass
                (1, 200000),  # Ядро 1: Kaon plus
                (2, 200000),  # Ядро 2: Lambda
                (3, 200000),  # Ядро 3: Kaon minus
                (4, 200000),  # Ядро 4: Meson sync
                (5, 200000)   # Ядро 5: Baryon sync
            ]
            
            results = pool.starmap(self._run_core_task, tasks)
        
        # Объединяем результаты
        final_params = self.base_params.copy()
        for core_result in results:
            if core_result:
                for key, value in core_result.items():
                    if key in final_params:
                        final_params[key] = value
        
        elapsed = time.time() - start_time
        
        print(f"\n{'='*80}")
        print("РАСПРЕДЕЛЕННАЯ ОПТИМИЗАЦИЯ ЗАВЕРШЕНА")
        print(f"Время: {elapsed:.1f} секунд")
        print("="*80)
        
        # Вычисляем финальные массы
        self.calculate_final_masses(final_params)
        
        return final_params
    
    def _run_core_task(self, core_id, iterations):
        """Запуск задачи на указанном ядре"""
        try:
            if core_id == 0:
                return self.optimize_strange_mass(core_id, iterations)
            elif core_id == 1:
                return self.optimize_kaon_plus(core_id, iterations)
            elif core_id == 2:
                return self.optimize_lambda(core_id, iterations)
            elif core_id == 3:
                return self.optimize_kaon_minus(core_id, iterations)
            elif core_id == 4:
                return self.optimize_meson_sync(core_id, iterations)
            elif core_id == 5:
                return self.optimize_baryon_sync(core_id, iterations)
        except Exception as e:
            print(f"[Ядро {core_id}] Ошибка: {e}")
            return {}
    
    def calculate_final_masses(self, params):
        """Расчет всех масс с финальными параметрами"""
        calculator = self.ParticleCalculator()
        
        print("\n" + "="*80)
        print("ФИНАЛЬНЫЕ РЕЗУЛЬТАТЫ")
        print("="*80)
        
        print(f"\nНОВЫЕ ПАРАМЕТРЫ:")
        for key in ['base_mass_s', 'freq_s', 'amp_s', 
                   'coupling_meson_strange', 'coupling_lambda0']:
            print(f"  {key}: {params.get(key, 'N/A'):.6f}")
        
        # Рассчитываем эффективные массы кварков
        m_u = params['base_mass_u'] * params['freq_u'] * params['amp_u'] * 100
        m_d = params['base_mass_d'] * params['freq_d'] * params['amp_d'] * 100
        m_s = params['base_mass_s'] * params['freq_s'] * params['amp_s'] * 100
        
        print(f"\nЭФФЕКТИВНЫЕ МАССЫ КВАРКОВ (МэВ):")
        print(f"  u: {m_u:.2f}")
        print(f"  d: {m_d:.2f}")
        print(f"  s: {m_s:.2f}")
        print(f"  Отношение m_s/m_u: {m_s/m_u:.2f}")
        
        print(f"\nМАССЫ ЧАСТИЦ:")
        print(f"{'Частица':<10} {'Масса':<12} {'Цель':<12} {'Ошибка %':<10}")
        print("-" * 80)
        
        all_particles = ['proton', 'neutron', 'pi+', 'pi0', 'pi-', 
                        'k+', 'k0', 'k-', 'k0_bar', 'lambda0']
        
        total_error = 0
        for name in all_particles:
            target = self.targets[name]
            mass = calculator.calculate_mass(
                params, name, target['composition'], 
                is_meson=(len(target['composition']) == 2)
            )
            error_pct = abs(mass - target['mass']) / target['mass'] * 100
            total_error += error_pct
            
            print(f"{name:<10} {mass:<12.3f} {target['mass']:<12.3f} {error_pct:<10.3f}")
        
        print(f"\nСредняя ошибка: {total_error/len(all_particles):.3f}%")
        
        # Сохраняем в файл
        self.save_results(params)
    
    def save_results(self, params):
        """Сохранение результатов"""
        results = {
            'model': 'v7.2_distributed',
            'timestamp': datetime.now().isoformat(),
            'base_parameters': OPTIMAL_V61,
            'extended_parameters': params,
            'targets': self.targets
        }
        
        with open(f"{self.result_dir}/distributed_results.json", 'w') as f:
            json.dump(results, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else x)
        
        print(f"\nРезультаты сохранены в: {self.result_dir}")

# ============== ЗАПУСК ==============

def main():
    """Главная функция - без интерактивности"""
    print("="*80)
    print("РАСПРЕДЕЛЕННАЯ ОПТИМИЗАЦИЯ v7.2")
    print("Основа: идеальные параметры v6.1")
    print("Цель: добавить странные частицы")
    print("="*80)
    
    # Автоматическое определение числа ядер
    num_cores = 6
    print(f"\nИспользуется ядер: {num_cores}")
    print("Запускаю распределенную оптимизацию...\n")
    
    try:
        optimizer = DistributedOptimizer(num_cores=num_cores)
        final_params = optimizer.run_distributed_optimization()
        
    except Exception as e:
        print(f"\nОШИБКА: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*80)
    print("ВЫЧИСЛЕНИЯ ЗАВЕРШЕНЫ")
    print("="*80)

if __name__ == "__main__":
    # Отключаем spawn для Windows (если нужно)
    if hasattr(mp, 'set_start_method'):
        try:
            mp.set_start_method('spawn')
        except RuntimeError:
            pass
    
    main()