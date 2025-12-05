"""
МОДЕЛЬ v7.1 - ПАРАЛЛЕЛЬНАЯ ОПТИМИЗАЦИЯ
Единая экспоненциальная модель синхронизации с параллельными вычислениями
"""

import numpy as np
import time
import json
from datetime import datetime
import os
import sys
import multiprocessing as mp
from multiprocessing import Pool, Manager
import warnings
warnings.filterwarnings('ignore')

# ============== ОСНОВНАЯ МОДЕЛЬ (без изменений) ==============

class ParticleModelV71:
    def __init__(self, params, particle_name, composition):
        self.params = params
        self.particle_name = particle_name
        self.composition = composition
        self.is_meson = len(composition) == 2
        self.scale = 100.0
        
        # Параметры напрямую из словаря
        for key, value in params.items():
            setattr(self, key, value)
    
    def get_quark_params(self, quark):
        quark_type = quark.replace('anti_', '')
        
        if quark_type == 'u':
            return self.base_mass_u, self.freq_u, self.amp_u
        elif quark_type == 'd':
            return self.base_mass_d, self.freq_d, self.amp_d
        elif quark_type == 's':
            return self.base_mass_s, self.freq_s, self.amp_s
        else:
            return 1.0, 1.0, 1.0
    
    def calculate_base_mass(self):
        total = 0.0
        for quark in self.composition:
            base_mass, freq, amp = self.get_quark_params(quark)
            total += base_mass * freq * amp
        return total
    
    def calculate_coherence(self):
        if self.is_meson:
            phases = [0, self.phase_shift]
            freqs = []
            for quark in self.composition:
                _, freq, _ = self.get_quark_params(quark)
                freqs.append(freq)
        else:
            if self.particle_name == 'proton':
                phases = [0, 0, np.pi/2]
            elif self.particle_name == 'neutron':
                phases = [0, np.pi/2, np.pi/2]
            elif self.particle_name == 'lambda0':
                phases = [0, np.pi/2, np.pi]
            else:
                phases = [0, np.pi/4, np.pi/2]
            
            freqs = []
            for quark in self.composition:
                _, freq, _ = self.get_quark_params(quark)
                freqs.append(freq)
        
        # Частотная когерентность
        if len(freqs) > 1:
            freq_std = np.std(freqs)
            freq_coherence = 1.0 / (1.0 + freq_std)
        else:
            freq_coherence = 1.0
        
        # Фазовая когерентность
        if len(phases) >= 2:
            phase_coherence_sum = 0.0
            pair_count = 0
            
            for i in range(len(phases)):
                for j in range(i+1, len(phases)):
                    phase_diff = abs(phases[i] - phases[j]) % (2*np.pi)
                    phase_diff = min(phase_diff, 2*np.pi - phase_diff)
                    
                    if self.is_meson:
                        phase_coherence_sum += np.cos(phase_diff + np.pi)
                    else:
                        phase_coherence_sum += np.cos(phase_diff)
                    
                    pair_count += 1
            
            phase_coherence = (phase_coherence_sum / pair_count + 1.0) / 2.0
        else:
            phase_coherence = 0.5
        
        total_coherence = (self.coherence_weights[0] * freq_coherence + 
                          self.coherence_weights[1] * phase_coherence)
        return np.clip(total_coherence, 0.0, 1.0)
    
    def calculate_sync_factor(self):
        coherence = self.calculate_coherence()
        
        if self.is_meson:
            return np.exp(self.coupling_meson * coherence)
        else:
            return np.exp(self.coupling_baryon * coherence)
    
    def calculate_mass(self):
        base_mass = self.calculate_base_mass()
        sync_factor = self.calculate_sync_factor()
        return base_mass * sync_factor * self.scale
    
    def calculate_charge(self):
        charges = {
            'u': 2/3, 'd': -1/3, 's': -1/3,
            'anti_u': -2/3, 'anti_d': 1/3, 'anti_s': 1/3
        }
        
        total = 0.0
        for quark in self.composition:
            total += charges.get(quark, 0.0)
        
        return round(total, 10)

# ============== ПАРАЛЛЕЛЬНЫЙ ОПТИМИЗАТОР ==============

class ParallelOptimizerV71:
    def __init__(self, num_cores=6):
        self.num_cores = num_cores
        self.target_particles = {
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
        
        # Базовые диапазоны параметров (физически обоснованные)
        self.param_ranges = {
            'base_mass_u': (1.5, 3.0),      # u-кварк легкий
            'base_mass_d': (2.5, 4.5),      # d-кварк немного тяжелее
            'base_mass_s': (40.0, 120.0),   # s-кварк значительно тяжелее
            
            'freq_u': (0.8, 1.2),
            'freq_d': (0.8, 1.2),
            'freq_s': (0.3, 0.8),
            
            'amp_u': (0.7, 1.3),
            'amp_d': (0.7, 1.3),
            'amp_s': (0.5, 1.1),
            
            'coupling_meson': (-3.0, -0.5),    # Отрицательный для мезонов
            'coupling_baryon': (0.05, 0.5),    # Положительный для барионов
            
            'phase_shift': (2.8, 3.4),        # Около π
            
            'coherence_weights': [(0.3, 0.7), (0.3, 0.7)]
        }
        
        # Лучшие результаты
        self.best_params = None
        self.best_error = float('inf')
        self.best_results = None
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.result_dir = f"parallel_v71_{timestamp}"
        os.makedirs(self.result_dir, exist_ok=True)
    
    def generate_random_params(self, seed=None):
        """Генерация случайных параметров в заданных диапазонах"""
        if seed is not None:
            np.random.seed(seed)
        
        params = {}
        for key, (min_val, max_val) in self.param_ranges.items():
            if key == 'coherence_weights':
                w1 = np.random.uniform(0.3, 0.7)
                w2 = 1.0 - w1
                params[key] = [w1, w2]
            elif key == 'coupling_meson':
                # Гарантированно отрицательный
                params[key] = -np.random.uniform(0.5, 3.0)
            elif key == 'coupling_baryon':
                # Гарантированно положительный
                params[key] = np.random.uniform(0.05, 0.5)
            else:
                params[key] = np.random.uniform(min_val, max_val)
        
        return params
    
    def evaluate_params(self, params):
        """Быстрая оценка параметров (без сохранения моделей)"""
        # Создаем все модели один раз
        models = {}
        for name, target in self.target_particles.items():
            models[name] = ParticleModelV71(params, name, target['composition'])
        
        # Рассчитываем массы
        results = {}
        for name, model in models.items():
            results[f'{name}_mass'] = model.calculate_mass()
            results[f'{name}_charge'] = model.calculate_charge()
        
        # Эффективные массы кварков
        m_u_eff = params['base_mass_u'] * params['freq_u'] * params['amp_u'] * 100
        m_d_eff = params['base_mass_d'] * params['freq_d'] * params['amp_d'] * 100
        m_s_eff = params['base_mass_s'] * params['freq_s'] * params['amp_s'] * 100
        
        results['m_u_eff_mev'] = m_u_eff
        results['m_d_eff_mev'] = m_d_eff
        results['m_s_eff_mev'] = m_s_eff
        results['mass_ratio_d_u'] = m_d_eff / m_u_eff if m_u_eff > 0 else 1.0
        results['mass_ratio_s_u'] = m_s_eff / m_u_eff if m_u_eff > 0 else 1.0
        
        return results
    
    def calculate_error(self, params, return_results=False):
        """Быстрый расчет ошибки"""
        results = self.evaluate_params(params)
        total_error = 0.0
        
        # Ошибки масс (квадратичные)
        for name, target in self.target_particles.items():
            target_mass = target['mass']
            calculated_mass = results[f'{name}_mass']
            rel_error = abs(calculated_mass - target_mass) / target_mass
            
            # Веса
            if name in ['proton', 'neutron']:
                weight = 30.0
            elif name in ['k+', 'k0', 'lambda0']:
                weight = 20.0
            elif name in ['pi+', 'pi0', 'pi-']:
                weight = 15.0
            else:
                weight = 5.0
            
            total_error += weight * rel_error ** 2
        
        # Ошибки зарядов
        for name, target in self.target_particles.items():
            if abs(results[f'{name}_charge'] - target['charge']) > 0.001:
                total_error += 500.0
        
        # Физические ограничения
        # 1. coupling_meson < 0
        if params['coupling_meson'] >= 0:
            total_error += 1000.0
        
        # 2. coupling_baryon > 0
        if params['coupling_baryon'] <= 0:
            total_error += 1000.0
        
        # 3. Нейтрон тяжелее протона
        if results['neutron_mass'] < results['proton_mass']:
            total_error += 300.0
        
        # 4. m_s/m_u в разумных пределах (15-40)
        ratio_s_u = results['mass_ratio_s_u']
        if ratio_s_u < 10 or ratio_s_u > 50:
            penalty = abs(ratio_s_u - 30) * 5.0
            total_error += penalty
        
        if return_results:
            return total_error, results
        return total_error
    
    def run_single_annealing(self, seed, iterations=500000, temperature=5.0):
        """Метод отжига для одного процесса"""
        np.random.seed(seed)
        
        # Начальные параметры
        current_params = self.generate_random_params(seed)
        current_error = self.calculate_error(current_params)
        
        best_params = current_params.copy()
        best_error = current_error
        best_results = None
        
        cooling_rate = 0.99999
        
        for i in range(iterations):
            # Мутация параметров
            new_params = current_params.copy()
            for key in current_params.keys():
                if key in self.param_ranges:
                    min_val, max_val = self.param_ranges[key]
                    
                    if key == 'coherence_weights':
                        # Мутация весов
                        w1 = current_params[key][0] + np.random.normal(0, 0.03)
                        w1 = max(0.1, min(0.9, w1))
                        w2 = 1.0 - w1
                        new_params[key] = [w1, w2]
                    else:
                        # Мутация обычного параметра
                        step = (max_val - min_val) * 0.05
                        mutation = np.random.normal(0, step) * temperature
                        new_val = current_params[key] + mutation
                        
                        # Ограничение диапазона
                        while new_val < min_val or new_val > max_val:
                            if new_val < min_val:
                                new_val = 2 * min_val - new_val
                            if new_val > max_val:
                                new_val = 2 * max_val - new_val
                        
                        new_params[key] = new_val
            
            # Оценка новых параметров
            new_error = self.calculate_error(new_params)
            
            # Принятие решения
            if new_error < current_error:
                current_params = new_params
                current_error = new_error
            else:
                delta = new_error - current_error
                probability = np.exp(-delta / temperature)
                if np.random.random() < probability:
                    current_params = new_params
                    current_error = new_error
            
            # Обновление лучшего
            if new_error < best_error:
                best_params = new_params.copy()
                best_error = new_error
            
            # Охлаждение
            temperature *= cooling_rate
        
        # Получаем результаты для лучших параметров
        best_error, best_results = self.calculate_error(best_params, return_results=True)
        
        return {
            'seed': seed,
            'best_params': best_params,
            'best_error': best_error,
            'best_results': best_results,
            'iterations': iterations
        }
    
    def run_parallel_optimization(self, total_iterations=3000000):
        """Параллельная оптимизация на всех ядрах"""
        print("="*80)
        print(f"ПАРАЛЛЕЛЬНАЯ ОПТИМИЗАЦИЯ v7.1")
        print(f"Ядер процессора: {self.num_cores}")
        print(f"Всего итераций: {total_iterations:,}")
        print("="*80)
        
        start_time = time.time()
        
        # Разделяем итерации между ядрами
        iterations_per_core = total_iterations // self.num_cores
        seeds = list(range(1000, 1000 + self.num_cores))
        
        print(f"\nКонфигурация:")
        print(f"  Итераций на ядро: {iterations_per_core:,}")
        print(f"  Seeds: {seeds}")
        print(f"  Запускаю {self.num_cores} процессов...\n")
        
        # Запускаем параллельные процессы
        with Pool(processes=self.num_cores) as pool:
            results = pool.starmap(self.run_single_annealing, 
                                  [(seed, iterations_per_core, 5.0) for seed in seeds])
        
        # Находим лучший результат среди всех процессов
        best_overall = min(results, key=lambda x: x['best_error'])
        
        elapsed = time.time() - start_time
        
        print(f"\n{'='*80}")
        print("ПАРАЛЛЕЛЬНАЯ ОПТИМИЗАЦИЯ ЗАВЕРШЕНА")
        print(f"Общее время: {elapsed:.1f} сек")
        print(f"Лучшая ошибка: {best_overall['best_error']:.3f}")
        print(f"Лучший seed: {best_overall['seed']}")
        print("="*80)
        
        self.best_params = best_overall['best_params']
        self.best_error = best_overall['best_error']
        self.best_results = best_overall['best_results']
        
        # Сохраняем результаты
        self.save_results(results, best_overall)
        
        return self.best_params, self.best_error, self.best_results
    
    def save_results(self, all_results, best_result):
        """Сохранение всех результатов"""
        # Основные результаты
        main_results = {
            'model_version': 'v7.1_parallel',
            'optimization_info': {
                'total_cores': self.num_cores,
                'total_iterations': sum(r['iterations'] for r in all_results),
                'best_error': best_result['best_error'],
                'best_seed': best_result['seed'],
                'timestamp': datetime.now().isoformat()
            },
            'best_parameters': best_result['best_params'],
            'best_results': best_result['best_results'],
            'all_results': [
                {
                    'seed': r['seed'],
                    'error': r['best_error'],
                    'iterations': r['iterations']
                } for r in all_results
            ]
        }
        
        with open(f"{self.result_dir}/parallel_results.json", 'w') as f:
            json.dump(main_results, f, indent=2, default=self.json_serializer)
        
        # Текстовый отчет
        self.save_text_report(best_result)
    
    def save_text_report(self, best_result):
        """Сохранение текстового отчета"""
        filename = f"{self.result_dir}/REPORT.txt"
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("ПАРАЛЛЕЛЬНАЯ МОДЕЛЬ v7.1 - ФИНАЛЬНЫЙ ОТЧЕТ\n")
            f.write("="*80 + "\n\n")
            
            f.write("ИНФОРМАЦИЯ ОБ ОПТИМИЗАЦИИ:\n")
            f.write(f"  Ядер процессора: {self.num_cores}\n")
            f.write(f"  Лучший seed: {best_result['seed']}\n")
            f.write(f"  Лучшая ошибка: {best_result['best_error']:.3f}\n\n")
            
            f.write("ЛУЧШИЕ ПАРАМЕТРЫ:\n")
            params = best_result['best_params']
            for key, value in params.items():
                if key == 'coherence_weights':
                    f.write(f"  {key}: [{value[0]:.4f}, {value[1]:.4f}]\n")
                else:
                    f.write(f"  {key}: {value:.6f}\n")
            
            f.write("\nМАССЫ ЧАСТИЦ:\n")
            f.write(f"{'Частица':<10} {'Масса (МэВ)':<12} {'Цель':<12} {'Ошибка (%)':<10}\n")
            f.write("-"*80 + "\n")
            
            key_particles = ['proton', 'neutron', 'pi+', 'k+', 'lambda0']
            for name in key_particles:
                target = self.target_particles[name]['mass']
                mass = best_result['best_results'].get(f'{name}_mass', 0)
                error_pct = abs(mass - target) / target * 100 if target > 0 else 0
                f.write(f"{name:<10} {mass:<12.3f} {target:<12.3f} {error_pct:<10.3f}\n")
            
            f.write("\nФИЗИЧЕСКИЕ ПАРАМЕТРЫ:\n")
            f.write(f"  Эффективная масса u-кварка: {best_result['best_results'].get('m_u_eff_mev', 0):.2f} МэВ\n")
            f.write(f"  Эффективная масса d-кварка: {best_result['best_results'].get('m_d_eff_mev', 0):.2f} МэВ\n")
            f.write(f"  Эффективная масса s-кварка: {best_result['best_results'].get('m_s_eff_mev', 0):.2f} МэВ\n")
            f.write(f"  Отношение m_d/m_u: {best_result['best_results'].get('mass_ratio_d_u', 0):.3f}\n")
            f.write(f"  Отношение m_s/m_u: {best_result['best_results'].get('mass_ratio_s_u', 0):.3f}\n")
            
            f.write("\nПРОВЕРКА ФИЗИЧЕСКИХ ОГРАНИЧЕНИЙ:\n")
            n_mass = best_result['best_results'].get('neutron_mass', 0)
            p_mass = best_result['best_results'].get('proton_mass', 0)
            f.write(f"  Нейтрон тяжелее протона: {n_mass > p_mass} (разность: {n_mass-p_mass:.3f} МэВ)\n")
            
            c_meson = params.get('coupling_meson', 0)
            c_baryon = params.get('coupling_baryon', 0)
            f.write(f"  coupling_meson < 0: {c_meson < 0} ({c_meson:.4f})\n")
            f.write(f"  coupling_baryon > 0: {c_baryon > 0} ({c_baryon:.4f})\n")
    
    def json_serializer(self, obj):
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return str(obj)

# ============== ГЛАВНАЯ ФУНКЦИЯ ==============

def main():
    """Главная функция без интерактивности"""
    print("="*80)
    print("ПАРАЛЛЕЛЬНАЯ ОПТИМИЗАЦИЯ МОДЕЛИ СИНХРОНИЗАЦИИ")
    print("Версия 7.1 - Экспоненциальная модель с 6 ядрами")
    print("="*80)
    
    # Автоматическое определение числа ядер
    num_cores = 6  # Фиксируем 6 ядер по твоей просьбе
    print(f"Используется ядер: {num_cores}")
    
    # Создаем оптимизатор
    optimizer = ParallelOptimizerV71(num_cores=num_cores)
    
    # Запускаем оптимизацию
    try:
        best_params, best_error, best_results = optimizer.run_parallel_optimization(
            total_iterations=3000000  # 3 миллиона итераций всего
        )
        
        # Выводим краткий отчет
        print("\nКРАТКИЙ ОТЧЕТ:")
        print(f"Лучшая ошибка: {best_error:.3f}")
        print(f"Отношение m_s/m_u: {best_results.get('mass_ratio_s_u', 0):.2f}")
        
        key_particles = ['proton', 'neutron', 'pi+', 'k+', 'lambda0']
        for name in key_particles:
            target = optimizer.target_particles[name]['mass']
            mass = best_results.get(f'{name}_mass', 0)
            error_pct = abs(mass - target) / target * 100
            print(f"  {name}: {mass:.1f} МэВ (цель {target:.1f}) - {error_pct:.1f}%")
        
        print(f"\nРезультаты сохранены в: {optimizer.result_dir}")
        
    except Exception as e:
        print(f"\nОШИБКА: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*80)
    print("ВЫЧИСЛЕНИЯ ЗАВЕРШЕНЫ")
    print("="*80)

# ============== ЗАПУСК ==============

if __name__ == "__main__":
    # Простой запуск без аргументов
    main()