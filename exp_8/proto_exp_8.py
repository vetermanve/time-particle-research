"""
МОДЕЛЬ v8.0 - ЦЕЛЕВОЙ ОТЖИГ С ИСПРАВЛЕНИЕМ π⁰ И s-КВАРКА
Основа: v6.1 (идеальные параметры)
Цель: исправить π⁰ и странные частицы
"""

import numpy as np
import time
import json
from datetime import datetime
import os
import multiprocessing as mp
from multiprocessing import Pool, cpu_count

# ============== БАЗОВАЯ МОДЕЛЬ С ИСПРАВЛЕНИЯМИ ==============

class FixedParticleModel:
    """Исправленная модель с отдельными правилами для разных типов частиц"""
    
    def __init__(self, params):
        self.params = params
        self.scale = params.get('scale_factor', 100.0)
        
    def calculate_mass(self, particle_name, composition):
        """Расчет массы с учётом типа частицы"""
        is_meson = len(composition) == 2
        has_strange = any('s' in q for q in composition)
        is_neutral_meson = (particle_name in ['pi0'])
        
        # БАЗОВАЯ МАССА
        base = 0.0
        for quark in composition:
            if quark in ['u', 'anti_u']:
                base += self.params['base_mass_u'] * self.params['freq_u'] * self.params['amp_u']
            elif quark in ['d', 'anti_d']:
                base += self.params['base_mass_d'] * self.params['freq_d'] * self.params['amp_d']
            elif quark in ['s', 'anti_s']:
                base += self.params['base_mass_s'] * self.params['freq_s'] * self.params['amp_s']
        
        # ЭНЕРГИЯ СИНХРОНИЗАЦИИ
        if not is_meson:  # БАРИОНЫ
            if particle_name == 'proton':
                coupling = self.params['coupling_proton']
                phases = [0, 0, np.pi/2]  # u, u, d
            elif particle_name == 'neutron':
                coupling = self.params['coupling_neutron']
                phases = [0, np.pi/2, np.pi/2]  # u, d, d
            elif particle_name == 'lambda0':
                coupling = self.params['coupling_lambda0']
                phases = [0, np.pi/2, np.pi]  # u, d, s
            
            # Когерентность для 3 нитей
            phase_coherence_sum = 0
            for i in range(3):
                for j in range(i+1, 3):
                    diff = abs(phases[i] - phases[j])
                    phase_coherence_sum += np.cos(diff)
            phase_coherence = (phase_coherence_sum / 3 + 1) / 2
            
            sync_energy = coupling * phase_coherence
            total = base + sync_energy  # Для барионов: ПЛЮС
            
        else:  # МЕЗОНЫ
            # Определяем coupling
            if has_strange:
                coupling = self.params['coupling_meson_strange']
            else:
                coupling = self.params['coupling_meson_light']
            
            # Определяем фазы
            if is_neutral_meson:
                # КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: для π⁰ фазы [0, 0]
                phases = [0, 0]
                # Для нейтральных мезонов используем специальный coupling
                if 'coupling_meson_neutral' in self.params:
                    coupling = self.params['coupling_meson_neutral']
            elif particle_name in ['pi+', 'pi-']:
                # Для заряженных пионов: [0, π]
                phases = [0, np.pi]
            else:
                # Для каонов: [0, π] как для заряженных пионов
                phases = [0, np.pi]
            
            # Когерентность для 2 нитей
            phase_diff = abs(phases[1] - phases[0])
            
            # Для мезонов: cos(phase_diff + π)
            if is_neutral_meson:
                # Для π⁰: особый случай
                phase_coherence = 0.5  # Среднее значение
            else:
                phase_coherence_value = np.cos(phase_diff + np.pi)
                phase_coherence = (phase_coherence_value + 1) / 2
            
            sync_energy = coupling * phase_coherence
            total = base - sync_energy  # Для мезонов: МИНУС
        
        return total * self.scale

# ============== ПАРАЛЛЕЛЬНЫЙ ОТЖИГ ==============

class ParallelAnnealingOptimizer:
    """Оптимизатор с параллельным отжигом на 6 ядрах"""
    
    def __init__(self):
        self.num_cores = 6
        
        # ИДЕАЛЬНЫЕ ПАРАМЕТРЫ v6.1
        self.base_params = {
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
        
        # НОВЫЕ ПАРАМЕТРЫ (для оптимизации)
        self.base_params.update({
            'base_mass_s': 9.0,  # Начальное значение
            'freq_s': 0.7,
            'amp_s': 0.75,
            'coupling_meson_strange': 8.0,
            'coupling_meson_neutral': 4.5,  # Новый параметр для π⁰
            'coupling_lambda0': 1.2
        })
        
        # ЦЕЛЕВЫЕ ЧАСТИЦЫ
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
        
        # ДИАПАЗОНЫ для оптимизации
        self.ranges = {
            'base_mass_s': (5.0, 15.0),        # s-кварк: 100-300 МэВ эффективно
            'freq_s': (0.5, 1.0),
            'amp_s': (0.5, 1.0),
            'coupling_meson_strange': (6.0, 10.0),
            'coupling_meson_neutral': (3.0, 6.0),  # Для π⁰
            'coupling_lambda0': (0.8, 2.0)
        }
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.result_dir = f"annealing_v80_{timestamp}"
        os.makedirs(self.result_dir, exist_ok=True)
    
    def evaluate_params(self, params):
        """Оценка параметров для всех частиц"""
        model = FixedParticleModel(params)
        results = {}
        
        for name, target in self.targets.items():
            results[f'{name}_mass'] = model.calculate_mass(name, target['composition'])
        
        # Физические параметры кварков
        results['m_u_eff'] = params['base_mass_u'] * params['freq_u'] * params['amp_u'] * 100
        results['m_d_eff'] = params['base_mass_d'] * params['freq_d'] * params['amp_d'] * 100
        results['m_s_eff'] = params['base_mass_s'] * params['freq_s'] * params['amp_s'] * 100
        
        return results
    
    def calculate_error(self, params):
        """Расчет общей ошибки"""
        results = self.evaluate_params(params)
        total_error = 0.0
        
        # ВЕСА ДЛЯ РАЗНЫХ ТИПОВ ЧАСТИЦ
        weights = {
            'proton': 30.0,
            'neutron': 30.0,
            'pi+': 20.0,
            'pi0': 25.0,  # Увеличенный вес для π⁰
            'pi-': 20.0,
            'k+': 15.0,
            'k0': 15.0,
            'k-': 15.0,
            'k0_bar': 15.0,
            'lambda0': 20.0
        }
        
        # Ошибки масс
        for name, target in self.targets.items():
            mass = results[f'{name}_mass']
            target_mass = target['mass']
            rel_error = abs(mass - target_mass) / target_mass
            
            # Квадратичная ошибка с весом
            total_error += weights[name] * (rel_error ** 2)
        
        # ДОПОЛНИТЕЛЬНЫЕ ШТРАФЫ
        
        # 1. Нейтрон должен быть тяжелее протона
        if results['neutron_mass'] < results['proton_mass']:
            total_error += 100.0
        
        # 2. Отношение m_s/m_u должно быть ~25-35
        ratio_s_u = results['m_s_eff'] / results['m_u_eff']
        if ratio_s_u < 20 or ratio_s_u > 40:
            penalty = abs(ratio_s_u - 30) * 10.0
            total_error += penalty
        
        # 3. coupling_meson_strange > coupling_meson_light
        if params['coupling_meson_strange'] <= params['coupling_meson_light']:
            total_error += 50.0
        
        # 4. coupling_meson_neutral около coupling_meson_light
        diff_neutral = abs(params['coupling_meson_neutral'] - params['coupling_meson_light'])
        if diff_neutral > 1.0:
            total_error += diff_neutral * 20.0
        
        return total_error, results
    
    def run_single_annealing(self, seed, iterations=200000):
        """Один поток отжига"""
        np.random.seed(seed)
        
        # Начальные параметры (случайные в пределах диапазонов)
        current_params = self.base_params.copy()
        for param, (min_val, max_val) in self.ranges.items():
            current_params[param] = np.random.uniform(min_val, max_val)
        
        current_error, current_results = self.calculate_error(current_params)
        
        best_params = current_params.copy()
        best_error = current_error
        best_results = current_results
        
        temperature = 5.0
        cooling_rate = 0.99999
        
        for i in range(iterations):
            # Мутация параметров
            new_params = current_params.copy()
            for param in self.ranges.keys():
                if param in current_params:
                    min_val, max_val = self.ranges[param]
                    step = (max_val - min_val) * 0.05
                    mutation = np.random.normal(0, step) * temperature
                    new_val = current_params[param] + mutation
                    
                    # Ограничение диапазона
                    while new_val < min_val or new_val > max_val:
                        if new_val < min_val:
                            new_val = 2 * min_val - new_val
                        if new_val > max_val:
                            new_val = 2 * max_val - new_val
                    
                    new_params[param] = new_val
            
            # Оценка новых параметров
            new_error, new_results = self.calculate_error(new_params)
            
            # Критерий принятия
            delta_error = new_error - current_error
            
            if delta_error < 0:
                # Улучшение
                current_params = new_params
                current_error = new_error
                current_results = new_results
            else:
                # Ухудшение - принимаем с вероятностью
                probability = np.exp(-delta_error / temperature)
                if np.random.random() < probability:
                    current_params = new_params
                    current_error = new_error
                    current_results = new_results
            
            # Обновление лучшего
            if new_error < best_error:
                best_params = new_params.copy()
                best_error = new_error
                best_results = new_results
            
            # Охлаждение
            temperature *= cooling_rate
        
        return {
            'seed': seed,
            'params': best_params,
            'error': best_error,
            'results': best_results,
            'iterations': iterations
        }
    
    def run_parallel_annealing(self, total_iterations=1200000):
        """Параллельный отжиг на всех ядрах"""
        print("="*80)
        print("ПАРАЛЛЕЛЬНЫЙ ОТЖИГ v8.0")
        print(f"Ядер: {self.num_cores}")
        print(f"Итераций на ядро: {total_iterations // self.num_cores:,}")
        print("="*80)
        
        start_time = time.time()
        
        # Разделяем итерации
        iterations_per_core = total_iterations // self.num_cores
        seeds = list(range(1000, 1000 + self.num_cores))
        
        # Запускаем параллельные отжиги
        with mp.Pool(processes=self.num_cores) as pool:
            results = pool.starmap(self.run_single_annealing, 
                                  [(seed, iterations_per_core) for seed in seeds])
        
        # Находим лучший результат
        best_result = min(results, key=lambda x: x['error'])
        
        elapsed = time.time() - start_time
        
        print(f"\n{'='*80}")
        print("ОТЖИГ ЗАВЕРШЕН")
        print(f"Время: {elapsed:.1f} сек")
        print(f"Лучшая ошибка: {best_result['error']:.3f}")
        print(f"Лучший seed: {best_result['seed']}")
        print("="*80)
        
        # Сохраняем и выводим результаты
        self.save_results(results, best_result)
        self.print_detailed_report(best_result)
        
        return best_result['params'], best_result['error'], best_result['results']
    
    def save_results(self, all_results, best_result):
        """Сохранение всех результатов"""
        summary = {
            'model': 'v8.0_parallel_annealing',
            'timestamp': datetime.now().isoformat(),
            'best_result': {
                'seed': best_result['seed'],
                'error': best_result['error'],
                'params': best_result['params'],
                'results': best_result['results']
            },
            'all_results': [
                {
                    'seed': r['seed'],
                    'error': r['error'],
                    'iterations': r['iterations']
                } for r in all_results
            ]
        }
        
        with open(f"{self.result_dir}/annealing_results.json", 'w') as f:
            json.dump(summary, f, indent=2, default=self.json_serializer)
    
    def print_detailed_report(self, best_result):
        """Детальный отчет"""
        params = best_result['params']
        results = best_result['results']
        
        print("\n" + "="*80)
        print("ДЕТАЛЬНЫЙ ОТЧЕТ v8.0")
        print("="*80)
        
        print(f"\nОПТИМИЗИРОВАННЫЕ ПАРАМЕТРЫ:")
        for param, value in params.items():
            if param in self.ranges:
                print(f"  {param}: {value:.6f}")
        
        print(f"\nЭФФЕКТИВНЫЕ МАССЫ КВАРКОВ (МэВ):")
        print(f"  u: {results['m_u_eff']:.2f}")
        print(f"  d: {results['m_d_eff']:.2f}")
        print(f"  s: {results['m_s_eff']:.2f}")
        print(f"  m_s/m_u: {results['m_s_eff']/results['m_u_eff']:.2f}")
        
        print(f"\nМАССЫ ЧАСТИЦ:")
        print(f"{'Частица':<10} {'Масса':<12} {'Цель':<12} {'Ошибка %':<10}")
        print("-" * 80)
        
        total_error_pct = 0
        for name in self.targets.keys():
            mass = results[f'{name}_mass']
            target = self.targets[name]['mass']
            error_pct = abs(mass - target) / target * 100
            total_error_pct += error_pct
            print(f"{name:<10} {mass:<12.3f} {target:<12.3f} {error_pct:<10.3f}")
        
        print(f"\nСредняя ошибка: {total_error_pct/len(self.targets):.3f}%")
        
        # Проверка физических ограничений
        print(f"\nПРОВЕРКА ФИЗИЧЕСКИХ ОГРАНИЧЕНИЙ:")
        print(f"  Нейтрон тяжелее протона: {results['neutron_mass'] > results['proton_mass']} "
              f"({results['neutron_mass'] - results['proton_mass']:.3f} МэВ)")
        print(f"  m_s/m_u в диапазоне 20-40: {20 <= results['m_s_eff']/results['m_u_eff'] <= 40} "
              f"({results['m_s_eff']/results['m_u_eff']:.2f})")
        print(f"  coupling_meson_strange > coupling_meson_light: "
              f"{params['coupling_meson_strange'] > params['coupling_meson_light']} "
              f"({params['coupling_meson_strange']:.3f} > {params['coupling_meson_light']:.3f})")
        
        print(f"\nРезультаты сохранены в: {self.result_dir}")
        print("="*80)
    
    def json_serializer(self, obj):
        """Сериализатор для JSON"""
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return str(obj)

# ============== ГЛАВНАЯ ФУНКЦИЯ ==============

def main():
    """Запуск параллельного отжига"""
    print("="*80)
    print("МОДЕЛЬ v8.0 - ЦЕЛЕВОЙ ОТЖИГ")
    print("Исправления: π⁰ и s-кварк")
    print("="*80)
    
    print("\nПРОБЛЕМЫ, КОТОРЫЕ ИСПРАВЛЯЕМ:")
    print("  1. π⁰ имеет массу 8 МэВ вместо 135 МэВ")
    print("  2. s-кварк слишком тяжелый (2887 МэВ вместо ~100 МэВ)")
    print("  3. Каоны имеют массу 2600 МэВ вместо 494 МэВ")
    
    print("\nСТРАТЕГИЯ:")
    print("  1. Вводим отдельный coupling_meson_neutral для π⁰")
    print("  2. Оптимизируем параметры s-кварка и coupling для странных частиц")
    print("  3. Сохраняем идеальные параметры v6.1 для u,d кварков")
    
    try:
        optimizer = ParallelAnnealingOptimizer()
        best_params, best_error, best_results = optimizer.run_parallel_annealing(
            total_iterations=1200000  # 1.2 млн итераций всего
        )
        
    except Exception as e:
        print(f"\nОШИБКА: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*80)
    print("ВЫЧИСЛЕНИЯ ЗАВЕРШЕНЫ")
    print("="*80)

# ============== ЗАПУСК ==============

if __name__ == "__main__":
    # Устанавливаем метод запуска процессов
    if hasattr(mp, 'set_start_method'):
        try:
            mp.set_start_method('spawn')
        except RuntimeError:
            pass
    
    # Запускаем оптимизацию
    main()