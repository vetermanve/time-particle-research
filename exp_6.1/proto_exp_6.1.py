"""
МОДЕЛЬ v6.1 - Полномасштабная оптимизация методом отжига (исправленная)
"""

import numpy as np
import time
import json
from datetime import datetime
import os

class ParticleModelV61:
    def __init__(self, params, particle_name, composition):
        self.params = params
        self.particle_name = particle_name
        self.composition = composition
        self.is_meson = particle_name.startswith('pi')
        
        # Параметры из v5.9 как база
        self.base_mass_u = params.get('base_mass_u', 2.247)
        self.base_mass_d = params.get('base_mass_d', 4.597)
        
        self.freq_u = params.get('freq_u', 0.951)
        self.freq_d = params.get('freq_d', 0.899)
        
        self.amp_u = params.get('amp_u', 1.001)
        self.amp_d = params.get('amp_d', 0.849)
        
        self.coupling_proton = params.get('coupling_proton', 1.676)
        self.coupling_neutron = params.get('coupling_neutron', 0.291)
        self.coupling_meson = params.get('coupling_meson', 4.251)
        
        self.phase_shift = params.get('phase_shift', 3.163802)
        self.scale = 100.0
        
    def calculate_base_mass(self):
        total = 0
        for quark in self.composition:
            base_type = quark.replace('anti_', '')
            if base_type == 'u':
                total += self.base_mass_u * self.freq_u * self.amp_u
            else:  # 'd' или 'anti_d'
                total += self.base_mass_d * self.freq_d * self.amp_d
        return total
    
    def calculate_sync_energy(self):
        if self.particle_name == 'proton':
            coupling = self.coupling_proton
            phases = [0, 0, np.pi/2]
        elif self.particle_name == 'neutron':
            coupling = self.coupling_neutron
            phases = [0, np.pi/2, np.pi/2]
        else:  # pi+
            coupling = self.coupling_meson
            phases = [0, self.phase_shift]
        
        thread_count = len(self.composition)
        
        # Частотная когерентность
        freq_coherence = 1.0  # Упрощаем для стабильности
        
        # Фазовая когерентность
        phase_coherence_sum = 0
        for i in range(thread_count):
            for j in range(i+1, thread_count):
                diff = abs(phases[i] - phases[j]) % (2*np.pi)
                diff = min(diff, 2*np.pi - diff)
                
                if self.is_meson:
                    phase_coherence_sum += np.cos(diff + np.pi)
                else:
                    phase_coherence_sum += np.cos(diff)
        
        max_pairs = thread_count * (thread_count - 1) / 2
        phase_coherence = (phase_coherence_sum / max_pairs + 1) / 2 if max_pairs > 0 else 0.5
        
        # Симметрия
        symmetry = 1.0
        if self.particle_name == 'proton':
            symmetry = 1.1
        elif self.particle_name == 'neutron':
            symmetry = 0.95
        
        sync_energy = coupling * (0.6 * freq_coherence + 0.4 * phase_coherence) * symmetry
        return sync_energy
    
    def calculate_mass(self):
        base = self.calculate_base_mass()
        sync = self.calculate_sync_energy()
        
        if self.is_meson:
            total = base - sync
        else:
            total = base + sync
        
        return total * self.scale
    
    def calculate_charge(self):
        charges = {'u': 2/3, 'd': -1/3, 'anti_d': 1/3}
        total = 0
        for quark in self.composition:
            total += charges.get(quark, 0)
        return round(total, 10)

class AnnealingOptimizerV61:
    def __init__(self):
        self.target_particles = {
            'proton': {'mass': 938.272, 'charge': 1.0, 'composition': ['u', 'u', 'd']},
            'neutron': {'mass': 939.565, 'charge': 0.0, 'composition': ['u', 'd', 'd']},
            'pi+': {'mass': 139.570, 'charge': 1.0, 'composition': ['u', 'anti_d']}
        }
        
        # Начальные параметры из v5.9
        self.current_params = {
            'base_mass_u': 2.247,
            'base_mass_d': 4.597,
            'freq_u': 0.951,
            'freq_d': 0.899,
            'amp_u': 1.001,
            'amp_d': 0.849,
            'coupling_proton': 1.676,
            'coupling_neutron': 0.291,
            'coupling_meson': 4.251,
            'phase_shift': 3.163802
        }
        
        # Диапазоны для поиска (±5% от v5.9)
        self.param_ranges = {
            'base_mass_u': (2.135, 2.359),
            'base_mass_d': (4.367, 4.827),
            'freq_u': (0.903, 0.999),
            'freq_d': (0.854, 0.944),
            'amp_u': (0.951, 1.051),
            'amp_d': (0.806, 0.891),
            'coupling_proton': (1.592, 1.760),
            'coupling_neutron': (0.276, 0.306),
            'coupling_meson': (4.038, 4.463),
            'phase_shift': (3.006, 3.322)
        }
        
        self.best_params = None
        self.best_error = float('inf')
        self.best_details = None
        self.history = []
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.result_dir = f"annealing_v61_{timestamp}"
        os.makedirs(self.result_dir, exist_ok=True)
    
    def evaluate_params(self, params):
        """Оценка параметров с расчетом масс и ошибок"""
        models = {}
        results = {}
        
        # Создаем модели для всех частиц
        for name, target in self.target_particles.items():
            models[name] = ParticleModelV61(params, name, target['composition'])
        
        # Рассчитываем массы и заряды
        for name, model in models.items():
            results[f'{name}_mass'] = model.calculate_mass()
            results[f'{name}_charge'] = model.calculate_charge()
        
        # Рассчитываем разность масс
        results['mass_diff'] = results['neutron_mass'] - results['proton_mass']
        
        # Рассчитываем эффективные массы кварков
        m_u_eff = params['base_mass_u'] * params['freq_u'] * params['amp_u']
        m_d_eff = params['base_mass_d'] * params['freq_d'] * params['amp_d']
        results['m_u_eff_mev'] = m_u_eff * 100
        results['m_d_eff_mev'] = m_d_eff * 100
        results['mass_ratio'] = m_d_eff / m_u_eff
        
        # Рассчитываем энергии связи
        results['sync_proton'] = models['proton'].calculate_sync_energy()
        results['sync_neutron'] = models['neutron'].calculate_sync_energy()
        results['sync_pion'] = models['pi+'].calculate_sync_energy()
        
        return results
    
    def calculate_total_error(self, params):
        """Расчет общей ошибки модели"""
        results = self.evaluate_params(params)
        
        # Ошибки масс
        mass_errors = []
        for name in ['proton', 'neutron', 'pi+']:
            target_mass = self.target_particles[name]['mass']
            calculated_mass = results[f'{name}_mass']
            # Абсолютная ошибка в МэВ (важнее относительной)
            abs_error = abs(calculated_mass - target_mass)
            # Комбинированная ошибка: абсолютная + относительная
            error = abs_error + abs_error / target_mass
            mass_errors.append(error)
        
        # Ошибка заряда (строгая)
        charge_errors = []
        for name in ['proton', 'neutron', 'pi+']:
            target_charge = self.target_particles[name]['charge']
            calculated_charge = results[f'{name}_charge']
            if abs(calculated_charge - target_charge) > 0.001:
                charge_errors.append(1000.0)  # Большой штраф
            else:
                charge_errors.append(0.0)
        
        # Ошибка разности масс (критически важна!)
        target_diff = 1.293
        diff_error = abs(results['mass_diff'] - target_diff)
        # Огромный вес разности масс
        diff_error_weighted = diff_error * 1000.0
        
        # Физические штрафы
        penalties = 0.0
        
        # Штраф за нефизичное отношение масс кварков
        mass_ratio = results['mass_ratio']
        if mass_ratio < 1.5 or mass_ratio > 2.0:
            penalties += abs(mass_ratio - 1.75) * 100.0
        
        # Штраф за отрицательную массу пиона
        if results['pi+_mass'] <= 0:
            penalties += 10000.0
        
        # Штраф за coupling_neutron > coupling_proton
        if params['coupling_neutron'] > params['coupling_proton']:
            penalties += 1000.0
        
        # Общая ошибка
        total_error = (sum(mass_errors) + sum(charge_errors) + 
                      diff_error_weighted + penalties)
        
        return total_error, results
    
    def mutate_params(self, params, temperature):
        """Мутация параметров с учетом температуры"""
        new_params = params.copy()
        mutation_strength = 0.01 * temperature
        
        for key in params.keys():
            if key in self.param_ranges:
                min_val, max_val = self.param_ranges[key]
                current = params[key]
                
                # Адаптивный шаг мутации
                range_width = max_val - min_val
                step = range_width * mutation_strength * np.random.randn()
                
                new_val = current + step
                
                # Отражение от границ
                while new_val < min_val or new_val > max_val:
                    if new_val < min_val:
                        new_val = 2 * min_val - new_val
                    if new_val > max_val:
                        new_val = 2 * max_val - new_val
                
                new_params[key] = new_val
        
        return new_params
    
    def run_annealing(self, iterations=3000000, initial_temp=5.0, 
                     cooling_rate=0.999997, save_interval=100000):
        """Запуск алгоритма отжига"""
        print("="*80)
        print("ЗАПУСК ОПТИМИЗАЦИИ v6.1")
        print(f"Итераций: {iterations:,}")
        print(f"Начальная температура: {initial_temp}")
        print(f"Скорость охлаждения: {cooling_rate}")
        print("="*80)
        
        current_params = self.current_params.copy()
        current_error, current_results = self.calculate_total_error(current_params)
        
        best_params = current_params.copy()
        best_error = current_error
        best_results = current_results
        
        temperature = initial_temp
        start_time = time.time()
        
        # Статистика
        stats = {
            'accepts': 0,
            'improves': 0,
            'rejects': 0
        }
        
        for i in range(iterations):
            # Генерация нового решения
            if i < 100000:  # Первые 100к итераций - более широкий поиск
                new_params = self.mutate_params(current_params, temperature * 2.0)
            else:
                new_params = self.mutate_params(current_params, temperature)
            
            # Оценка нового решения
            new_error, new_results = self.calculate_total_error(new_params)
            
            # Критерий принятия решения (Метрополис)
            delta_error = new_error - current_error
            if delta_error < 0:
                # Лучшее решение - всегда принимаем
                current_params = new_params
                current_error = new_error
                current_results = new_results
                stats['accepts'] += 1
                stats['improves'] += 1
            else:
                # Худшее решение - принимаем с вероятностью
                probability = np.exp(-delta_error / temperature)
                if np.random.random() < probability:
                    current_params = new_params
                    current_error = new_error
                    current_results = new_results
                    stats['accepts'] += 1
                else:
                    stats['rejects'] += 1
            
            # Обновление лучшего решения
            if new_error < best_error:
                best_params = new_params.copy()
                best_error = new_error
                best_results = new_results
                
                # Сохраняем в историю
                self.history.append({
                    'iteration': i,
                    'error': best_error,
                    'params': best_params.copy(),
                    'results': best_results.copy(),
                    'temperature': temperature
                })
            
            # Охлаждение
            temperature *= cooling_rate
            
            # Вывод прогресса и сохранение
            if i % 10000 == 0:
                elapsed = time.time() - start_time
                progress = (i / iterations) * 100
                
                print(f"\rИтерация {i:,}/{iterations:,} ({progress:.1f}%) | "
                      f"Ошибка: {best_error:.4f} | "
                      f"Темп: {temperature:.4f} | "
                      f"Протон: {best_results['proton_mass']:.1f} | "
                      f"Нейтрон: {best_results['neutron_mass']:.1f} | "
                      f"Пион: {best_results['pi+_mass']:.1f} | "
                      f"Разность: {best_results['mass_diff']:.3f} | "
                      f"Время: {elapsed:.0f}с", end='', flush=True)
            
            if i % save_interval == 0 and i > 0:
                self.save_checkpoint(i, best_params, best_error, best_results)
        
        # Финальные результаты
        elapsed = time.time() - start_time
        print(f"\n\n{'='*80}")
        print("ОПТИМИЗАЦИЯ ЗАВЕРШЕНА")
        print(f"Всего итераций: {iterations:,}")
        print(f"Время выполнения: {elapsed:.1f} сек")
        print(f"Лучшая ошибка: {best_error:.6f}")
        print(f"Принято решений: {stats['accepts']}")
        print(f"Улучшений: {stats['improves']}")
        print(f"Отклонено: {stats['rejects']}")
        
        self.best_params = best_params
        self.best_error = best_error
        self.best_details = best_results
        
        self.save_final_results()
        self.print_summary()
        
        return best_params, best_error, best_results
    
    def save_checkpoint(self, iteration, params, error, results):
        """Сохранение контрольной точки"""
        checkpoint = {
            'iteration': iteration,
            'error': error,
            'params': params,
            'results': results,
            'timestamp': datetime.now().isoformat()
        }
        
        filename = f"{self.result_dir}/checkpoint_{iteration:08d}.json"
        with open(filename, 'w') as f:
            json.dump(checkpoint, f, indent=2, default=self.json_serializer)
    
    def save_final_results(self):
        """Сохранение финальных результатов"""
        results = {
            'optimization_info': {
                'best_error': self.best_error,
                'timestamp': datetime.now().isoformat(),
                'history_size': len(self.history)
            },
            'model_parameters': self.best_params,
            'results': self.best_details
        }
        
        # JSON
        with open(f"{self.result_dir}/final_results.json", 'w') as f:
            json.dump(results, f, indent=2, default=self.json_serializer)
        
        # Текстовый файл
        self.save_text_report()
    
    def save_text_report(self):
        """Сохранение отчета в текстовом формате"""
        filename = f"{self.result_dir}/FINAL_REPORT.txt"
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("ФИНАЛЬНЫЙ ОТЧЕТ МОДЕЛИ v6.1\n")
            f.write("="*80 + "\n\n")
            
            f.write("ОПТИМИЗАЦИОННАЯ ИНФОРМАЦИЯ:\n")
            f.write(f"  Лучшая ошибка: {self.best_error:.8f}\n")
            f.write(f"  Улучшений найдено: {len(self.history)}\n\n")
            
            f.write("ПАРАМЕТРЫ МОДЕЛИ:\n")
            for key, value in self.best_params.items():
                f.write(f"  {key}: {value:.6f}\n")
            
            f.write("\nРЕЗУЛЬТАТЫ РАСЧЕТА:\n")
            f.write(f"{'Частица':<10} {'Масса (МэВ)':<15} {'Цель (МэВ)':<15} {'Ошибка (МэВ)':<15} {'Ошибка (%)':<10}\n")
            f.write("-"*80 + "\n")
            
            for name in ['proton', 'neutron', 'pi+']:
                target = self.target_particles[name]['mass']
                calculated = self.best_details[f'{name}_mass']
                abs_error = abs(calculated - target)
                rel_error = (abs_error / target) * 100
                f.write(f"{name:<10} {calculated:<15.3f} {target:<15.3f} "
                       f"{abs_error:<15.3f} {rel_error:<10.6f}\n")
            
            f.write(f"\nРАЗНОСТЬ МАСС НЕЙТРОН-ПРОТОН:\n")
            calculated_diff = self.best_details['mass_diff']
            target_diff = 1.293
            diff_error = abs(calculated_diff - target_diff)
            f.write(f"  Расчетная: {calculated_diff:.6f} МэВ\n")
            f.write(f"  Целевая: {target_diff:.6f} МэВ\n")
            f.write(f"  Ошибка: {diff_error:.6f} МэВ\n")
            f.write(f"  Относительная ошибка: {(diff_error/target_diff)*100:.6f}%\n")
            
            f.write(f"\nЭФФЕКТИВНЫЕ МАССЫ КВАРКОВ:\n")
            f.write(f"  u-кварк: {self.best_details['m_u_eff_mev']:.2f} МэВ\n")
            f.write(f"  d-кварк: {self.best_details['m_d_eff_mev']:.2f} МэВ\n")
            f.write(f"  Отношение m_d/m_u: {self.best_details['mass_ratio']:.3f}\n")
            
            f.write(f"\nЭНЕРГИИ СВЯЗИ (в единицах модели):\n")
            f.write(f"  Протон: {self.best_details['sync_proton']:.3f}\n")
            f.write(f"  Нейтрон: {self.best_details['sync_neutron']:.3f}\n")
            f.write(f"  Пион: {self.best_details['sync_pion']:.3f}\n")
            
            f.write(f"\nЗАРЯДЫ:\n")
            for name in ['proton', 'neutron', 'pi+']:
                charge = self.best_details[f'{name}_charge']
                target = self.target_particles[name]['charge']
                status = "✓" if abs(charge - target) < 0.001 else "✗"
                f.write(f"  {name}: {charge:.6f} (цель {target:.3f}) {status}\n")
            
            f.write("\n" + "="*80 + "\n")
    
    def print_summary(self):
        """Вывод сводки результатов"""
        print(f"\n{'='*80}")
        print("СВОДКА РЕЗУЛЬТАТОВ")
        print("="*80)
        
        print(f"\nПАРАМЕТРЫ МОДЕЛИ:")
        for key, value in self.best_params.items():
            print(f"  {key}: {value:.6f}")
        
        print(f"\nТОЧНОСТЬ МОДЕЛИ:")
        print(f"{'Частица':<10} {'Масса (МэВ)':<15} {'Цель (МэВ)':<15} {'Ошибка (%)':<12}")
        print("-"*80)
        
        for name in ['proton', 'neutron', 'pi+']:
            target = self.target_particles[name]['mass']
            calculated = self.best_details[f'{name}_mass']
            error_pct = abs(calculated - target) / target * 100
            print(f"{name:<10} {calculated:<15.3f} {target:<15.3f} {error_pct:<12.6f}")
        
        print(f"\nРАЗНОСТЬ МАСС НЕЙТРОН-ПРОТОН:")
        calculated_diff = self.best_details['mass_diff']
        target_diff = 1.293
        diff_error = abs(calculated_diff - target_diff)
        print(f"  Расчетная: {calculated_diff:.6f} МэВ")
        print(f"  Целевая: {target_diff:.6f} МэВ")
        print(f"  Ошибка: {diff_error:.6f} МэВ")
        print(f"  Относительная ошибка: {(diff_error/target_diff)*100:.6f}%")
        
        print(f"\nФИЗИЧЕСКИЕ ПАРАМЕТРЫ:")
        print(f"  Эффективная масса u-кварка: {self.best_details['m_u_eff_mev']:.2f} МэВ")
        print(f"  Эффективная масса d-кварка: {self.best_details['m_d_eff_mev']:.2f} МэВ")
        print(f"  Отношение m_d/m_u: {self.best_details['mass_ratio']:.3f}")
        
        print(f"\nCoupling параметры:")
        print(f"  coupling_proton: {self.best_params['coupling_proton']:.3f}")
        print(f"  coupling_neutron: {self.best_params['coupling_neutron']:.3f}")
        print(f"  coupling_meson: {self.best_params['coupling_meson']:.3f}")
        print(f"  Отношение neutron/proton: {self.best_params['coupling_neutron']/self.best_params['coupling_proton']:.3f}")
        
        print(f"\nРезультаты сохранены в директории: {self.result_dir}")
        print("="*80)
    
    def json_serializer(self, obj):
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return str(obj)

# ================= ЗАПУСК =================
if __name__ == "__main__":
    print("="*80)
    print("МОДЕЛЬ СИНХРОНИЗАЦИИ НИТЕЙ v6.1")
    print("ПОЛНОМАСШТАБНАЯ ОПТИМИЗАЦИЯ (3,000,000 ИТЕРАЦИЙ)")
    print("="*80)
    
    optimizer = AnnealingOptimizerV61()
    
    # Запускаем оптимизацию
    best_params, best_error, best_results = optimizer.run_annealing(
        iterations=3000000,      # 3 миллиона итераций
        initial_temp=5.0,        # Начальная температура
        cooling_rate=0.999997,   # Скорость охлаждения
        save_interval=200000     # Сохраняем каждые 200к итераций
    )