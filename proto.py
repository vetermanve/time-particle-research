"""
МОДЕЛЬ v6.7 - ДВУХМАСШТАБНАЯ МОДЕЛЬ
Отдельные параметры для легких и странных частиц
"""

import numpy as np
import time
import json
from datetime import datetime
import os

class TwoScaleParticleModel:
    def __init__(self, params, particle_name, composition):
        self.params = params
        self.particle_name = particle_name
        self.composition = composition
        self.is_meson = len(composition) == 2
        self.has_strange = any('s' in q for q in composition)
        
        # МАСШТАБЫ: разные для легких и странных частиц
        self.scale_light = 100.0  # Для легких частиц
        self.scale_strange = params.get('scale_strange', 10.0)  # Для странных частиц
        
        # Параметры из v6.1 (фиксированные для легких кварков)
        self.base_mass_u = 2.203806
        self.base_mass_d = 4.583020
        self.base_mass_s = params.get('base_mass_s', 60.0)  # Настраиваемый, но в других единицах
        
        self.freq_u = 0.956359
        self.freq_d = 0.868115
        self.freq_s = params.get('freq_s', 0.2)  # Низкая частота для s-кварка
        
        self.amp_u = 1.032476
        self.amp_d = 0.877773
        self.amp_s = params.get('amp_s', 0.5)
        
        # Силы связи
        self.coupling_proton = 1.613565
        self.coupling_neutron = 0.285395
        self.coupling_lambda0 = params.get('coupling_lambda0', 1.0)
        self.coupling_meson_light = 4.273121
        self.coupling_meson_strange = params.get('coupling_meson_strange', 15.0)  # Может быть большим!
        
        self.phase_shift = 3.173848
    
    def get_quark_params(self, quark):
        """Получить параметры кварка"""
        quark_type = quark.replace('anti_', '')
        
        if quark_type == 'u':
            return self.base_mass_u, self.freq_u, self.amp_u
        elif quark_type == 'd':
            return self.base_mass_d, self.freq_d, self.amp_d
        elif quark_type == 's':
            return self.base_mass_s, self.freq_s, self.amp_s
        else:
            if quark_type == 'u':
                return self.base_mass_u, self.freq_u, self.amp_u
            elif quark_type == 'd':
                return self.base_mass_d, self.freq_d, self.amp_d
            elif quark_type == 's':
                return self.base_mass_s, self.freq_s, self.amp_s
        
        return 1.0, 1.0, 1.0
    
    def calculate_base_mass(self):
        """Расчет базовой массы"""
        total = 0
        for quark in self.composition:
            base_mass, freq, amp = self.get_quark_params(quark)
            total += base_mass * freq * amp
        return total
    
    def calculate_sync_energy(self):
        """Расчет энергии синхронизации"""
        if self.particle_name == 'proton':
            coupling = self.coupling_proton
            phases = [0, 0, np.pi/2]
        elif self.particle_name == 'neutron':
            coupling = self.coupling_neutron
            phases = [0, np.pi/2, np.pi/2]
        elif self.particle_name == 'lambda0':
            coupling = self.coupling_lambda0
            phases = [0, np.pi/2, np.pi]
        elif self.particle_name in ['pi+', 'pi0', 'pi-']:
            coupling = self.coupling_meson_light
            phases = [0, self.phase_shift]
        elif self.particle_name in ['k+', 'k0', 'k-', 'k0_bar']:
            coupling = self.coupling_meson_strange
            phases = [0, self.phase_shift]
        else:
            coupling = self.coupling_meson_light
            phases = [0, self.phase_shift] if self.is_meson else [0, np.pi/2, np.pi/2]
        
        thread_count = len(self.composition)
        
        # Проверка количества фаз
        if len(phases) != thread_count:
            phases = phases[:thread_count]
        
        # Частотная когерентность
        freq_coherence = 1.0
        
        # Фазовая когерентность
        if thread_count >= 2:
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
            phase_coherence = (phase_coherence_sum / max_pairs + 1) / 2
        else:
            phase_coherence = 0.5
        
        # Симметрия
        symmetry = 1.0
        if not self.is_meson:
            if self.particle_name == 'proton':
                symmetry = 1.1
            elif self.particle_name == 'neutron':
                symmetry = 0.95
            elif self.particle_name == 'lambda0':
                symmetry = 0.9
        
        # Ключевое изменение: для странных частиц энергия связи может зависеть от массы
        if self.has_strange and self.is_meson:
            # Для странных мезонов усиливаем энергию связи
            mass_factor = np.sqrt(self.calculate_base_mass())  # Нелинейная зависимость
            sync_energy = coupling * (0.6 * freq_coherence + 0.4 * phase_coherence) * symmetry * mass_factor
        else:
            sync_energy = coupling * (0.6 * freq_coherence + 0.4 * phase_coherence) * symmetry
        
        return sync_energy
    
    def calculate_mass(self):
        """Расчет массы частицы"""
        base = self.calculate_base_mass()
        sync = self.calculate_sync_energy()
        
        if self.is_meson:
            total = base - sync
        else:
            total = base + sync
        
        # Ключевое изменение: разные масштабы
        if self.has_strange:
            return total * self.scale_strange
        else:
            return total * self.scale_light
    
    def calculate_charge(self):
        """Расчет заряда частицы"""
        charges = {
            'u': 2/3, 'd': -1/3, 's': -1/3,
            'anti_u': -2/3, 'anti_d': 1/3, 'anti_s': 1/3
        }
        
        total = 0
        for quark in self.composition:
            total += charges.get(quark, 0)
        
        return round(total, 10)

class FocusedOptimizerV67:
    def __init__(self):
        # Только ключевые частицы для фокусировки
        self.target_particles = {
            'proton': {'mass': 938.272, 'charge': 1.0, 'composition': ['u', 'u', 'd']},
            'neutron': {'mass': 939.565, 'charge': 0.0, 'composition': ['u', 'd', 'd']},
            'pi+': {'mass': 139.570, 'charge': 1.0, 'composition': ['u', 'anti_d']},
            'k+': {'mass': 493.677, 'charge': 1.0, 'composition': ['u', 'anti_s']},
            'lambda0': {'mass': 1115.683, 'charge': 0.0, 'composition': ['u', 'd', 's']},
        }
        
        # Настраиваем ТОЛЬКО параметры для странных частиц
        self.current_params = {
            'base_mass_s': 120.0,  # Большая базовая масса
            'freq_s': 0.15,        # Очень низкая частота
            'amp_s': 0.3,          # Низкая амплитуда
            
            'coupling_lambda0': 1.5,
            'coupling_meson_strange': 20.0,  # Очень большой!
            
            'scale_strange': 10.0  # Другой масштаб
        }
        
        # Широкие диапазоны для странных параметров
        self.param_ranges = {
            'base_mass_s': (50.0, 300.0),
            'freq_s': (0.05, 0.5),
            'amp_s': (0.1, 0.8),
            
            'coupling_lambda0': (0.5, 3.0),
            'coupling_meson_strange': (10.0, 50.0),
            
            'scale_strange': (5.0, 20.0)
        }
        
        self.best_params = None
        self.best_error = float('inf')
        self.best_results = None
        self.history = []
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.result_dir = f"focused_v67_{timestamp}"
        os.makedirs(self.result_dir, exist_ok=True)
    
    def evaluate_params(self, params):
        """Оценка параметров"""
        results = {}
        
        # Создаем модели
        models = {}
        for name, target in self.target_particles.items():
            models[name] = TwoScaleParticleModel(params, name, target['composition'])
        
        # Рассчитываем массы и заряды
        for name, model in models.items():
            results[f'{name}_mass'] = model.calculate_mass()
            results[f'{name}_charge'] = model.calculate_charge()
        
        # Рассчитываем эффективные массы кварков
        # Для u и d используем фиксированные параметры
        m_u_eff = 2.203806 * 0.956359 * 1.032476
        m_d_eff = 4.583020 * 0.868115 * 0.877773
        m_s_eff = params['base_mass_s'] * params['freq_s'] * params['amp_s']
        
        results['m_u_eff_mev'] = m_u_eff * 100
        results['m_d_eff_mev'] = m_d_eff * 100
        results['m_s_eff_mev'] = m_s_eff * params['scale_strange']  # В масштабе странных!
        results['mass_ratio_s_u'] = m_s_eff * params['scale_strange'] / (m_u_eff * 100)
        
        return results
    
    def calculate_total_error(self, params):
        """Функция ошибки с фокусом на странные частицы"""
        results = self.evaluate_params(params)
        total_error = 0.0
        
        # Ошибки с разными весами
        weights = {
            'proton': 100.0,    # Должен быть идеален
            'neutron': 100.0,   # Должен быть идеален
            'pi+': 50.0,        # Должен быть идеален
            'k+': 150.0,        # Критически важен
            'lambda0': 150.0,   # Критически важен
        }
        
        for name, weight in weights.items():
            target = self.target_particles[name]['mass']
            calculated = results[f'{name}_mass']
            
            # Квадратичная ошибка с весом
            error = abs(calculated - target) / target
            total_error += weight * error ** 2
            
            # Штраф за большие отклонения
            if error > 0.05:  # >5%
                total_error += weight * 10.0 * (error - 0.05)
        
        # Ошибки зарядов
        for name, target in self.target_particles.items():
            calculated = results[f'{name}_charge']
            if abs(calculated - target['charge']) > 0.001:
                total_error += 10000.0
        
        # Физические ограничения
        # 1. Нейтрон должен быть тяжелее протона
        n_mass = results['neutron_mass']
        p_mass = results['proton_mass']
        if n_mass <= p_mass:
            total_error += 5000.0 * (p_mass - n_mass + 1.0)
        
        # 2. coupling_neutron < coupling_proton (фиксировано: 0.285 < 1.614)
        
        # 3. Масса Λ⁰ должна быть разумной
        lambda_mass = results['lambda0_mass']
        if lambda_mass < 1000 or lambda_mass > 1300:
            total_error += 1000.0 * abs(lambda_mass - 1115.683) / 1115.683
        
        # 4. Штраф за отрицательные массы
        for name in self.target_particles.keys():
            if results[f'{name}_mass'] <= 0:
                total_error += 100000.0
        
        return total_error, results
    
    def mutate_params(self, params, temperature):
        """Мутация только странных параметров"""
        new_params = params.copy()
        
        for key in params.keys():
            if key in self.param_ranges:
                min_val, max_val = self.param_ranges[key]
                current = params[key]
                
                # Шаг мутации зависит от параметра
                if key == 'coupling_meson_strange':
                    step_factor = 0.1
                elif key == 'base_mass_s':
                    step_factor = 0.05
                else:
                    step_factor = 0.03
                
                range_width = max_val - min_val
                mutation = np.random.normal(0, step_factor * temperature) * range_width
                new_val = current + mutation
                
                # Ограничение диапазоном с отражением
                while new_val < min_val or new_val > max_val:
                    if new_val < min_val:
                        new_val = 2 * min_val - new_val
                    if new_val > max_val:
                        new_val = 2 * max_val - new_val
                
                new_params[key] = new_val
        
        return new_params
    
    def run_optimization(self, iterations=1000000, initial_temp=5.0,
                        cooling_rate=0.999995, save_interval=100000):
        """Запуск оптимизации"""
        print("="*80)
        print("ФОКУСИРОВАННАЯ ОПТИМИЗАЦИЯ v6.7")
        print("Только странные частицы, фиксированные легкие кварки")
        print(f"Итераций: {iterations:,}")
        print("="*80)
        
        current_params = self.current_params.copy()
        current_error, current_results = self.calculate_total_error(current_params)
        
        self.best_params = current_params.copy()
        self.best_error = current_error
        self.best_results = current_results
        
        temperature = initial_temp
        start_time = time.time()
        
        stats = {'accepts': 0, 'improves': 0, 'rejects': 0}
        
        for i in range(iterations):
            # Генерация нового решения
            new_params = self.mutate_params(current_params, temperature)
            
            # Оценка
            new_error, new_results = self.calculate_total_error(new_params)
            
            # Критерий принятия
            delta_error = new_error - current_error
            
            if delta_error < 0:
                current_params = new_params
                current_error = new_error
                current_results = new_results
                stats['accepts'] += 1
                stats['improves'] += 1
            else:
                probability = np.exp(-delta_error / temperature)
                if np.random.random() < probability:
                    current_params = new_params
                    current_error = new_error
                    current_results = new_results
                    stats['accepts'] += 1
                else:
                    stats['rejects'] += 1
            
            # Обновление лучшего
            if new_error < self.best_error:
                self.best_params = new_params.copy()
                self.best_error = new_error
                self.best_results = new_results
                
                self.history.append({
                    'iteration': i,
                    'error': self.best_error,
                    'params': self.best_params.copy(),
                    'results': self.best_results.copy(),
                    'temperature': temperature
                })
            
            # Охлаждение
            temperature *= cooling_rate
            
            # Вывод прогресса
            if i % 50000 == 0:
                elapsed = time.time() - start_time
                progress = (i / iterations) * 100
                
                p_err = abs(self.best_results.get('proton_mass', 0) - 938.272)
                k_err = abs(self.best_results.get('k+_mass', 0) - 493.677)
                l_err = abs(self.best_results.get('lambda0_mass', 0) - 1115.683)
                
                print(f"\rИтерация {i:,}/{iterations:,} ({progress:.1f}%) | "
                      f"Ошибка: {self.best_error:.1f} | "
                      f"Протон: {p_err:.1f} МэВ | "
                      f"K⁺: {k_err:.1f} МэВ | "
                      f"Λ⁰: {l_err:.1f} МэВ | "
                      f"Время: {elapsed:.0f}с", end='', flush=True)
            
            # Сохранение
            if i % save_interval == 0 and i > 0:
                self.save_checkpoint(i)
        
        # Финальные результаты
        elapsed = time.time() - start_time
        print(f"\n\n{'='*80}")
        print("ОПТИМИЗАЦИЯ ЗАВЕРШЕНА")
        print(f"Время: {elapsed:.1f} сек, Итераций: {iterations:,}")
        print(f"Лучшая ошибка: {self.best_error:.3f}")
        print(f"Улучшений: {stats['improves']}")
        
        self.save_final_results()
        self.print_detailed_report()
        
        return self.best_params, self.best_error, self.best_results
    
    def save_checkpoint(self, iteration):
        """Сохранение контрольной точки"""
        checkpoint = {
            'iteration': iteration,
            'error': self.best_error,
            'params': self.best_params,
            'results': self.best_results,
            'timestamp': datetime.now().isoformat()
        }
        
        filename = f"{self.result_dir}/checkpoint_{iteration:08d}.json"
        with open(filename, 'w') as f:
            json.dump(checkpoint, f, indent=2, default=self.json_serializer)
    
    def save_final_results(self):
        """Сохранение результатов"""
        results = {
            'optimization_info': {
                'best_error': self.best_error,
                'iterations': len(self.history),
                'timestamp': datetime.now().isoformat()
            },
            'fixed_parameters': {
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
                'scale_light': 100.0
            },
            'optimized_parameters': self.best_params,
            'results': self.best_results
        }
        
        with open(f"{self.result_dir}/final_results.json", 'w') as f:
            json.dump(results, f, indent=2, default=self.json_serializer)
        
        self.save_text_report()
    
    def save_text_report(self):
        """Сохранение текстового отчета"""
        filename = f"{self.result_dir}/FINAL_REPORT.txt"
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("ФИНАЛЬНЫЙ ОТЧЕТ МОДЕЛИ v6.7\n")
            f.write("Двухмасштабная модель с фиксированными легкими кварками\n")
            f.write("="*80 + "\n\n")
            
            f.write("ФИКСИРОВАННЫЕ ПАРАМЕТРЫ (из v6.1):\n")
            fixed = {
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
                'scale_light': 100.0
            }
            for key, value in fixed.items():
                f.write(f"  {key}: {value:.6f}\n")
            
            f.write("\nОПТИМИЗИРОВАННЫЕ ПАРАМЕТРЫ:\n")
            for key, value in self.best_params.items():
                f.write(f"  {key}: {value:.6f}\n")
            
            f.write("\nРЕЗУЛЬТАТЫ:\n")
            f.write(f"{'Частица':<10} {'Масса (МэВ)':<12} {'Цель (МэВ)':<12} {'Ошибка (%)':<10}\n")
            f.write("-"*80 + "\n")
            
            for name in self.target_particles.keys():
                target = self.target_particles[name]['mass']
                mass = self.best_results.get(f'{name}_mass', 0)
                error_pct = abs(mass - target) / target * 100 if target > 0 else 0
                f.write(f"{name:<10} {mass:<12.3f} {target:<12.3f} {error_pct:<10.3f}\n")
    
    def print_detailed_report(self):
        """Вывод детального отчета"""
        print(f"\n{'='*80}")
        print("ДЕТАЛЬНЫЙ ОТЧЕТ v6.7")
        print("="*80)
        
        print(f"\nФИКСИРОВАННЫЕ ПАРАМЕТРЫ (из v6.1):")
        print(f"  u-кварк: base_mass=2.203806, freq=0.956359, amp=1.032476")
        print(f"  d-кварк: base_mass=4.583020, freq=0.868115, amp=0.877773")
        print(f"  Масштаб для легких частиц: 100.0")
        
        print(f"\nОПТИМИЗИРОВАННЫЕ ПАРАМЕТРЫ ДЛЯ СТРАННЫХ ЧАСТИЦ:")
        for key, value in self.best_params.items():
            print(f"  {key}: {value:.6f}")
        
        print(f"\nРЕЗУЛЬТАТЫ:")
        for name in self.target_particles.keys():
            target = self.target_particles[name]['mass']
            mass = self.best_results.get(f'{name}_mass', 0)
            error_pct = abs(mass - target) / target * 100
            print(f"  {name}: {mass:.3f} МэВ (цель {target:.3f}) - ошибка {error_pct:.3f}%")
        
        print(f"\nЭФФЕКТИВНЫЕ МАССЫ КВАРКОВ:")
        print(f"  u-кварк: {self.best_results.get('m_u_eff_mev', 0):.2f} МэВ (в легких частицах)")
        print(f"  d-кварк: {self.best_results.get('m_d_eff_mev', 0):.2f} МэВ (в легких частицах)")
        print(f"  s-кварк: {self.best_results.get('m_s_eff_mev', 0):.2f} МэВ (в странных частицах)")
        print(f"  Отношение m_s/m_u: {self.best_results.get('mass_ratio_s_u', 0):.3f}")
        
        print(f"\nCoupling ПАРАМЕТРЫ:")
        print(f"  coupling_proton: 1.614 (фиксировано)")
        print(f"  coupling_neutron: 0.285 (фиксировано)")
        print(f"  coupling_lambda0: {self.best_params.get('coupling_lambda0', 0):.3f}")
        print(f"  coupling_meson_light: 4.273 (фиксировано)")
        print(f"  coupling_meson_strange: {self.best_params.get('coupling_meson_strange', 0):.3f}")
        
        print(f"\nПРОВЕРКА ФИЗИЧЕСКИХ ОГРАНИЧЕНИЙ:")
        n_mass = self.best_results.get('neutron_mass', 0)
        p_mass = self.best_results.get('proton_mass', 0)
        print(f"  Нейтрон тяжелее протона: {n_mass > p_mass} (разность: {n_mass-p_mass:.3f} МэВ)")
        
        print(f"\nМОДЕЛЬНЫЕ ПРЕДСКАЗАНИЯ:")
        print(f"  Масштаб для странных частиц: {self.best_params.get('scale_strange', 0):.3f}")
        print(f"  Отношение масштабов strange/light: {self.best_params.get('scale_strange', 0)/100.0:.3f}")
        
        print(f"\nРезультаты сохранены в: {self.result_dir}")
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
    print("МОДЕЛЬ v6.7 - РАДИКАЛЬНОЕ РЕШЕНИЕ")
    print("Фиксированные легкие кварки + отдельные параметры для странных")
    print("="*80)
    
    print("\nФУНДАМЕНТАЛЬНАЯ ПРОБЛЕМА v6.6:")
    print(f"  1. Невозможно одновременно описать легкие и странные частицы")
    print(f"  2. Для каонов нужны огромные энергии связи (>60 единиц)")
    print(f"  3. Текущая формула ограничивает coupling ~5-7")
    
    print("\nРЕШЕНИЕ v6.7:")
    print(f"  1. Фиксируем параметры u и d из v6.1 (идеально работают)")
    print(f"  2. Вводим отдельный масштаб для странных частиц")
    print(f"  3. Разрешаем coupling_meson_strange до 50")
    print(f"  4. Используем scale_strange для преобразования единиц")
    
    print(f"\nМАТЕМАТИЧЕСКАЯ СУТЬ:")
    print(f"  Для легких частиц: M = (база ± синхр) × 100")
    print(f"  Для странных частиц: M = (база ± синхр) × scale_strange")
    print(f"  где scale_strange ~ 10-20 (намного меньше 100)")
    
    print(f"\nФИЗИЧЕСКИЙ СМЫСЛ:")
    print(f"  s-кварк в адронах имеет иную 'эффективную массу'")
    print(f"  Энергии связи для странных частиц могут быть другими")
    print(f"  Разные масштабы = разные единицы измерения для разных секторов")
    
    print(f"\nОЖИДАЕМЫЕ РЕЗУЛЬТАТЫ:")
    print(f"  1. Протон, нейтрон, пион: ошибка < 0.1% (фиксировано)")
    print(f"  2. Каоны: ошибка < 3%")
    print(f"  3. Λ⁰: ошибка < 5%")
    print(f"  4. scale_strange ~ 10-15")
    print(f"  5. coupling_meson_strange ~ 15-30")
    print("="*80)
    
    optimizer = FocusedOptimizerV67()
    
    try:
        best_params, best_error, best_results = optimizer.run_optimization(
            iterations=1000000,
            initial_temp=5.0,
            cooling_rate=0.999995,
            save_interval=100000
        )
        
    except KeyboardInterrupt:
        print("\n\nОптимизация прервана.")
        if optimizer.best_params:
            print(f"\nЛучшие параметры (ошибка: {optimizer.best_error:.1f}):")
            for key, value in optimizer.best_params.items():
                print(f"  {key}: {value:.6f}")
    
    print("\n" + "="*80)
    print("ЭКСПЕРИМЕНТ ЗАВЕРШЕН")
    print("="*80)