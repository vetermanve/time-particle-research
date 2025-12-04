"""
МОДЕЛЬ v6.4 - Исправленная версия с корректной обработкой фаз
"""

import numpy as np
import time
import json
from datetime import datetime
import os

class ParticleModelV64:
    def __init__(self, params, particle_name, composition):
        self.params = params
        self.particle_name = particle_name
        self.composition = composition
        self.is_meson = len(composition) == 2
        
        # Базовые параметры из v6.1
        self.base_mass_u = params.get('base_mass_u', 2.203806)
        self.base_mass_d = params.get('base_mass_d', 4.583020)
        self.base_mass_s = params.get('base_mass_s', 18.0)  # Новая: s-кварк
        
        self.freq_u = params.get('freq_u', 0.956359)
        self.freq_d = params.get('freq_d', 0.868115)
        self.freq_s = params.get('freq_s', 0.85)  # Частота s-кварка
        
        self.amp_u = params.get('amp_u', 1.032476)
        self.amp_d = params.get('amp_d', 0.877773)
        self.amp_s = params.get('amp_s', 0.95)  # Амплитуда s-кварка
        
        # Силы связи
        self.coupling_proton = params.get('coupling_proton', 1.613565)
        self.coupling_neutron = params.get('coupling_neutron', 0.285395)
        self.coupling_lambda0 = params.get('coupling_lambda0', 1.0)  # Для Λ⁰
        self.coupling_meson_light = params.get('coupling_meson_light', 4.273121)  # Для легких мезонов
        self.coupling_meson_strange = params.get('coupling_meson_strange', 4.8)  # Для странных мезонов
        
        self.phase_shift = params.get('phase_shift', 3.173848)
        self.scale = 100.0
        
    def get_quark_params(self, quark):
        """Получить параметры кварка по его типу"""
        quark_type = quark.replace('anti_', '')
        
        if quark_type == 'u':
            return self.base_mass_u, self.freq_u, self.amp_u
        elif quark_type == 'd':
            return self.base_mass_d, self.freq_d, self.amp_d
        elif quark_type == 's':
            return self.base_mass_s, self.freq_s, self.amp_s
        else:
            # Для антикварков используем те же параметры
            if quark_type == 'u':
                return self.base_mass_u, self.freq_u, self.amp_u
            elif quark_type == 'd':
                return self.base_mass_d, self.freq_d, self.amp_d
            elif quark_type == 's':
                return self.base_mass_s, self.freq_s, self.amp_s
        
        return 1.0, 1.0, 1.0  # Запасной вариант
    
    def calculate_base_mass(self):
        """Расчет базовой массы"""
        total = 0
        for quark in self.composition:
            base_mass, freq, amp = self.get_quark_params(quark)
            total += base_mass * freq * amp
        return total
    
    def calculate_sync_energy(self):
        """Расчет энергии синхронизации (ИСПРАВЛЕННЫЙ)"""
        # Определяем фазы в зависимости от частицы
        if self.particle_name == 'proton':
            coupling = self.coupling_proton
            phases = [0, 0, np.pi/2]  # u, u, d
        elif self.particle_name == 'neutron':
            coupling = self.coupling_neutron
            phases = [0, np.pi/2, np.pi/2]  # u, d, d
        elif self.particle_name == 'lambda0':
            coupling = self.coupling_lambda0
            phases = [0, np.pi/2, np.pi]  # u, d, s (нужны 3 фазы!)
        elif self.particle_name in ['pi+', 'pi0', 'pi-']:
            coupling = self.coupling_meson_light
            phases = [0, self.phase_shift]  # 2 фазы для мезонов
        elif self.particle_name in ['k+', 'k0', 'k-', 'k0_bar']:
            coupling = self.coupling_meson_strange
            phases = [0, self.phase_shift]  # 2 фазы для мезонов
        else:
            # Запасной вариант
            coupling = self.coupling_meson_light
            phases = [0, self.phase_shift] if self.is_meson else [0, np.pi/2, np.pi/2]
        
        thread_count = len(self.composition)
        
        # Проверка, что количество фаз совпадает с количеством кварков
        if len(phases) != thread_count:
            raise ValueError(f"Неверное количество фаз для {self.particle_name}: "
                           f"ожидалось {thread_count}, получено {len(phases)}")
        
        # Частотная когерентность
        freq_coherence = 1.0
        
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
        if not self.is_meson:
            if self.particle_name == 'proton':
                symmetry = 1.1
            elif self.particle_name == 'neutron':
                symmetry = 0.95
            elif self.particle_name == 'lambda0':
                symmetry = 0.9  # Меньше симметрии для странного бариона
        
        sync_energy = coupling * (0.6 * freq_coherence + 0.4 * phase_coherence) * symmetry
        return sync_energy
    
    def calculate_mass(self):
        """Расчет массы частицы"""
        try:
            base = self.calculate_base_mass()
            sync = self.calculate_sync_energy()
            
            if self.is_meson:
                total = base - sync
            else:
                total = base + sync
            
            return total * self.scale
        except Exception as e:
            print(f"Ошибка при расчете массы {self.particle_name}: {e}")
            return 0.0
    
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

class ExtendedAnnealingOptimizer:
    def __init__(self):
        # Целевые частицы и их массы (в МэВ)
        self.target_particles = {
            # Уже известные (из v6.1)
            'proton': {'mass': 938.272, 'charge': 1.0, 'composition': ['u', 'u', 'd']},
            'neutron': {'mass': 939.565, 'charge': 0.0, 'composition': ['u', 'd', 'd']},
            'pi+': {'mass': 139.570, 'charge': 1.0, 'composition': ['u', 'anti_d']},
            
            # Новые: нейтральные и отрицательные пионы
            'pi0': {'mass': 134.9768, 'charge': 0.0, 'composition': ['u', 'anti_u']},  # π⁰
            'pi-': {'mass': 139.570, 'charge': -1.0, 'composition': ['d', 'anti_u']},  # π⁻
            
            # Странные мезоны (каоны)
            'k+': {'mass': 493.677, 'charge': 1.0, 'composition': ['u', 'anti_s']},    # K⁺
            'k0': {'mass': 497.611, 'charge': 0.0, 'composition': ['d', 'anti_s']},    # K⁰
            'k-': {'mass': 493.677, 'charge': -1.0, 'composition': ['s', 'anti_u']},   # K⁻
            'k0_bar': {'mass': 497.611, 'charge': 0.0, 'composition': ['s', 'anti_d']},  # K̄⁰
            
            # Лямбда-гиперон (простейший странный барион)
            'lambda0': {'mass': 1115.683, 'charge': 0.0, 'composition': ['u', 'd', 's']},  # Λ⁰
        }
        
        # Начальные параметры из v6.1 + новые для s-кварка
        self.current_params = {
            'base_mass_u': 2.203806,
            'base_mass_d': 4.583020,
            'base_mass_s': 25.0,  # Начальная оценка: в ~11 раз тяжелее u-кварка
            
            'freq_u': 0.956359,
            'freq_d': 0.868115,
            'freq_s': 0.8,  # Частота s-кварка
            
            'amp_u': 1.032476,
            'amp_d': 0.877773,
            'amp_s': 0.85,  # Амплитуда s-кварка
            
            'coupling_proton': 1.613565,
            'coupling_neutron': 0.285395,
            'coupling_lambda0': 1.5,  # Для Λ⁰
            'coupling_meson_light': 4.273121,
            'coupling_meson_strange': 5.0,  # Для странных мезонов
            
            'phase_shift': 3.173848
        }
        
        # Диапазоны для поиска
        self.param_ranges = {
            'base_mass_u': (1.76, 2.65),
            'base_mass_d': (3.67, 5.50),
            'base_mass_s': (15.0, 35.0),  # s-кварк тяжелее
            
            'freq_u': (0.765, 1.148),
            'freq_d': (0.694, 1.042),
            'freq_s': (0.6, 1.0),
            
            'amp_u': (0.826, 1.239),
            'amp_d': (0.702, 1.053),
            'amp_s': (0.6, 1.0),
            
            'coupling_proton': (1.29, 1.936),
            'coupling_neutron': (0.228, 0.342),
            'coupling_lambda0': (1.0, 2.5),  # Для Λ⁰
            'coupling_meson_light': (3.418, 5.128),
            'coupling_meson_strange': (4.0, 6.0),  # Диапазон для странных мезонов
            
            'phase_shift': (2.539, 3.809)
        }
        
        self.best_params = None
        self.best_error = float('inf')
        self.best_results = None
        self.history = []
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.result_dir = f"extended_optimization_v64_{timestamp}"
        os.makedirs(self.result_dir, exist_ok=True)
        
        # Сохраняем конфигурацию
        self.save_configuration()
    
    def save_configuration(self):
        """Сохранение конфигурации"""
        config = {
            'target_particles': self.target_particles,
            'initial_params': self.current_params,
            'param_ranges': self.param_ranges,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(f"{self.result_dir}/config.json", 'w') as f:
            json.dump(config, f, indent=2, default=self.json_serializer)
    
    def evaluate_params(self, params):
        """Оценка параметров для всех частиц"""
        results = {}
        
        # Создаем модели для всех частиц
        models = {}
        for name, target in self.target_particles.items():
            models[name] = ParticleModelV64(params, name, target['composition'])
        
        # Рассчитываем массы и заряды
        for name, model in models.items():
            try:
                results[f'{name}_mass'] = model.calculate_mass()
                results[f'{name}_charge'] = model.calculate_charge()
            except Exception as e:
                print(f"Ошибка при расчете {name}: {e}")
                results[f'{name}_mass'] = 0.0
                results[f'{name}_charge'] = 0.0
        
        # Рассчитываем эффективные массы кварков
        m_u_eff = params['base_mass_u'] * params['freq_u'] * params['amp_u']
        m_d_eff = params['base_mass_d'] * params['freq_d'] * params['amp_d']
        m_s_eff = params['base_mass_s'] * params['freq_s'] * params['amp_s']
        
        results['m_u_eff_mev'] = m_u_eff * 100
        results['m_d_eff_mev'] = m_d_eff * 100
        results['m_s_eff_mev'] = m_s_eff * 100
        results['mass_ratio_d_u'] = m_d_eff / m_u_eff
        results['mass_ratio_s_u'] = m_s_eff / m_u_eff
        results['mass_ratio_s_d'] = m_s_eff / m_d_eff
        
        # Рассчитываем энергии связи для ключевых частиц
        key_particles = ['proton', 'neutron', 'pi+', 'k+', 'lambda0']
        for name in key_particles:
            try:
                results[f'sync_{name}'] = models[name].calculate_sync_energy()
            except:
                results[f'sync_{name}'] = 0.0
        
        return results
    
    def calculate_total_error(self, params):
        """Расчет общей ошибки модели"""
        try:
            results = self.evaluate_params(params)
        except Exception as e:
            print(f"Ошибка при оценке параметров: {e}")
            return 1e10, {}
        
        total_error = 0.0
        
        # Ошибки масс для каждой частицы с разными весами
        mass_weights = {
            'proton': 10.0,
            'neutron': 10.0,
            'pi+': 5.0,
            'pi0': 3.0,
            'pi-': 3.0,
            'k+': 8.0,
            'k0': 8.0,
            'k-': 8.0,
            'k0_bar': 8.0,
            'lambda0': 15.0
        }
        
        for name, target in self.target_particles.items():
            weight = mass_weights.get(name, 2.0)
            calculated_mass = results[f'{name}_mass']
            target_mass = target['mass']
            
            # Комбинированная ошибка
            abs_error = abs(calculated_mass - target_mass)
            rel_error = abs_error / target_mass if target_mass > 0 else 1.0
            
            mass_error = weight * (abs_error + rel_error)
            total_error += mass_error
            
            # Дополнительный штраф за большие отклонения
            if rel_error > 0.05:
                total_error += weight * 50.0 * (rel_error - 0.05)
        
        # Ошибки зарядов (строгие)
        for name, target in self.target_particles.items():
            calculated_charge = results[f'{name}_charge']
            target_charge = target['charge']
            
            if abs(calculated_charge - target_charge) > 0.001:
                total_error += 1000.0
        
        # Физические ограничения и штрафы
        penalties = 0.0
        
        # 1. Проверка отношения масс кварков
        mass_ratio_s_u = results.get('mass_ratio_s_u', 1.0)
        if mass_ratio_s_u < 5.0 or mass_ratio_s_u > 20.0:
            penalties += 500.0 * abs(mass_ratio_s_u - 10.0)
        
        # 2. Проверка coupling_neutron < coupling_proton
        if params['coupling_neutron'] >= params['coupling_proton']:
            penalties += 1000.0
        
        # 3. Штраф за отрицательные массы
        for name in self.target_particles.keys():
            if results[f'{name}_mass'] <= 0:
                penalties += 10000.0
        
        # 4. Проверка масс пионов (π⁰ должен быть легче π⁺)
        pi0_mass = results.get('pi0_mass', 0)
        piplus_mass = results.get('pi+_mass', 0)
        if pi0_mass > piplus_mass and piplus_mass > 0:
            penalties += 300.0 * (pi0_mass - piplus_mass)
        
        # 5. Проверка масс каонов
        k0_mass = results.get('k0_mass', 0)
        kplus_mass = results.get('k+_mass', 0)
        k0_target = self.target_particles['k0']['mass']
        kplus_target = self.target_particles['k+']['mass']
        
        if k0_mass > 0 and kplus_mass > 0:
            expected_diff = k0_target - kplus_target
            actual_diff = k0_mass - kplus_mass
            if abs(actual_diff - expected_diff) > 5.0:
                penalties += 100.0 * abs(actual_diff - expected_diff)
        
        total_error += penalties
        
        return total_error, results
    
    def mutate_params(self, params, temperature):
        """Мутация параметров с адаптивным шагом"""
        new_params = params.copy()
        
        for key in params.keys():
            if key in self.param_ranges:
                min_val, max_val = self.param_ranges[key]
                current = params[key]
                
                # Адаптивный шаг мутации
                range_width = max_val - min_val
                
                # Для разных параметров разная чувствительность
                if 'base_mass' in key:
                    step_factor = 0.05
                elif 'coupling' in key:
                    step_factor = 0.03
                elif 'freq' in key or 'amp' in key:
                    step_factor = 0.02
                else:
                    step_factor = 0.025
                
                step = range_width * step_factor * temperature * np.random.randn()
                
                # Применяем мутацию
                new_val = current + step
                
                # Ограничение диапазоном
                if new_val < min_val:
                    new_val = min_val + abs(new_val - min_val) * 0.1
                if new_val > max_val:
                    new_val = max_val - abs(new_val - max_val) * 0.1
                
                new_params[key] = new_val
        
        return new_params
    
    def run_optimization(self, iterations=2000000, initial_temp=5.0, 
                        cooling_rate=0.999999, save_interval=100000):
        """Запуск оптимизации методом отжига"""
        print("="*80)
        print("РАСШИРЕННАЯ ОПТИМИЗАЦИЯ МОДЕЛИ v6.4 (ИСПРАВЛЕННАЯ)")
        print("Добавлены: π⁰, π⁻, K⁺, K⁰, K⁻, K̄⁰, Λ⁰")
        print(f"Итераций: {iterations:,}")
        print(f"Частиц: {len(self.target_particles)}")
        print("="*80)
        
        current_params = self.current_params.copy()
        current_error, current_results = self.calculate_total_error(current_params)
        
        self.best_params = current_params.copy()
        self.best_error = current_error
        self.best_results = current_results
        
        temperature = initial_temp
        start_time = time.time()
        
        # Статистика
        stats = {
            'accepts': 0,
            'improves': 0,
            'rejects': 0
        }
        
        for i in range(iterations):
            try:
                # Стратегия мутации
                if i < 300000:  # Первые 300к - широкий поиск
                    new_params = self.mutate_params(current_params, temperature * 1.5)
                else:  # Тонкая настройка
                    new_params = self.mutate_params(current_params, temperature)
                
                # Оценка нового решения
                new_error, new_results = self.calculate_total_error(new_params)
                
                # Критерий принятия решения
                delta_error = new_error - current_error
                
                if delta_error < 0:
                    # Лучшее решение
                    current_params = new_params
                    current_error = new_error
                    current_results = new_results
                    stats['accepts'] += 1
                    stats['improves'] += 1
                else:
                    # Принимаем с вероятностью
                    probability = np.exp(-delta_error / temperature)
                    if np.random.random() < probability:
                        current_params = new_params
                        current_error = new_error
                        current_results = new_results
                        stats['accepts'] += 1
                    else:
                        stats['rejects'] += 1
                
                # Обновление лучшего решения
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
                    
                    # Ключевые показатели
                    proton_err = abs(self.best_results.get('proton_mass', 0) - 938.272)
                    neutron_err = abs(self.best_results.get('neutron_mass', 0) - 939.565)
                    kplus_err = abs(self.best_results.get('k+_mass', 0) - 493.677)
                    
                    print(f"\rИтерация {i:,}/{iterations:,} ({progress:.1f}%) | "
                          f"Ошибка: {self.best_error:.2f} | "
                          f"Протон: {proton_err:.1f} МэВ | "
                          f"K⁺: {kplus_err:.1f} МэВ | "
                          f"Темп: {temperature:.3f} | "
                          f"Время: {elapsed:.0f}с", end='', flush=True)
                
                # Сохранение контрольной точки
                if i % save_interval == 0 and i > 0:
                    self.save_checkpoint(i, self.best_params, self.best_error, self.best_results)
                    
            except Exception as e:
                print(f"\nОшибка на итерации {i}: {e}")
                continue
        
        # Финальные результаты
        elapsed = time.time() - start_time
        print(f"\n\n{'='*80}")
        print("ОПТИМИЗАЦИЯ ЗАВЕРШЕНА")
        print(f"Всего итераций: {iterations:,}")
        print(f"Время выполнения: {elapsed:.1f} сек")
        print(f"Лучшая ошибка: {self.best_error:.6f}")
        print(f"Принято решений: {stats['accepts']}")
        print(f"Улучшений: {stats['improves']}")
        
        self.save_final_results()
        self.print_detailed_report()
        
        return self.best_params, self.best_error, self.best_results
    
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
                'iterations': len(self.history),
                'timestamp': datetime.now().isoformat()
            },
            'model_parameters': self.best_params,
            'results': self.best_results
        }
        
        # JSON
        with open(f"{self.result_dir}/final_results.json", 'w') as f:
            json.dump(results, f, indent=2, default=self.json_serializer)
        
        # Текстовый отчет
        self.save_text_report()
    
    def save_text_report(self):
        """Сохранение текстового отчета"""
        filename = f"{self.result_dir}/FINAL_REPORT.txt"
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("ФИНАЛЬНЫЙ ОТЧЕТ МОДЕЛИ v6.4\n")
            f.write("Расширение на странные частицы\n")
            f.write("="*80 + "\n\n")
            
            f.write("ПАРАМЕТРЫ МОДЕЛИ:\n")
            for key, value in self.best_params.items():
                f.write(f"  {key}: {value:.6f}\n")
            
            f.write("\nРЕЗУЛЬТАТЫ ДЛЯ КЛЮЧЕВЫХ ЧАСТИЦ:\n")
            f.write(f"{'Частица':<10} {'Масса (МэВ)':<12} {'Цель (МэВ)':<12} {'Ошибка (%)':<10}\n")
            f.write("-"*80 + "\n")
            
            key_particles = ['proton', 'neutron', 'pi+', 'pi0', 'pi-', 'k+', 'k0', 'lambda0']
            for name in key_particles:
                if name in self.target_particles:
                    target = self.target_particles[name]['mass']
                    mass = self.best_results.get(f'{name}_mass', 0)
                    rel_error = abs(mass - target) / target * 100 if target > 0 else 100
                    
                    f.write(f"{name:<10} {mass:<12.3f} {target:<12.3f} {rel_error:<10.3f}\n")
    
    def print_detailed_report(self):
        """Вывод детального отчета"""
        print(f"\n{'='*80}")
        print("РЕЗУЛЬТАТЫ МОДЕЛИ v6.4")
        print("="*80)
        
        print(f"\nТОЧНОСТЬ ДЛЯ ВСЕХ ЧАСТИЦ:")
        print(f"{'Частица':<10} {'Масса (МэВ)':<12} {'Цель (МэВ)':<12} {'Ошибка (%)':<10}")
        print("-"*80)
        
        for name in self.target_particles.keys():
            target = self.target_particles[name]['mass']
            mass = self.best_results.get(f'{name}_mass', 0)
            error_pct = abs(mass - target) / target * 100 if target > 0 else 100
            print(f"{name:<10} {mass:<12.3f} {target:<12.3f} {error_pct:<10.3f}")
        
        print(f"\nФИЗИЧЕСКИЕ ПАРАМЕТРЫ КВАРКОВ:")
        print(f"  u-кварк: {self.best_results.get('m_u_eff_mev', 0):.2f} МэВ")
        print(f"  d-кварк: {self.best_results.get('m_d_eff_mev', 0):.2f} МэВ")
        print(f"  s-кварк: {self.best_results.get('m_s_eff_mev', 0):.2f} МэВ")
        print(f"  Отношение m_s/m_u: {self.best_results.get('mass_ratio_s_u', 0):.3f}")
        
        print(f"\nCoupling ПАРАМЕТРЫ:")
        for key in ['coupling_proton', 'coupling_neutron', 'coupling_lambda0', 
                   'coupling_meson_light', 'coupling_meson_strange']:
            print(f"  {key}: {self.best_params.get(key, 0):.3f}")
        
        print(f"\nРезультаты сохранены в директории: {self.result_dir}")
        print("="*80)
    
    def json_serializer(self, obj):
        """Сериализатор для JSON"""
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return str(obj)

# ================= ЗАПУСК ОПТИМИЗАЦИИ =================
if __name__ == "__main__":
    print("="*80)
    print("МОДЕЛЬ v6.4 - РАСШИРЕНИЕ НА СТРАННЫЕ ЧАСТИЦЫ")
    print("ИСПРАВЛЕННАЯ ВЕРСИЯ")
    print("="*80)
    
    print("\nЦЕЛЕВЫЕ ЧАСТИЦЫ:")
    for name, info in [
        ('proton', '938.272 МэВ (uud)'),
        ('neutron', '939.565 МэВ (udd)'),
        ('pi+, pi0, pi-', '~139.6 МэВ'),
        ('K⁺, K⁰, K⁻, K̄⁰', '~494-498 МэВ'),
        ('Λ⁰', '1115.683 МэВ (uds)')
    ]:
        print(f"  {name:<20} {info}")
    
    print(f"\nПАРАМЕТРЫ ОПТИМИЗАЦИИ:")
    print(f"  Итераций: 2,000,000")
    print(f"  Начальная температура: 5.0")
    print(f"  Скорость охлаждения: 0.999999")
    
    print(f"\nОЖИДАЕМЫЕ РЕЗУЛЬТАТЫ:")
    print(f"  1. Нуклоны и пионы: ошибка < 0.1%")
    print(f"  2. Каоны: ошибка < 5%")
    print(f"  3. Λ⁰: ошибка < 10%")
    print(f"  4. Отношение m_s/m_u: ~10-15")
    print("="*80)
    
    # Создаем и запускаем оптимизатор
    optimizer = ExtendedAnnealingOptimizer()
    
    try:
        best_params, best_error, best_results = optimizer.run_optimization(
            iterations=2000000,
            initial_temp=5.0,
            cooling_rate=0.999999,
            save_interval=100000
        )
        
    except KeyboardInterrupt:
        print("\n\nОптимизация прервана пользователем.")
        if optimizer.best_params:
            print(f"\nЛучшие найденные параметры:")
            for key, value in optimizer.best_params.items():
                print(f"  {key}: {value:.6f}")
    
    print("\n" + "="*80)
    print("ЭКСПЕРИМЕНТ ЗАВЕРШЕН")
    print("="*80)