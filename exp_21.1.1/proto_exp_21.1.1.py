"""
МОДЕЛЬ v21.1.1 — Иерархическая модель синхронизации нитей времени
Ядро: v6.1 + электромагнитные поправки + суперпозиция π⁰
"""

import numpy as np
import json
from datetime import datetime
import os

class HierarchicalThreadModel:
    """Иерархическая модель синхронизации нитей времени"""
    
    def __init__(self, params=None):
        # ПАРАМЕТРЫ УРОВНЯ 1: СИНХРОНИЗАЦИЯ НИТЕЙ (v6.1)
        if params is None:
            self.params = {
                # Базовые эффективные массы кварков в адронах (МэВ)
                'm_u_eff': 217.61,    # Эффективная масса u-кварка в адронах
                'm_d_eff': 349.23,    # Эффективная масса d-кварка
                
                # Энергии синхронизации (МэВ)
                'E_sync_proton': 153.822,    # Энергия синхронизации протона
                'E_sync_neutron': 23.495,    # Энергия синхронизации нейтрона
                'E_sync_meson_charged': 427.27,  # Энергия синхронизации заряженных мезонов
                'E_sync_meson_neutral': 431.86,  # Энергия синхронизации нейтральных мезонов (π⁰)
                
                # ЭЛЕКТРОМАГНИТНЫЕ ПОПРАВКИ (УРОВЕНЬ 2)
                'EM_correction_charged': 4.593,  # Электромагнитная поправка для заряженных пионов (МэВ)
                'delta_iso': 0.01,               # Нарушение изоспина (МэВ)
                
                # ПАРАМЕТРЫ СИНХРОНИЗАЦИИ
                'phase_proton': [0.0, 0.0, np.pi/2],      # Фазы кварков протона
                'phase_neutron': [0.0, np.pi/2, np.pi/2], # Фазы кварков нейтрона
                'phase_meson_charged': [0.0, np.pi],      # Фазы кварков заряженных мезонов
                'phase_meson_neutral': [0.0, np.pi/2],    # Фазы кварков нейтральных мезонов
            }
        else:
            self.params = params
        
        # ЦЕЛЕВЫЕ МАССЫ (МэВ) - 5 частиц для начала
        self.targets = {
            'proton': {'mass': 938.272, 'charge': 1, 'composition': ['u', 'u', 'd']},
            'neutron': {'mass': 939.565, 'charge': 0, 'composition': ['u', 'd', 'd']},
            'pi+': {'mass': 139.570, 'charge': 1, 'composition': ['u', 'anti_d']},
            'pi0': {'mass': 134.9768, 'charge': 0, 'composition': ['u', 'anti_u', 'd', 'anti_d']},  # Суперпозиция!
            'pi-': {'mass': 139.570, 'charge': -1, 'composition': ['d', 'anti_u']},
        }
    
    def calculate_base_mass(self, composition):
        """Базовая масса из эффективных масс кварков"""
        mass = 0.0
        for quark in composition:
            base_type = quark.replace('anti_', '')
            if base_type == 'u':
                mass += self.params['m_u_eff']
            elif base_type == 'd':
                mass += self.params['m_d_eff']
        return mass
    
    def calculate_sync_energy(self, particle_name, composition):
        """Энергия синхронизации в зависимости от типа частицы"""
        if particle_name == 'proton':
            return self.params['E_sync_proton']
        elif particle_name == 'neutron':
            return self.params['E_sync_neutron']
        elif particle_name in ['pi+', 'pi-']:
            # Заряженные мезоны
            return self.params['E_sync_meson_charged']
        elif particle_name == 'pi0':
            # Нейтральный пион - особая суперпозиция
            return self.params['E_sync_meson_neutral']
        else:
            # По умолчанию для мезонов
            return self.params['E_sync_meson_charged']
    
    def electromagnetic_correction(self, particle_name, charge):
        """Электромагнитная поправка"""
        if particle_name in ['pi+', 'pi-']:
            return self.params['EM_correction_charged']
        elif particle_name == 'pi0':
            # Нейтральный пион: маленькая поправка из-за разницы масс u/d
            return -self.params['delta_iso']
        else:
            return 0.0
    
    def calculate_pi0_mass(self):
        """Расчет массы π⁰ через суперпозицию u-анти-u и d-анти-d"""
        # Масса u-анти-u состояния
        M_uu = 2 * self.params['m_u_eff'] - self.params['E_sync_meson_charged']
        
        # Масса d-анти-d состояния
        M_dd = 2 * self.params['m_d_eff'] - self.params['E_sync_meson_charged']
        
        # π⁰ - это суперпозиция (u-анти-u + d-анти-d)/√2
        # В приближении идеальной суперпозиции:
        M_pi0_superposition = np.sqrt((M_uu**2 + M_dd**2) / 2)
        
        # Добавляем нарушение изоспина из-за разницы масс u/d
        M_pi0 = M_pi0_superposition - self.params['delta_iso']
        
        return M_pi0
    
    def calculate_mass(self, particle_name, composition=None, charge=None):
        """Расчет массы частицы с учётом всех поправок"""
        if composition is None:
            composition = self.targets[particle_name]['composition']
        
        if charge is None:
            charge = self.targets[particle_name]['charge']
        
        # БАЗОВАЯ ФОРМУЛА: M = базовые массы + знак * синхронизация + поправки
        
        base_mass = self.calculate_base_mass(composition)
        sync_energy = self.calculate_sync_energy(particle_name, composition)
        
        # Определяем знак для энергии синхронизации
        if len(composition) == 3:  # Барионы
            mass = base_mass + sync_energy
        else:  # Мезоны
            # Для мезонов энергия синхронизации УМЕНЬШАЕТ массу
            mass = base_mass - sync_energy
        
        # Электромагнитные поправки
        em_correction = self.electromagnetic_correction(particle_name, charge)
        mass += em_correction
        
        # Особый расчёт для π⁰
        if particle_name == 'pi0':
            mass = self.calculate_pi0_mass()
        
        return mass
    
    def calculate_charge(self, composition):
        """Расчёт электрического заряда"""
        charges = {'u': 2/3, 'd': -1/3, 'anti_u': -2/3, 'anti_d': 1/3}
        total = sum(charges[q] for q in composition)
        return round(total, 10)

class HierarchicalOptimizer:
    """Оптимизатор для иерархической модели"""
    
    def __init__(self):
        self.model = HierarchicalThreadModel()
        self.best_params = None
        self.best_error = float('inf')
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.result_dir = f"v21_1_1_results_{timestamp}"
        os.makedirs(self.result_dir, exist_ok=True)
    
    def evaluate_error(self, params):
        """Оценка ошибки модели"""
        self.model.params.update(params)
        
        errors = []
        details = {}
        
        for name, target in self.model.targets.items():
            calculated_mass = self.model.calculate_mass(name)
            target_mass = target['mass']
            
            # Абсолютная ошибка в МэВ
            abs_error = abs(calculated_mass - target_mass)
            
            # Относительная ошибка
            rel_error = abs_error / target_mass * 100
            
            # Сохраняем детали
            details[name] = {
                'calculated': calculated_mass,
                'target': target_mass,
                'abs_error': abs_error,
                'rel_error': rel_error,
                'charge': self.model.calculate_charge(target['composition'])
            }
            
            # Комбинированная ошибка (больше вес у абсолютной ошибки)
            error = abs_error + rel_error
            errors.append(error)
        
        # Специальная проверка: разность масс n-p
        mass_diff = details['neutron']['calculated'] - details['proton']['calculated']
        target_diff = 1.293
        diff_error = abs(mass_diff - target_diff) * 1000  # Большой вес!
        
        # Физические штрафы
        penalties = 0
        
        # Штраф за отрицательные массы
        if any(d['calculated'] <= 0 for d in details.values()):
            penalties += 10000
        
        # Штраф за неправильные заряды
        for name, detail in details.items():
            target_charge = self.model.targets[name]['charge']
            if abs(detail['charge'] - target_charge) > 0.001:
                penalties += 1000
        
        # Штраф за нефизичные отношения масс кварков
        ratio = self.model.params['m_d_eff'] / self.model.params['m_u_eff']
        if ratio < 1.3 or ratio > 2.2:
            penalties += abs(ratio - 1.6) * 100
        
        total_error = sum(errors) + diff_error + penalties
        return total_error, details, mass_diff
    
    def run_genetic_optimization(self, generations=1000, population_size=50):
        """Генетический алгоритм оптимизации"""
        print("\n" + "="*80)
        print("v21.1.1: ГЕНЕТИЧЕСКАЯ ОПТИМИЗАЦИЯ ИЕРАРХИЧЕСКОЙ МОДЕЛИ")
        print("="*80)
        
        param_names = list(self.model.params.keys())
        
        # Диапазоны параметров (±30% от начальных значений)
        ranges = {}
        for name, value in self.model.params.items():
            if 'mass' in name or 'E_sync' in name:
                ranges[name] = (value * 0.7, value * 1.3)
            elif 'phase' in name:
                ranges[name] = (0, 2*np.pi)
            else:
                ranges[name] = (value * 0.5, value * 2.0)
        
        # Начальная популяция
        population = []
        for _ in range(population_size):
            individual = {}
            for name in param_names:
                min_val, max_val = ranges[name]
                individual[name] = np.random.uniform(min_val, max_val)
            population.append(individual)
        
        best_error = float('inf')
        best_individual = None
        
        for gen in range(generations):
            # Оценка популяции
            scores = []
            details_list = []
            for individual in population:
                error, details, _ = self.evaluate_error(individual)
                scores.append(error)
                details_list.append(details)
            
            # Сортировка по качеству
            sorted_indices = np.argsort(scores)
            
            # Лучший индивид
            current_best_error = scores[sorted_indices[0]]
            if current_best_error < best_error:
                best_error = current_best_error
                best_individual = population[sorted_indices[0]].copy()
                best_details = details_list[sorted_indices[0]]
            
            # Отбор лучших (элитизм)
            elite_size = max(2, population_size // 10)
            elite = [population[i] for i in sorted_indices[:elite_size]]
            
            # Скрещивание и мутация
            new_population = elite.copy()
            
            while len(new_population) < population_size:
                # Выбор родителей
                parent1 = elite[np.random.randint(0, len(elite))]
                parent2 = elite[np.random.randint(0, len(elite))]
                
                # Скрещивание
                child = {}
                for name in param_names:
                    if np.random.random() < 0.5:
                        child[name] = parent1[name]
                    else:
                        child[name] = parent2[name]
                
                # Мутация
                if np.random.random() < 0.3:
                    mut_param = np.random.choice(param_names)
                    min_val, max_val = ranges[mut_param]
                    child[mut_param] = np.random.uniform(min_val, max_val)
                
                new_population.append(child)
            
            population = new_population[:population_size]
            
            # Вывод прогресса
            if gen % 100 == 0:
                print(f"Поколение {gen:4d} | Лучшая ошибка: {best_error:.4f} | "
                      f"Протон: {best_details['proton']['calculated']:.2f} | "
                      f"Нейтрон: {best_details['neutron']['calculated']:.2f} | "
                      f"π⁺: {best_details['pi+']['calculated']:.2f}")
        
        # Сохранение результатов
        self.best_params = best_individual
        self.model.params.update(best_params)
        
        self.save_results(best_individual, best_error, best_details)
        
        return best_individual, best_error, best_details
    
    def save_results(self, params, error, details):
        """Сохранение результатов оптимизации"""
        results = {
            'model_version': 'v21.1.1',
            'timestamp': datetime.now().isoformat(),
            'total_error': error,
            'parameters': params,
            'predictions': details,
            'targets': self.model.targets
        }
        
        # JSON файл
        with open(f"{self.result_dir}/results.json", 'w') as f:
            json.dump(results, f, indent=2, default=self.json_serializer)
        
        # Текстовый отчёт
        self.save_text_report(results)
    
    def save_text_report(self, results):
        """Сохранение текстового отчёта"""
        filename = f"{self.result_dir}/REPORT.txt"
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("v21.1.1 — ИЕРАРХИЧЕСКАЯ МОДЕЛЬ СИНХРОНИЗАЦИИ НИТЕЙ ВРЕМЕНИ\n")
            f.write("="*80 + "\n\n")
            
            f.write("ОПТИМИЗАЦИОННЫЕ РЕЗУЛЬТАТЫ:\n")
            f.write(f"  Общая ошибка: {results['total_error']:.6f}\n")
            f.write(f"  Время: {results['timestamp']}\n\n")
            
            f.write("ФИНАЛЬНЫЕ ПАРАМЕТРЫ:\n")
            for key, value in results['parameters'].items():
                f.write(f"  {key}: {value:.6f}\n")
            
            f.write("\nПРЕДСКАЗАНИЯ МОДЕЛИ:\n")
            f.write(f"{'Частица':<10} {'Расчёт (МэВ)':<15} {'Цель (МэВ)':<15} {'Ошибка (МэВ)':<15} {'Ошибка (%)':<10}\n")
            f.write("-"*80 + "\n")
            
            for name in self.model.targets.keys():
                pred = results['predictions'][name]
                f.write(f"{name:<10} {pred['calculated']:<15.3f} {pred['target']:<15.3f} "
                       f"{pred['abs_error']:<15.3f} {pred['rel_error']:<10.3f}\n")
            
            # Разность масс n-p
            mass_diff = (results['predictions']['neutron']['calculated'] - 
                        results['predictions']['proton']['calculated'])
            f.write(f"\nРазность масс n-p: {mass_diff:.3f} МэВ (цель: 1.293 МэВ)\n")
            
            # Физические параметры
            m_u = results['parameters']['m_u_eff']
            m_d = results['parameters']['m_d_eff']
            ratio = m_d / m_u
            f.write(f"\nФизические параметры:\n")
            f.write(f"  Эффективная масса u-кварка: {m_u:.2f} МэВ\n")
            f.write(f"  Эффективная масса d-кварка: {m_d:.2f} МэВ\n")
            f.write(f"  Отношение m_d/m_u: {ratio:.3f}\n")
            
            # Свойства синхронизации
            f.write(f"\nЭнергии синхронизации:\n")
            f.write(f"  Протон: {results['parameters']['E_sync_proton']:.3f} МэВ\n")
            f.write(f"  Нейтрон: {results['parameters']['E_sync_neutron']:.3f} МэВ\n")
            f.write(f"  Отношение neutron/proton: {results['parameters']['E_sync_neutron']/results['parameters']['E_sync_proton']:.3f}\n")
            f.write(f"  Мезоны (заряженные): {results['parameters']['E_sync_meson_charged']:.3f} МэВ\n")
            f.write(f"  Мезоны (нейтральные): {results['parameters']['E_sync_meson_neutral']:.3f} МэВ\n")
            
            f.write("\n" + "="*80 + "\n")
            f.write("ФИЗИЧЕСКАЯ ИНТЕРПРЕТАЦИЯ:\n")
            f.write("-"*80 + "\n")
            f.write("1. Барионы: энергия синхронизации увеличивает массу\n")
            f.write("2. Мезоны: энергия синхронизации уменьшает массу\n")
            f.write("3. Заряженные мезоны имеют электромагнитную поправку\n")
            f.write("4. Нейтральный пион — суперпозиция состояний\n")
            f.write("="*80 + "\n")
    
    def json_serializer(self, obj):
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return str(obj)

def run_initial_test():
    """Быстрый тест модели с начальными параметрами"""
    print("\n" + "="*80)
    print("v21.1.1: БЫСТРЫЙ ТЕСТ С НАЧАЛЬНЫМИ ПАРАМЕТРАМИ")
    print("="*80)
    
    model = HierarchicalThreadModel()
    
    results = {}
    for name in model.targets.keys():
        mass = model.calculate_mass(name)
        target = model.targets[name]['mass']
        error = abs(mass - target) / target * 100
        results[name] = (mass, target, error)
        
        status = "✓" if error < 0.1 else "⚠" if error < 1.0 else "✗"
        print(f"{status} {name:6}: {mass:8.3f} МэВ (цель {target:7.3f}) — ошибка {error:5.2f}%")
    
    # Разность масс n-p
    diff = results['neutron'][0] - results['proton'][0]
    target_diff = 1.293
    diff_error = abs(diff - target_diff)
    print(f"\nРазность масс n-p: {diff:.3f} МэВ (цель {target_diff:.3f})")
    print(f"Ошибка разности: {diff_error:.3f} МэВ ({diff_error/target_diff*100:.2f}%)")
    
    # Физические параметры
    print(f"\nФизические параметры:")
    print(f"  m_u_eff: {model.params['m_u_eff']:.2f} МэВ")
    print(f"  m_d_eff: {model.params['m_d_eff']:.2f} МэВ")
    print(f"  Отношение m_d/m_u: {model.params['m_d_eff']/model.params['m_u_eff']:.3f}")
    
    return model

# ================= ЗАПУСК =================
if __name__ == "__main__":
    print("="*80)
    print("ПРОЕКТ v21.1.1 — ИЕРАРХИЧЕСКАЯ МОДЕЛЬ СИНХРОНИЗАЦИИ")
    print("="*80)
    
    # Шаг 1: Быстрый тест с начальными параметрами
    model = run_initial_test()
    
    # Шаг 2: Оптимизация
    proceed = input("\nЗапустить полную оптимизацию? (y/n): ")
    
    if proceed.lower() == 'y':
        optimizer = HierarchicalOptimizer()
        
        # Запуск генетического алгоритма
        best_params, best_error, best_details = optimizer.run_genetic_optimization(
            generations=500,      # Количество поколений
            population_size=30    # Размер популяции
        )
        
        print("\n" + "="*80)
        print("ОПТИМИЗАЦИЯ ЗАВЕРШЕНА!")
        print("="*80)
        
        # Создание улучшенной модели
        improved_model = HierarchicalThreadModel(best_params)
        
        # Тестирование улучшенной модели
        print("\nУЛУЧШЕННАЯ МОДЕЛЬ:")
        total_error = 0
        for name in improved_model.targets.keys():
            mass = improved_model.calculate_mass(name)
            target = improved_model.targets[name]['mass']
            error = abs(mass - target) / target * 100
            total_error += error
            
            status = "✓" if error < 0.1 else "⚠" if error < 1.0 else "✗"
            print(f"{status} {name:6}: {mass:8.3f} МэВ (цель {target:7.3f}) — ошибка {error:5.2f}%")
        
        avg_error = total_error / len(improved_model.targets)
        print(f"\nСредняя ошибка: {avg_error:.2f}%")
        
        # Разность масс
        diff = improved_model.calculate_mass('neutron') - improved_model.calculate_mass('proton')
        print(f"Разность масс n-p: {diff:.3f} МэВ")
        
        # Сохранение финальной модели
        with open(f"{optimizer.result_dir}/final_model.json", 'w') as f:
            json.dump(best_params, f, indent=2)
        
        print(f"\nРезультаты сохранены в директории: {optimizer.result_dir}")
    
    else:
        print("\nТолько тестовый прогон завершён.")