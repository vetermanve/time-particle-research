"""
v21.5.0 — ФУНДАМЕНТАЛЬНАЯ МОДЕЛЬ СИНХРОНИЗАЦИИ НИТЕЙ
Основа: энергия синхронизации пропорциональна "сложности" резонансной конфигурации
"""

import numpy as np
from scipy.optimize import differential_evolution
import json
from datetime import datetime
import os

class FundamentalThreadModel:
    """Фундаментальная модель на основе синхронизации нитей"""
    
    def __init__(self):
        # ФУНДАМЕНТАЛЬНЫЕ КОНСТАНТЫ (всего 5 параметров!)
        self.params = {
            'E0': 100.0,       # Базовая энергия синхронизации (МэВ)
            'alpha': 1.0,      # Коэффициент сложности конфигурации
            'beta': 1.0,       # Коэффициент для заряженных частиц
            'gamma': 1.0,      # Коэффициент для нейтральных частиц
            'delta': 0.1,      # Коэффициент для разности масс u/d
        }
        
        # ЦЕЛЕВЫЕ МАССЫ
        self.targets = {
            'proton': 938.272,
            'neutron': 939.565,
            'pi+': 139.570,
            'pi0': 134.977,
            'pi-': 139.570,
        }
    
    def configuration_complexity(self, particle):
        """Сложность конфигурации синхронизации"""
        # Чем сложнее синхронизировать нити, тем больше масса
        complexities = {
            'proton': 3.0,     # Три нити, две одинаковые (u,u,d)
            'neutron': 3.1,    # Три нити, две одинаковые (u,d,d) - немного сложнее
            'pi+': 1.0,        # Две нити (кварк-антикварк)
            'pi0': 0.9,        # Нейтральная конфигурация проще
            'pi-': 1.0,        # Как π⁺
        }
        return complexities.get(particle, 1.0)
    
    def charge_factor(self, particle):
        """Фактор заряда"""
        if particle in ['pi+', 'pi-', 'proton']:
            return 1.0 + self.params['beta']
        elif particle == 'pi0':
            return 1.0 - self.params['gamma']
        else:
            return 1.0
    
    def calculate_mass(self, particle):
        """Фундаментальная формула массы"""
        E0 = self.params['E0']
        alpha = self.params['alpha']
        delta = self.params['delta']
        
        # Базовая энергия синхронизации
        base_energy = E0 * self.configuration_complexity(particle)
        
        # Коррекция для различия u/d кварков
        if particle == 'neutron':
            quark_correction = delta * 0.5  # Нейтрон имеет больше d-кварков
        elif particle == 'proton':
            quark_correction = -delta * 0.3
        else:
            quark_correction = 0.0
        
        # Фактор заряда
        charge_fact = self.charge_factor(particle)
        
        # Итоговая масса
        mass = base_energy * alpha * charge_fact + quark_correction
        
        # Особые случаи
        if particle == 'pi0':
            mass *= 0.96  # Нейтральный пион легче
        
        return mass
    
    def calculate_all_masses(self):
        """Рассчитать все массы"""
        masses = {}
        for particle in self.targets.keys():
            masses[particle] = self.calculate_mass(particle)
        return masses

class FundamentalOptimizer:
    """Оптимизатор фундаментальной модели"""
    
    def __init__(self):
        self.model = FundamentalThreadModel()
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.result_dir = f"v21_5_0_results_{timestamp}"
        os.makedirs(self.result_dir, exist_ok=True)
        
        print("\n" + "="*80)
        print("v21.5.0 — ФУНДАМЕНТАЛЬНАЯ МОДЕЛЬ СИНХРОНИЗАЦИИ (5 параметров)")
        print("="*80)
    
    def objective_function(self, params):
        """Целевая функция для оптимизации"""
        # Распаковываем параметры
        self.model.params['E0'] = params[0]
        self.model.params['alpha'] = params[1]
        self.model.params['beta'] = params[2]
        self.model.params['gamma'] = params[3]
        self.model.params['delta'] = params[4]
        
        # Рассчитываем массы
        masses = self.model.calculate_all_masses()
        
        # Считаем ошибку
        total_error = 0.0
        
        for particle, target_mass in self.model.targets.items():
            calculated = masses[particle]
            # Относительная ошибка с весом
            rel_error = abs(calculated - target_mass) / target_mass
            
            # Большой вес для нуклонов
            if particle in ['proton', 'neutron']:
                weight = 2.0
            else:
                weight = 1.0
            
            total_error += weight * rel_error
        
        # Штраф за отрицательные массы
        if any(m <= 0 for m in masses.values()):
            total_error += 100.0
        
        # Штраф за нефизичные параметры
        if params[0] <= 0 or params[1] <= 0:
            total_error += 100.0
        
        return total_error
    
    def run_evolution_optimization(self):
        """Оптимизация методом дифференциальной эволюции"""
        print("\nЗапуск глобальной оптимизации...")
        
        # Границы параметров
        bounds = [
            (50.0, 500.0),    # E0
            (0.5, 5.0),       # alpha
            (0.0, 2.0),       # beta
            (0.0, 2.0),       # gamma
            (0.0, 10.0),      # delta
        ]
        
        # Запуск оптимизации
        result = differential_evolution(
            self.objective_function,
            bounds,
            maxiter=1000,
            popsize=20,
            disp=True,
            workers=-1,
            updating='deferred'
        )
        
        if result.success:
            print("✅ Оптимизация успешна!")
            best_params = result.x
            
            # Применяем лучшие параметры
            self.model.params['E0'] = best_params[0]
            self.model.params['alpha'] = best_params[1]
            self.model.params['beta'] = best_params[2]
            self.model.params['gamma'] = best_params[3]
            self.model.params['delta'] = best_params[4]
            
            # Рассчитываем финальные массы
            masses = self.model.calculate_all_masses()
            
            return best_params, masses, result.fun
        
        else:
            print("❌ Оптимизация не удалась")
            return None, None, None
    
    def save_results(self, params, masses, error):
        """Сохранение результатов"""
        results = {
            'model': 'v21.5.0',
            'timestamp': datetime.now().isoformat(),
            'total_error': error,
            'parameters': {
                'E0': float(params[0]),
                'alpha': float(params[1]),
                'beta': float(params[2]),
                'gamma': float(params[3]),
                'delta': float(params[4]),
            },
            'predictions': {k: float(v) for k, v in masses.items()},
            'targets': self.model.targets
        }
        
        filename = os.path.join(self.result_dir, "results.json")
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        
        self.save_text_report(results)
    
    def save_text_report(self, results):
        """Сохранение текстового отчёта"""
        filename = os.path.join(self.result_dir, "REPORT.txt")
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("v21.5.0 — ФУНДАМЕНТАЛЬНАЯ МОДЕЛЬ СИНХРОНИЗАЦИИ\n")
            f.write("="*80 + "\n\n")
            
            f.write("ОПТИМИЗИРОВАННЫЕ ПАРАМЕТРЫ:\n")
            for key, value in results['parameters'].items():
                f.write(f"  {key}: {value:.6f}\n")
            
            f.write("\nПРЕДСКАЗАНИЯ МОДЕЛИ:\n")
            f.write(f"{'Частица':<10} {'Расчёт':<12} {'Цель':<12} {'Ошибка':<12} {'%':<8}\n")
            f.write("-"*80 + "\n")
            
            total_rel_error = 0
            for particle in ['proton', 'neutron', 'pi+', 'pi0', 'pi-']:
                calculated = results['predictions'][particle]
                target = results['targets'][particle]
                abs_error = abs(calculated - target)
                rel_error = abs_error / target * 100
                total_rel_error += rel_error
                
                status = "✓" if rel_error < 1.0 else "⚠" if rel_error < 5.0 else "✗"
                f.write(f"{status} {particle:<10} {calculated:<12.3f} {target:<12.3f} "
                       f"{abs_error:<12.3f} {rel_error:<8.2f}\n")
            
            avg_error = total_rel_error / 5
            f.write(f"\nСредняя ошибка: {avg_error:.2f}%\n")
            
            # Разность масс n-p
            diff = results['predictions']['neutron'] - results['predictions']['proton']
            f.write(f"Разность масс n-p: {diff:.3f} МэВ (цель 1.293 МэВ)\n")
            
            # Физическая интерпретация
            f.write("\n" + "="*80 + "\n")
            f.write("ФИЗИЧЕСКАЯ ИНТЕРПРЕТАЦИЯ:\n")
            f.write("-"*80 + "\n")
            f.write("E0 = {:.1f} МэВ — базовая энергия синхронизации одной нити\n".format(
                results['parameters']['E0']))
            f.write("α = {:.3f} — общий масштабный коэффициент\n".format(
                results['parameters']['alpha']))
            f.write("β = {:.3f} — поправка для заряженных частиц\n".format(
                results['parameters']['beta']))
            f.write("γ = {:.3f} — поправка для нейтральных частиц\n".format(
                results['parameters']['gamma']))
            f.write("δ = {:.3f} — разница между u и d кварками\n".format(
                results['parameters']['delta']))
            f.write("\nФормула: M = E0 × сложность × α × фактор_заряда ± δ\n")
            f.write("="*80 + "\n")

def run_fundamental_model():
    """Запуск фундаментальной модели"""
    print("="*80)
    print("ЗАПУСК v21.5.0 — ФУНДАМЕНТАЛЬНАЯ МОДЕЛЬ")
    print("="*80)
    
    # Создаём модель и оптимизатор
    model = FundamentalThreadModel()
    optimizer = FundamentalOptimizer()
    
    # Начальное приближение
    print("\nНачальные параметры (до оптимизации):")
    for key, value in model.params.items():
        print(f"  {key}: {value:.3f}")
    
    print("\nНачальные предсказания:")
    initial_masses = model.calculate_all_masses()
    for particle in ['proton', 'neutron', 'pi+', 'pi0', 'pi-']:
        mass = initial_masses[particle]
        target = model.targets[particle]
        error = abs(mass - target) / target * 100
        print(f"  {particle:6}: {mass:8.1f} МэВ (цель {target:7.1f}) — {error:5.1f}%")
    
    # Запуск оптимизации
    proceed = input("\nЗапустить глобальную оптимизацию? (y/n): ")
    
    if proceed.lower() == 'y':
        best_params, masses, error = optimizer.run_evolution_optimization()
        
        if best_params is not None:
            print("\n" + "="*80)
            print("ОПТИМИЗАЦИЯ ЗАВЕРШЕНА!")
            print("="*80)
            
            print("\nЛУЧШИЕ ПАРАМЕТРЫ:")
            print(f"  E0:     {best_params[0]:.3f} МэВ")
            print(f"  alpha:  {best_params[1]:.3f}")
            print(f"  beta:   {best_params[2]:.3f}")
            print(f"  gamma:  {best_params[3]:.3f}")
            print(f"  delta:  {best_params[4]:.3f}")
            
            print("\nФИНАЛЬНЫЕ ПРЕДСКАЗАНИЯ:")
            total_error = 0
            for particle in ['proton', 'neutron', 'pi+', 'pi0', 'pi-']:
                mass = masses[particle]
                target = model.targets[particle]
                error_pct = abs(mass - target) / target * 100
                total_error += error_pct
                
                status = "✓" if error_pct < 1.0 else "⚠" if error_pct < 5.0 else "✗"
                print(f"{status} {particle:6}: {mass:8.3f} МэВ (цель {target:7.3f}) — {error_pct:5.2f}%")
            
            avg_error = total_error / 5
            print(f"\nСредняя ошибка: {avg_error:.2f}%")
            
            # Разность масс
            diff = masses['neutron'] - masses['proton']
            print(f"Разность масс n-p: {diff:.3f} МэВ (цель 1.293 МэВ)")
            
            # Сохранение
            optimizer.save_results(best_params, masses, error)
            print(f"\nРезультаты сохранены в: {optimizer.result_dir}")
        
        else:
            print("Оптимизация не дала результатов.")
    
    else:
        print("\nОптимизация пропущена.")

# ================= ЗАПУСК =================
if __name__ == "__main__":
    run_fundamental_model()