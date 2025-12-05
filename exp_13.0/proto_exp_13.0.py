import numpy as np
from scipy.optimize import minimize
import json
from datetime import datetime
import os

class KnotTheoryHadronModel:
    """Модель адронов на основе теории узлов и синхронизации"""
    
    def __init__(self):
        # Полиномы Джонса для различных узлов
        self.knot_polynomials = {
            'unknot': lambda t: 1,  # тривиальный узел
            'trefoil': lambda t: t + t**3 - t**4,  # трилистник (протон)
            'figure8': lambda t: t**-2 - t**-1 + 1 - t + t**2,  # восьмёрка (нейтрон)
            'hopf': lambda t: -t**(5/2) - t**(1/2),  # зацепление Хопфа (мезон)
        }
        
        # Соответствие частиц узлам
        self.particle_knots = {
            'proton': 'trefoil',
            'neutron': 'figure8', 
            'pi+': 'hopf',
            'pi0': 'hopf',  # но с другим параметром
            'pi-': 'hopf',
            'K+': 'hopf',   # с возбуждением
            'K0': 'hopf',
            'K-': 'hopf',
            'eta': 'unknot',  # особый случай
            'Lambda0': 'trefoil',  # с возбуждением
        }
        
        # Целевые массы
        self.target_masses = {
            'proton': 938.272,
            'neutron': 939.565,
            'pi+': 139.570,
            'pi0': 134.9768,
            'pi-': 139.570,
            'K+': 493.677,
            'K0': 497.611,
            'K-': 493.677,
            'eta': 547.862,
            'Lambda0': 1115.683,
        }
        
        # Параметры модели
        self.params = {
            'm_u': 2.2,      # эффективная масса u (в единицах модели)
            'm_d': 3.5,      # эффективная масса d
            'm_s': 10.0,     # эффективная масса s
            'alpha': 1.5,    # коэффициент топологической энергии
            'beta': 0.3,     # коэффициент возбуждения (для s-кварка)
            'gamma': 0.1,    # электромагнитная поправка
            't_value': np.exp(1j * np.pi/3),  # точка вычисления полинома Джонса
            'scale': 100.0,
        }
    
    def knot_energy(self, knot_type, excitation=0):
        """Топологическая энергия из полинома Джонса"""
        poly = self.knot_polynomials[knot_type]
        val = poly(self.params['t_value'])
        # Модуль комплексного числа
        energy = np.abs(val) * self.params['alpha']
        
        # Поправка на возбуждение (для странных частиц)
        if excitation > 0:
            energy *= (1 + self.params['beta'] * excitation)
        
        return energy
    
    def calculate_mass(self, particle):
        """Расчёт массы частицы"""
        knot_type = self.particle_knots[particle]
        
        # Базовая масса из состава
        base_mass = 0
        excitation = 0
        
        if particle == 'proton':  # uud
            base_mass = 2*self.params['m_u'] + self.params['m_d']
        elif particle == 'neutron':  # udd
            base_mass = self.params['m_u'] + 2*self.params['m_d']
        elif particle in ['pi+', 'pi0', 'pi-']:  # мезоны
            base_mass = self.params['m_u'] + self.params['m_d']
        elif particle in ['K+', 'K0', 'K-']:  # странные мезоны
            base_mass = self.params['m_u'] + self.params['m_s']
            excitation = 1
        elif particle == 'eta':  # особый случай
            base_mass = 2*self.params['m_u'] + 2*self.params['m_d'] + 2*self.params['m_s']
            # η имеет тривиальный узел, но большую массу из-за смешивания
            knot_energy = self.knot_energy(knot_type, 0)
            # Для η энергия связи отрицательна (мезон), но есть дополнительная масса смешивания
            mix_energy = 4.0  # параметр смешивания
            return (base_mass - knot_energy + mix_energy) * self.params['scale']
        elif particle == 'Lambda0':  # uds
            base_mass = self.params['m_u'] + self.params['m_d'] + self.params['m_s']
            excitation = 1
        
        knot_energy = self.knot_energy(knot_type, excitation)
        
        # Знак: + для барионов, - для мезонов
        if particle in ['proton', 'neutron', 'Lambda0']:
            mass = (base_mass + knot_energy) * self.params['scale']
        else:
            mass = (base_mass - knot_energy) * self.params['scale']
        
        # Электромагнитная поправка для заряженных частиц
        if particle in ['pi+', 'pi-', 'K+', 'K-', 'proton']:
            mass += self.params['gamma'] * 5.0  # ~5 МэВ
        
        return mass
    
    def error_function(self, params_array):
        """Функция ошибки для оптимизации"""
        # Обновляем параметры
        param_names = list(self.params.keys())[:-1]  # кроме scale
        for i, name in enumerate(param_names):
            self.params[name] = params_array[i]
        
        total_error = 0
        for particle, target in self.target_masses.items():
            calculated = self.calculate_mass(particle)
            error = (calculated - target) / target
            total_error += error**2
        
        return total_error
    
    def optimize(self):
        """Оптимизация параметров"""
        initial_guess = [self.params[name] for name in ['m_u', 'm_d', 'm_s', 'alpha', 'beta', 'gamma']]
        
        bounds = [
            (1.5, 3.0),   # m_u
            (2.5, 5.0),   # m_d
            (7.0, 15.0),  # m_s
            (0.5, 3.0),   # alpha
            (0.0, 1.0),   # beta
            (0.0, 0.5),   # gamma
        ]
        
        result = minimize(self.error_function, initial_guess, 
                         bounds=bounds, method='L-BFGS-B',
                         options={'maxiter': 10000, 'ftol': 1e-10})
        
        # Обновляем оптимальные параметры
        optimal_params = result.x
        param_names = ['m_u', 'm_d', 'm_s', 'alpha', 'beta', 'gamma']
        for i, name in enumerate(param_names):
            self.params[name] = optimal_params[i]
        
        return result.fun, optimal_params
    
    def calculate_all(self):
        """Расчёт всех масс"""
        results = {}
        for particle in self.target_masses:
            results[particle] = self.calculate_mass(particle)
        return results

def main():
    print("="*80)
    print("МОДЕЛЬ АДРОНОВ v13.0 - ТЕОРИЯ УЗЛОВ И СИНХРОНИЗАЦИЯ")
    print("="*80)
    
    model = KnotTheoryHadronModel()
    
    print("\nНачальные параметры:")
    for k, v in model.params.items():
        if k != 't_value':
            print(f"{k}: {v}")
    
    print("\nНачальные массы:")
    init_masses = model.calculate_all()
    for p in ['proton', 'neutron', 'pi+', 'pi0', 'K+', 'eta', 'Lambda0']:
        target = model.target_masses[p]
        calc = init_masses[p]
        err = abs(calc - target) / target * 100
        print(f"{p}: {calc:.3f} МэВ (цель {target:.3f}) - {err:.2f}%")
    
    print("\nЗапуск оптимизации...")
    best_error, best_params = model.optimize()
    
    print(f"\nЛучшая ошибка: {best_error:.6f}")
    
    print("\nОптимальные параметры:")
    param_names = ['m_u', 'm_d', 'm_s', 'alpha', 'beta', 'gamma']
    for name, value in zip(param_names, best_params):
        print(f"{name}: {value:.6f}")
    
    print("\nФинальные массы:")
    final_masses = model.calculate_all()
    
    particles = ['proton', 'neutron', 'pi+', 'pi0', 'pi-', 'K+', 'K0', 'K-', 'eta', 'Lambda0']
    
    print(f"\n{'Частица':<10} {'Расчёт':<10} {'Цель':<10} {'Ошибка %':<8}")
    print("-"*45)
    
    for p in particles:
        target = model.target_masses[p]
        calc = final_masses[p]
        err = abs(calc - target) / target * 100
        print(f"{p:<10} {calc:<10.3f} {target:<10.3f} {err:<8.3f}")
    
    # Сохранение результатов
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"knot_model_v13_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    
    results = {
        'parameters': dict(zip(param_names, best_params)),
        'error': best_error,
        'masses': final_masses,
        'model': 'v13.0_knot_theory'
    }
    
    with open(f"{results_dir}/results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nРезультаты сохранены в {results_dir}/")
    print("="*80)

if __name__ == "__main__":
    main()