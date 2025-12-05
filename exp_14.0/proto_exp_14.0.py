import numpy as np
import json
from datetime import datetime
import os

class TopologicalCouplingModelV14:
    """Модель v14.0: coupling вычисляется из топологии и цвета"""
    
    def __init__(self):
        # Константы из v6.1 (оптимальные)
        self.base_params = {
            'm_u_eff': 2.203806,  # m_u × freq_u × amp_u из v6.1
            'm_d_eff': 4.583020,  # m_d × freq_d × amp_d из v6.1
            'm_s_eff': 13.0,      # Оценочно для s-кварка (будем оптимизировать)
            'scale': 100.0
        }
        
        # Целевые массы
        self.targets = {
            'proton': 938.272,
            'neutron': 939.565,
            'pi+': 139.570,
            'pi0': 134.9768,
            'pi-': 139.570,
            'K+': 493.677,
            'K0': 497.611,
            'eta': 547.862,
            'Lambda0': 1115.683,
        }
        
        # Состав
        self.composition = {
            'proton': ['u', 'u', 'd'],
            'neutron': ['u', 'd', 'd'],
            'pi+': ['u', 'anti_d'],
            'pi0': ['u', 'anti_u'],
            'pi-': ['d', 'anti_u'],
            'K+': ['u', 'anti_s'],
            'K0': ['d', 'anti_s'],
            'eta': ['mixed'],  # uubar + ddbar + ssbar смесь
            'Lambda0': ['u', 'd', 's'],
        }
        
        # Параметры для вычисления coupling
        self.coupling_params = {
            'color_factor_baryon': 1.0,    # Цветовой фактор для барионов
            'color_factor_meson': 2.0,     # Цветовой фактор для мезонов
            'phase_coherence_proton': 1.1, # Фазовая когерентность протона
            'phase_coherence_neutron': 0.95, # Фазовая когерентность нейтрона
            'phase_coherence_meson': 1.0,  # Фазовая когерентность мезонов
            'spin_factor_proton': 1.0,     # Спиновый фактор
            'spin_factor_neutron': 0.9,
            'strangeness_factor': 0.7,     # Фактор странности
            'eta_mixing': 0.4,             # Параметр смешивания для η
        }
    
    def calculate_coupling(self, particle):
        """Вычисление coupling из топологических параметров"""
        if particle == 'proton':
            # uud, цветовая конфигурация RGB, спин 1/2
            color = self.coupling_params['color_factor_baryon']
            phase = self.coupling_params['phase_coherence_proton']
            spin = self.coupling_params['spin_factor_proton']
            # Формула из v6.1: 1.613565
            return color * phase * spin * 1.5  # Подгоночный коэффициент
        
        elif particle == 'neutron':
            # udd, цветовая конфигурация RGB, спин 1/2
            color = self.coupling_params['color_factor_baryon']
            phase = self.coupling_params['phase_coherence_neutron']
            spin = self.coupling_params['spin_factor_neutron']
            # Важно: coupling нейтрона в 5.65 раз меньше протонного!
            return color * phase * spin * 0.3
        
        elif particle in ['pi+', 'pi-']:
            # Кварк-антикварк, цвет R-anti_R
            color = self.coupling_params['color_factor_meson']
            phase = self.coupling_params['phase_coherence_meson']
            # Мезоны: coupling большой (4.273121 в v6.1)
            return color * phase * 2.1
        
        elif particle == 'pi0':
            # Нейтральный пион: u-anti_u и d-anti_d смесь
            # coupling немного меньше из-за смешивания
            color = self.coupling_params['color_factor_meson']
            phase = self.coupling_params['phase_coherence_meson'] * 0.95
            return color * phase * 2.1
        
        elif particle in ['K+', 'K0', 'K-']:
            # Странные мезоны: есть s-кварк
            color = self.coupling_params['color_factor_meson']
            phase = self.coupling_params['phase_coherence_meson']
            strangeness = self.coupling_params['strangeness_factor']
            return color * phase * strangeness * 2.5
        
        elif particle == 'eta':
            # η-мезон: сложное смешивание
            color = self.coupling_params['color_factor_meson']
            mixing = self.coupling_params['eta_mixing']
            return color * mixing * 3.0
        
        elif particle == 'Lambda0':
            # uds барион
            color = self.coupling_params['color_factor_baryon']
            strangeness = self.coupling_params['strangeness_factor']
            return color * strangeness * 1.8
        
        else:
            return 1.0
    
    def calculate_base_mass(self, particle):
        """Базовая масса из состава"""
        comp = self.composition[particle]
        total = 0
        
        if particle == 'eta':
            # η = (uubar + ddbar + ssbar)/√3 примерно
            return (2*self.base_params['m_u_eff'] + 
                    2*self.base_params['m_d_eff'] + 
                    self.base_params['m_s_eff']) / 3.0
        
        for quark in comp:
            if quark in ['u', 'anti_u']:
                total += self.base_params['m_u_eff']
            elif quark in ['d', 'anti_d']:
                total += self.base_params['m_d_eff']
            elif quark in ['s', 'anti_s']:
                total += self.base_params['m_s_eff']
        
        return total
    
    def calculate_mass(self, particle):
        """Расчёт массы по формуле v6.1"""
        base = self.calculate_base_mass(particle)
        coupling = self.calculate_coupling(particle)
        
        # Барионы: base + coupling, мезоны: base - coupling
        if particle in ['proton', 'neutron', 'Lambda0']:
            mass = (base + coupling) * self.base_params['scale']
        else:
            mass = (base - coupling) * self.base_params['scale']
        
        return mass
    
    def error_function(self, params):
        """Функция ошибки"""
        # params: [m_s_eff, color_baryon, color_meson, phase_proton, 
        #          phase_neutron, phase_meson, spin_proton, spin_neutron,
        #          strangeness, eta_mixing]
        
        # Обновляем параметры
        self.base_params['m_s_eff'] = params[0]
        
        self.coupling_params = {
            'color_factor_baryon': params[1],
            'color_factor_meson': params[2],
            'phase_coherence_proton': params[3],
            'phase_coherence_neutron': params[4],
            'phase_coherence_meson': params[5],
            'spin_factor_proton': params[6],
            'spin_factor_neutron': params[7],
            'strangeness_factor': params[8],
            'eta_mixing': params[9],
        }
        
        # Вычисляем ошибку
        total_error = 0
        for particle, target in self.targets.items():
            calculated = self.calculate_mass(particle)
            rel_error = (calculated - target) / target
            total_error += rel_error ** 2
        
        # Штрафы за нефизичность
        if self.coupling_params['phase_coherence_neutron'] > \
           self.coupling_params['phase_coherence_proton']:
            total_error += 10.0
        
        if params[0] < self.base_params['m_d_eff']:  # m_s должна быть > m_d
            total_error += 10.0
        
        return total_error

def optimize_v14():
    """Оптимизация модели v14.0"""
    print("="*80)
    print("МОДЕЛЬ v14.0 - ТОПОЛОГИЧЕСКОЕ ОБОСНОВАНИЕ v6.1")
    print("="*80)
    
    model = TopologicalCouplingModelV14()
    
    # Начальные значения параметров
    initial_params = [
        13.0,  # m_s_eff
        1.0,   # color_baryon
        2.0,   # color_meson
        1.1,   # phase_proton
        0.95,  # phase_neutron
        1.0,   # phase_meson
        1.0,   # spin_proton
        0.9,   # spin_neutron
        0.7,   # strangeness
        0.4,   # eta_mixing
    ]
    
    # Границы параметров
    bounds = [
        (8.0, 20.0),   # m_s_eff
        (0.5, 3.0),    # color_baryon
        (1.0, 4.0),    # color_meson
        (0.8, 1.5),    # phase_proton
        (0.7, 1.2),    # phase_neutron
        (0.5, 2.0),    # phase_meson
        (0.5, 1.5),    # spin_proton
        (0.5, 1.5),    # spin_neutron
        (0.3, 1.2),    # strangeness
        (0.1, 1.0),    # eta_mixing
    ]
    
    # Простая оптимизация методом случайного поиска
    best_params = initial_params
    best_error = float('inf')
    history = []
    
    np.random.seed(42)
    
    for iteration in range(10000):
        # Генерация случайных параметров
        trial_params = []
        for i, (min_val, max_val) in enumerate(bounds):
            if iteration == 0:
                trial_params.append(initial_params[i])
            else:
                # Локальный поиск вокруг лучших параметров
                current = best_params[i]
                step = (max_val - min_val) * 0.1
                trial = current + np.random.uniform(-step, step)
                trial = max(min_val, min(max_val, trial))
                trial_params.append(trial)
        
        # Оценка
        error = model.error_function(trial_params)
        
        if error < best_error:
            best_error = error
            best_params = trial_params.copy()
            
            if iteration % 100 == 0:
                print(f"Iteration {iteration}: error = {best_error:.6f}")
                history.append((iteration, best_error, best_params.copy()))
    
    print(f"\nОптимизация завершена. Лучшая ошибка: {best_error:.6f}")
    
    # Вычисляем финальные массы
    model.base_params['m_s_eff'] = best_params[0]
    model.coupling_params = {
        'color_factor_baryon': best_params[1],
        'color_factor_meson': best_params[2],
        'phase_coherence_proton': best_params[3],
        'phase_coherence_neutron': best_params[4],
        'phase_coherence_meson': best_params[5],
        'spin_factor_proton': best_params[6],
        'spin_factor_neutron': best_params[7],
        'strangeness_factor': best_params[8],
        'eta_mixing': best_params[9],
    }
    
    # Вывод результатов
    print("\n" + "="*80)
    print("ФИНАЛЬНЫЕ ПАРАМЕТРЫ")
    print("="*80)
    
    print(f"\nБазовые параметры (из v6.1):")
    print(f"  m_u_eff: {model.base_params['m_u_eff']:.6f}")
    print(f"  m_d_eff: {model.base_params['m_d_eff']:.6f}")
    print(f"  m_s_eff: {model.base_params['m_s_eff']:.6f}")
    
    print(f"\nТопологические параметры coupling:")
    param_names = [
        'color_factor_baryon', 'color_factor_meson',
        'phase_coherence_proton', 'phase_coherence_neutron',
        'phase_coherence_meson', 'spin_factor_proton',
        'spin_factor_neutron', 'strangeness_factor', 'eta_mixing'
    ]
    
    for i, name in enumerate(param_names):
        print(f"  {name}: {best_params[i+1]:.6f}")
    
    print(f"\nРасчётные coupling:")
    for particle in model.targets:
        coupling = model.calculate_coupling(particle)
        print(f"  {particle}: {coupling:.6f}")
    
    print(f"\nМАССЫ ЧАСТИЦ:")
    print(f"{'Частица':<10} {'Расчёт':<10} {'Цель':<10} {'Ошибка %':<10}")
    print("-"*50)
    
    total_error = 0
    for particle, target in model.targets.items():
        mass = model.calculate_mass(particle)
        error_pct = abs(mass - target) / target * 100
        total_error += error_pct
        
        status = "✓" if error_pct < 1.0 else "⚠" if error_pct < 5.0 else "✗"
        print(f"{status} {particle:<8} {mass:<10.3f} {target:<10.3f} {error_pct:<10.3f}")
    
    avg_error = total_error / len(model.targets)
    print(f"\nСредняя ошибка: {avg_error:.2f}%")
    
    # Сохранение результатов
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"topological_coupling_v14_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    
    results = {
        'model': 'v14.0_topological_coupling',
        'timestamp': datetime.now().isoformat(),
        'base_params': model.base_params,
        'coupling_params': model.coupling_params,
        'optimization': {
            'best_error': best_error,
            'iterations': 10000,
            'history': history[:10]  # первые 10 записей
        },
        'masses': {p: float(model.calculate_mass(p)) for p in model.targets},
        'couplings': {p: float(model.calculate_coupling(p)) for p in model.targets}
    }
    
    with open(f"{results_dir}/results.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Текстовый отчёт
    report_file = f"{results_dir}/report.txt"
    with open(report_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("МОДЕЛЬ v14.0 - ОТЧЁТ\n")
        f.write("="*80 + "\n\n")
        
        f.write("ФОРМУЛА МОДЕЛИ:\n")
        f.write("  M = (sum(m_i) + sign * coupling) * 100\n")
        f.write("  где sign = +1 для барионов, -1 для мезонов\n")
        f.write("  coupling вычисляется из топологии, цвета и спина\n\n")
        
        f.write("БАЗОВЫЕ ПАРАМЕТРЫ:\n")
        f.write(f"  m_u_eff: {model.base_params['m_u_eff']:.6f}\n")
        f.write(f"  m_d_eff: {model.base_params['m_d_eff']:.6f}\n")
        f.write(f"  m_s_eff: {model.base_params['m_s_eff']:.6f}\n")
        f.write(f"  scale: {model.base_params['scale']:.1f}\n\n")
        
        f.write("РАСЧЁТНЫЕ COUPLING И МАССЫ:\n")
        f.write(f"{'Частица':<10} {'Coupling':<10} {'Масса':<10} {'Цель':<10} {'Ошибка %':<10}\n")
        f.write("-"*60 + "\n")
        
        for particle in model.targets:
            coupling = model.calculate_coupling(particle)
            mass = model.calculate_mass(particle)
            target = model.targets[particle]
            error_pct = abs(mass - target) / target * 100
            f.write(f"{particle:<10} {coupling:<10.3f} {mass:<10.3f} {target:<10.3f} {error_pct:<10.3f}\n")
    
    print(f"\nРезультаты сохранены в: {results_dir}")
    print("="*80)
    
    return model, best_params, best_error

if __name__ == "__main__":
    model, params, error = optimize_v14()