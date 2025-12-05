"""
МОДЕЛЬ v7.3 - СРОЧНЫЙ ФИКС ДЛЯ π⁰ И s-КВАРКА
1. Исправляем фазы для одинаковых кварк-антикварк пар
2. Уменьшаем массу s-кварка в 6 раз
3. Корректируем coupling для странных частиц
"""

import numpy as np
import time
from datetime import datetime
import os

# ============== ЭКСТРЕННЫЕ ИСПРАВЛЕНИЯ ==============

class EmergencyFixModel:
    def __init__(self):
        # Берем идеальные параметры v6.1
        self.params = {
            'base_mass_u': 2.203806,
            'base_mass_d': 4.583020,
            'base_mass_s': 9.166,  # ЭКСТРЕННО: уменьшили в 6 раз! (55 → 9.166)
            
            'freq_u': 0.956359,
            'freq_d': 0.868115,
            'freq_s': 0.7,  # Частота s-кварка
            
            'amp_u': 1.032476,
            'amp_d': 0.877773,
            'amp_s': 0.75,  # Амплитуда s-кварка
            
            'coupling_proton': 1.613565,
            'coupling_neutron': 0.285395,
            'coupling_meson_light': 4.273121,
            'coupling_meson_strange': 8.0,  # ЭКСТРЕННО: увеличили на 60% (5 → 8)
            'coupling_lambda0': 1.2,  # ЭКСТРЕННО: увеличили в 4 раза (0.3 → 1.2)
            
            'phase_shift': 3.173848,
            'scale_factor': 100.0
        }
        
        # НОВЫЕ ПРАВИЛА ДЛЯ ФАЗ:
        self.phase_rules = {
            'proton': [0, 0, np.pi/2],           # u, u, d
            'neutron': [0, np.pi/2, np.pi/2],    # u, d, d
            'lambda0': [0, np.pi/2, np.pi],      # u, d, s
            
            'pi+': [0, np.pi],                   # u, anti_d
            'pi0': [0, 0],                       # u, anti_u - ИЗМЕНИЛИ!
            'pi-': [0, np.pi],                   # d, anti_u
            
            'k+': [0, np.pi],                    # u, anti_s
            'k0': [0, np.pi],                    # d, anti_s
            'k-': [0, np.pi],                    # s, anti_u
            'k0_bar': [0, np.pi]                 # s, anti_d
        }
    
    def calculate_mass(self, particle_name, composition):
        """Быстрый расчет массы с экстренными исправлениями"""
        # 1. Расчет базовой массы
        base = 0.0
        for quark in composition:
            if quark in ['u', 'anti_u']:
                base += self.params['base_mass_u'] * self.params['freq_u'] * self.params['amp_u']
            elif quark in ['d', 'anti_d']:
                base += self.params['base_mass_d'] * self.params['freq_d'] * self.params['amp_d']
            elif quark in ['s', 'anti_s']:
                base += self.params['base_mass_s'] * self.params['freq_s'] * self.params['amp_s']
        
        # 2. Энергия синхронизации
        is_meson = len(composition) == 2
        
        if particle_name in ['proton', 'neutron', 'lambda0']:
            if particle_name == 'proton':
                coupling = self.params['coupling_proton']
            elif particle_name == 'neutron':
                coupling = self.params['coupling_neutron']
            elif particle_name == 'lambda0':
                coupling = self.params['coupling_lambda0']
            
            phases = self.phase_rules[particle_name]
            
            # Для барионов: расчет когерентности 3 нитей
            phase_coherence_sum = 0
            for i in range(3):
                for j in range(i+1, 3):
                    diff = abs(phases[i] - phases[j])
                    phase_coherence_sum += np.cos(diff)
            phase_coherence = (phase_coherence_sum / 3 + 1) / 2
            
            sync_energy = coupling * phase_coherence
            total = base + sync_energy  # Для барионов: ПЛЮС
            
        else:  # Мезоны
            if 'k' in particle_name:  # Каоны
                coupling = self.params['coupling_meson_strange']
            else:  # Пионы
                coupling = self.params['coupling_meson_light']
            
            phases = self.phase_rules[particle_name]
            phase_coherence = np.cos(phases[1] - phases[0] + np.pi)
            phase_coherence = (phase_coherence + 1) / 2
            
            sync_energy = coupling * phase_coherence
            total = base - sync_energy  # Для мезонов: МИНУС
        
        return total * self.params['scale_factor']

# ============== БЫСТРАЯ ОПТИМИЗАЦИЯ ==============

def quick_optimization():
    """Быстрая ручная оптимизация с фокусом на проблемах"""
    print("="*80)
    print("ЭКСТРЕННЫЙ ФИКС v7.3")
    print("Исправляем π⁰ и s-кварк")
    print("="*80)
    
    model = EmergencyFixModel()
    targets = {
        'proton': 938.272,
        'neutron': 939.565,
        'pi+': 139.570,
        'pi0': 134.9768,
        'pi-': 139.570,
        'k+': 493.677,
        'k0': 497.611,
        'k-': 493.677,
        'k0_bar': 497.611,
        'lambda0': 1115.683
    }
    
    # Рассчитываем текущие массы
    print("\nТЕКУЩИЕ МАССЫ:")
    total_error = 0
    for name, target in targets.items():
        if name == 'proton':
            mass = model.calculate_mass('proton', ['u', 'u', 'd'])
        elif name == 'neutron':
            mass = model.calculate_mass('neutron', ['u', 'd', 'd'])
        elif name == 'lambda0':
            mass = model.calculate_mass('lambda0', ['u', 'd', 's'])
        elif name == 'pi+':
            mass = model.calculate_mass('pi+', ['u', 'anti_d'])
        elif name == 'pi0':
            mass = model.calculate_mass('pi0', ['u', 'anti_u'])
        elif name == 'pi-':
            mass = model.calculate_mass('pi-', ['d', 'anti_u'])
        elif name == 'k+':
            mass = model.calculate_mass('k+', ['u', 'anti_s'])
        elif name == 'k0':
            mass = model.calculate_mass('k0', ['d', 'anti_s'])
        elif name == 'k-':
            mass = model.calculate_mass('k-', ['s', 'anti_u'])
        elif name == 'k0_bar':
            mass = model.calculate_mass('k0_bar', ['s', 'anti_d'])
        
        error = abs(mass - target) / target * 100
        total_error += error
        print(f"  {name}: {mass:.3f} МэВ (цель {target:.3f}) - {error:.2f}%")
    
    # Рассчитываем эффективные массы
    m_u = model.params['base_mass_u'] * model.params['freq_u'] * model.params['amp_u'] * 100
    m_d = model.params['base_mass_d'] * model.params['freq_d'] * model.params['amp_d'] * 100
    m_s = model.params['base_mass_s'] * model.params['freq_s'] * model.params['amp_s'] * 100
    
    print(f"\nЭФФЕКТИВНЫЕ МАССЫ:")
    print(f"  u: {m_u:.2f} МэВ")
    print(f"  d: {m_d:.2f} МэВ")
    print(f"  s: {m_s:.2f} МэВ")
    print(f"  m_s/m_u: {m_s/m_u:.2f}")
    
    print(f"\nCoupling параметры:")
    print(f"  coupling_meson_light: {model.params['coupling_meson_light']:.3f}")
    print(f"  coupling_meson_strange: {model.params['coupling_meson_strange']:.3f}")
    print(f"  coupling_lambda0: {model.params['coupling_lambda0']:.3f}")
    
    print(f"\nСредняя ошибка: {total_error/len(targets):.2f}%")
    
    # Сохранение параметров
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = f"emergency_fix_v73_{timestamp}"
    os.makedirs(result_dir, exist_ok=True)
    
    with open(f"{result_dir}/params.json", 'w') as f:
        import json
        json.dump(model.params, f, indent=2)
    
    print(f"\nПараметры сохранены в: {result_dir}")
    
    return model.params

# ============== ЗАПУСК ==============

if __name__ == "__main__":
    print("="*80)
    print("ЭКСТРЕННЫЙ ФИКС v7.3 - ЗАПУСК")
    print("Исправления:")
    print("  1. base_mass_s уменьшен в 6 раз (55 → 9.166)")
    print("  2. coupling_meson_strange увеличен на 60% (5 → 8)")
    print("  3. coupling_lambda0 увеличен в 4 раза (0.3 → 1.2)")
    print("  4. Фаза для π⁰ изменена с [0, π] на [0, 0]")
    print("="*80)
    
    try:
        params = quick_optimization()
        
        # Краткий анализ
        print("\n" + "="*80)
        print("АНАЛИЗ ИСПРАВЛЕНИЙ:")
        print("="*80)
        
        # Создаем модель с исправлениями
        model = EmergencyFixModel()
        
        # Тестируем ключевые проблемы
        pi0_mass = model.calculate_mass('pi0', ['u', 'anti_u'])
        k_plus_mass = model.calculate_mass('k+', ['u', 'anti_s'])
        lambda_mass = model.calculate_mass('lambda0', ['u', 'd', 's'])
        
        print(f"  π⁰ масса: {pi0_mass:.1f} МэВ (было 8, нужно 135)")
        print(f"  K⁺ масса: {k_plus_mass:.1f} МэВ (было 2605, нужно 494)")
        print(f"  Λ⁰ масса: {lambda_mass:.1f} МэВ (было 3464, нужно 1116)")
        
        if pi0_mass > 100 and pi0_mass < 170:
            print(f"  ✓ π⁰ исправлен!")
        else:
            print(f"  ✗ π⁰ всё ещё неверен")
        
        if k_plus_mass < 600 and k_plus_mass > 400:
            print(f"  ✓ K⁺ приблизился к цели!")
        else:
            print(f"  ✗ K⁺ всё ещё далеко")
        
        print(f"\nДальнейшие шаги:")
        print(f"  1. Если π⁰ исправлен, но другие частицы ухудшились - нужна тонкая настройка")
        print(f"  2. Если всё ещё плохо - пересмотреть модель синхронизации для мезонов")
        print(f"  3. Возможно, нужны разные формулы для разных типов частиц")
        
    except Exception as e:
        print(f"\nОШИБКА: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*80)
    print("ВЫЧИСЛЕНИЯ ЗАВЕРШЕНЫ")
    print("="*80)