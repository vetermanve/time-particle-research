"""
МОДЕЛЬ v21.2.0 — Исправленная версия с правильными единицами
Базовые параметры в условных единицах (как v6.1), затем масштабирование
"""

import numpy as np
import json
from datetime import datetime
import os

class ThreadModelV210:
    """Модель v21.2.0 с исправленными единицами"""
    
    def __init__(self, params=None):
        # ПАРАМЕТРЫ В УСЛОВНЫХ ЕДИНИЦАХ (как в v6.1)
        if params is None:
            self.params = {
                # БАЗОВЫЕ МАССЫ КВАРКОВ (условные единицы, потом ×100 даст МэВ)
                'm_u_base': 2.203806,  # Условные единицы, эффективная = m_u_base × 100 ≈ 220 МэВ
                'm_d_base': 4.583020,  # Условные единицы, эффективная = m_d_base × 100 ≈ 458 МэВ
                
                # ЭНЕРГИИ СИНХРОНИЗАЦИИ (условные единицы)
                'E_sync_proton': 1.613565,   # ×100 = 161 МэВ
                'E_sync_neutron': 0.285395,  # ×100 = 28.5 МэВ
                'E_sync_pion': 4.273121,     # ×100 = 427 МэВ
                
                # ЭЛЕКТРОМАГНИТНЫЕ ПОПРАВКИ (уже в МэВ!)
                'EM_pi_charged': 0.0,        # МэВ - будет оптимизироваться
                'delta_pi0': -5.0,           # МэВ, разница π⁰-π⁺
                
                # МАСШТАБ
                'scale': 100.0
            }
        else:
            self.params = params
        
        # ЦЕЛЕВЫЕ МАССЫ (МэВ)
        self.targets = {
            'proton': 938.272,
            'neutron': 939.565,
            'pi+': 139.570,
            'pi0': 134.977,
            'pi-': 139.570,
        }
    
    def calculate_mass(self, particle):
        """Расчет массы частицы"""
        if particle == 'proton':
            # Протон: uud
            base = 2 * self.params['m_u_base'] + self.params['m_d_base']
            sync = self.params['E_sync_proton']
            return (base + sync) * self.params['scale']
        
        elif particle == 'neutron':
            # Нейтрон: udd
            base = self.params['m_u_base'] + 2 * self.params['m_d_base']
            sync = self.params['E_sync_neutron']
            return (base + sync) * self.params['scale']
        
        elif particle == 'pi+':
            # π⁺: u-anti-d
            base = self.params['m_u_base'] + self.params['m_d_base']
            sync = self.params['E_sync_pion']
            mass = (base - sync) * self.params['scale']  # Мезоны: МИНУС синхронизация!
            # Электромагнитная поправка (уже в МэВ)
            return mass + self.params['EM_pi_charged']
        
        elif particle == 'pi-':
            # π⁻: d-anti-u
            base = self.params['m_u_base'] + self.params['m_d_base']
            sync = self.params['E_sync_pion']
            mass = (base - sync) * self.params['scale']
            return mass + self.params['EM_pi_charged']
        
        elif particle == 'pi0':
            # π⁰: суперпозиция (u-anti-u + d-anti-d)/√2
            # Приближение: среднее между двумя заряженными состояниями минус поправка
            base = self.params['m_u_base'] + self.params['m_d_base']
            sync = self.params['E_sync_pion']
            mass = (base - sync) * self.params['scale']
            # Нейтральный пион легче заряженного
            return mass + self.params['delta_pi0']
        
        else:
            raise ValueError(f"Неизвестная частица: {particle}")

class OptimizerV210:
    """Оптимизатор для v21.2.0"""
    
    def __init__(self):
        self.model = ThreadModelV210()
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.result_dir = f"v21_2_0_results_{timestamp}"
        os.makedirs(self.result_dir, exist_ok=True)
        
        print("\n" + "="*80)
        print("v21.2.0: ИСПРАВЛЕННАЯ МОДЕЛЬ С ПРАВИЛЬНЫМИ ЕДИНИЦАМИ")
        print("="*80)
    
    def calculate_error(self, params):
        """Вычисление ошибки"""
        self.model.params.update(params)
        
        total_error = 0.0
        details = {}
        
        for particle, target_mass in self.model.targets.items():
            calculated = self.model.calculate_mass(particle)
            abs_error = abs(calculated - target_mass)
            rel_error = abs_error / target_mass * 100
            
            details[particle] = {
                'calculated': calculated,
                'target': target_mass,
                'abs_error': abs_error,
                'rel_error': rel_error
            }
            
            # Комбинированная ошибка
            total_error += abs_error + rel_error
        
        # Штраф за отрицательные массы
        if any(d['calculated'] <= 0 for d in details.values()):
            total_error += 10000
        
        # Штраф за нефизичное отношение масс кварков
        m_u_eff = params['m_u_base'] * 100
        m_d_eff = params['m_d_base'] * 100
        ratio = m_d_eff / m_u_eff if m_u_eff > 0 else 1.0
        
        if ratio < 1.3 or ratio > 2.2:
            total_error += abs(ratio - 1.6) * 100
        
        return total_error, details
    
    def run_simulated_annealing(self, iterations=200000):
        """Метод отжига для оптимизации"""
        print("\nЗапуск метода отжига...")
        
        # Начальные параметры (из v6.1)
        current_params = self.model.params.copy()
        current_error, current_details = self.calculate_error(current_params)
        
        best_params = current_params.copy()
        best_error = current_error
        best_details = current_details
        
        temperature = 1.0
        cooling_rate = 0.999995
        
        param_names = list(current_params.keys())
        
        # Диапазоны для параметров (±10% от v6.1 значений)
        ranges = {
            'm_u_base': (2.0, 2.4),
            'm_d_base': (4.2, 5.0),
            'E_sync_proton': (1.4, 1.8),
            'E_sync_neutron': (0.2, 0.35),
            'E_sync_pion': (4.0, 4.5),
            'EM_pi_charged': (-2.0, 2.0),     # МэВ
            'delta_pi0': (-10.0, 0.0),        # МэВ
            'scale': (99.0, 101.0)
        }
        
        for i in range(iterations):
            # Мутация
            new_params = current_params.copy()
            for param in param_names:
                if param in ranges:
                    min_val, max_val = ranges[param]
                    step = (max_val - min_val) * 0.1 * temperature
                    mutation = np.random.normal(0, step)
                    new_val = current_params[param] + mutation
                    
                    # Отражение от границ
                    while new_val < min_val or new_val > max_val:
                        if new_val < min_val:
                            new_val = 2 * min_val - new_val
                        if new_val > max_val:
                            new_val = 2 * max_val - new_val
                    
                    new_params[param] = new_val
            
            # Оценка
            new_error, new_details = self.calculate_error(new_params)
            
            # Критерий Метрополиса
            if new_error < current_error:
                current_params = new_params
                current_error = new_error
                current_details = new_details
            else:
                delta = new_error - current_error
                prob = np.exp(-delta / temperature)
                if np.random.random() < prob:
                    current_params = new_params
                    current_error = new_error
                    current_details = new_details
            
            # Обновление лучшего
            if new_error < best_error:
                best_params = new_params.copy()
                best_error = new_error
                best_details = new_details
            
            # Охлаждение
            temperature *= cooling_rate
            
            # Прогресс
            if i % 20000 == 0:
                print(f"Итерация {i:6d} | Ошибка: {best_error:.2f} | "
                      f"Протон: {best_details['proton']['calculated']:.1f} | "
                      f"Пион: {best_details['pi+']['calculated']:.1f} | "
                      f"Темп: {temperature:.4f}")
        
        return best_params, best_error, best_details
    
    def save_results(self, params, error, details):
        """Сохранение результатов"""
        results = {
            'model': 'v21.2.0',
            'timestamp': datetime.now().isoformat(),
            'total_error': error,
            'parameters': params,
            'predictions': details,
            'targets': self.model.targets
        }
        
        filename = os.path.join(self.result_dir, "results.json")
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2, default=self.json_serializer)
        
        self.save_text_report(results)
    
    def save_text_report(self, results):
        """Сохранение текстового отчёта"""
        filename = os.path.join(self.result_dir, "REPORT.txt")
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("v21.2.0 — ИСПРАВЛЕННАЯ МОДЕЛЬ С ПРАВИЛЬНЫМИ ЕДИНИЦАМИ\n")
            f.write("="*80 + "\n\n")
            
            f.write("ПАРАМЕТРЫ (условные единицы):\n")
            for key, value in results['parameters'].items():
                if key in ['m_u_base', 'm_d_base', 'E_sync_proton', 'E_sync_neutron', 'E_sync_pion']:
                    f.write(f"  {key}: {value:.6f} → {value*100:.1f} МэВ\n")
                else:
                    f.write(f"  {key}: {value:.6f}\n")
            
            f.write("\nПРЕДСКАЗАНИЯ:\n")
            f.write(f"{'Частица':<10} {'Расчёт':<12} {'Цель':<12} {'Ошибка':<12} {'%':<8}\n")
            f.write("-"*80 + "\n")
            
            total_error = 0
            for particle in ['proton', 'neutron', 'pi+', 'pi0', 'pi-']:
                det = results['predictions'][particle]
                f.write(f"{particle:<10} {det['calculated']:<12.3f} {det['target']:<12.3f} "
                       f"{det['abs_error']:<12.3f} {det['rel_error']:<8.3f}\n")
                total_error += det['rel_error']
            
            avg_error = total_error / 5
            f.write(f"\nСредняя ошибка: {avg_error:.3f}%\n")
            
            # Разность масс n-p
            diff = (results['predictions']['neutron']['calculated'] - 
                   results['predictions']['proton']['calculated'])
            f.write(f"Разность масс n-p: {diff:.3f} МэВ (цель 1.293 МэВ)\n")
            
            # Эффективные массы
            m_u_eff = results['parameters']['m_u_base'] * 100
            m_d_eff = results['parameters']['m_d_base'] * 100
            ratio = m_d_eff / m_u_eff
            f.write(f"\nЭффективные массы кварков:\n")
            f.write(f"  u: {m_u_eff:.1f} МэВ\n")
            f.write(f"  d: {m_d_eff:.1f} МэВ\n")
            f.write(f"  Отношение m_d/m_u: {ratio:.3f}\n")
            
            f.write("\n" + "="*80 + "\n")
    
    def json_serializer(self, obj):
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        return str(obj)

def quick_test():
    """Быстрый тест с параметрами v6.1"""
    print("\n" + "="*80)
    print("БЫСТРЫЙ ТЕСТ С ПАРАМЕТРАМИ v6.1")
    print("="*80)
    
    # Параметры из v6.1
    params_v61 = {
        'm_u_base': 2.203806,
        'm_d_base': 4.583020,
        'E_sync_proton': 1.613565,
        'E_sync_neutron': 0.285395,
        'E_sync_pion': 4.273121,
        'EM_pi_charged': 0.0,
        'delta_pi0': -4.593,  # Разница 139.57 - 134.977 = 4.593 МэВ
        'scale': 100.0
    }
    
    model = ThreadModelV210(params_v61)
    
    print("\nРасчёт с параметрами v6.1:")
    for particle in ['proton', 'neutron', 'pi+', 'pi0', 'pi-']:
        mass = model.calculate_mass(particle)
        target = model.targets[particle]
        error = abs(mass - target) / target * 100
        status = "✓" if error < 0.1 else "⚠" if error < 1.0 else "✗"
        print(f"{status} {particle:6}: {mass:8.3f} МэВ (цель {target:7.3f}) — {error:5.2f}%")
    
    # Разность масс
    diff = model.calculate_mass('neutron') - model.calculate_mass('proton')
    print(f"\nРазность масс n-p: {diff:.3f} МэВ (цель 1.293 МэВ)")
    
    return model

# ================= ЗАПУСК =================
if __name__ == "__main__":
    print("="*80)
    print("МОДЕЛЬ v21.2.0 — ИСПРАВЛЕНИЕ ЕДИНИЦ И ФОРМУЛ")
    print("="*80)
    
    # Шаг 1: Быстрый тест с параметрами v6.1
    model = quick_test()
    
    # Шаг 2: Оптимизация
    proceed = input("\nЗапустить оптимизацию методом отжига? (y/n): ")
    
    if proceed.lower() == 'y':
        optimizer = OptimizerV210()
        
        print("\n" + "="*80)
        print("ЗАПУСК МЕТОДА ОТЖИГА (200,000 итераций)")
        print("="*80)
        
        best_params, best_error, best_details = optimizer.run_simulated_annealing(
            iterations=200000
        )
        
        # Финальный отчёт
        print("\n" + "="*80)
        print("ОПТИМИЗАЦИЯ ЗАВЕРШЕНА!")
        print("="*80)
        
        print("\nЛУЧШИЕ ПАРАМЕТРЫ:")
        for key, value in best_params.items():
            print(f"  {key}: {value:.6f}")
        
        print("\nРЕЗУЛЬТАТЫ:")
        total_error = 0
        for particle in ['proton', 'neutron', 'pi+', 'pi0', 'pi-']:
            det = best_details[particle]
            print(f"  {particle:6}: {det['calculated']:8.3f} МэВ (цель {det['target']:7.3f}) — {det['rel_error']:5.2f}%")
            total_error += det['rel_error']
        
        avg_error = total_error / 5
        print(f"\nСредняя ошибка: {avg_error:.2f}%")
        
        # Сохранение
        optimizer.save_results(best_params, best_error, best_details)
        print(f"\nРезультаты сохранены в: {optimizer.result_dir}")
    
    else:
        print("\nТолько тестовый прогон завершён.")