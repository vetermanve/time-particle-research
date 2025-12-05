Iteration 0: error = 8.053988

Оптимизация завершена. Лучшая ошибка: 1.913884

================================================================================
ФИНАЛЬНЫЕ ПАРАМЕТРЫ
================================================================================

Базовые параметры (из v6.1):
  m_u_eff: 2.203806
  m_d_eff: 4.583020
  m_s_eff: 8.000000

Топологические параметры coupling:
  color_factor_baryon: 0.500000
  color_factor_meson: 1.442583
  phase_coherence_proton: 1.390273
  phase_coherence_neutron: 0.966283
  phase_coherence_meson: 1.550088
  spin_factor_proton: 0.520593
  spin_factor_neutron: 0.929889
  strangeness_factor: 1.128301
  eta_mixing: 0.378481

Расчётные coupling:
  proton: 0.542824
  neutron: 0.134780
  pi+: 4.695875
  pi0: 4.461081
  pi-: 4.695875
  K+: 6.307572
  K0: 6.307572
  eta: 1.637969
  Lambda0: 1.015471

МАССЫ ЧАСТИЦ:
Частица    Расчёт     Цель       Ошибка %
--------------------------------------------------
⚠ proton   953.346    938.272    1.607
✗ neutron  1150.463   939.565    22.446
✗ pi+      209.095    139.570    49.814
✗ pi0      -5.347     134.977    103.961
✗ pi-      209.095    139.570    49.814
✗ K+       389.623    493.677    21.077
✗ K0       627.545    497.611    26.112
⚠ eta      555.325    547.862    1.362
✗ Lambda0  1580.230   1115.683   41.638

Средняя ошибка: 35.31%


Вы абсолютно правы! Проблема в том, что мы не проводим полноценную оптимизацию. В v6.1 мы делали **3 миллиона итераций метода отжига**, а в v14.0 — всего 10,000 случайных итераций. Кроме того, у нас осталась ошибка: **π⁰ имеет отрицательную массу (-5.3 МэВ)**, что физически невозможно.

Давайте создадим **v15.0** с правильным методом отжига и учтём ключевые уроки:

## Анализ проблем v14.0:

1. **Нет отжига** — только случайный поиск
2. **Неправильные границы** параметров
3. **Нет штрафа за отрицательные массы**
4. **Отсутствие физических ограничений** (разность масс n-p, связь между coupling)

## v15.0 — Полноценная оптимизация с отжигом

```python
import numpy as np
import json
from datetime import datetime
import os
import sys

class TopologicalModelV15:
    """Полноценная топологическая модель с методом отжига"""
    
    def __init__(self):
        # Фиксируем scale = 100 как в v6.1
        self.scale = 100.0
        
        # Целевые массы (МэВ) - расширенный набор
        self.targets = {
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
        
        # Состав частиц
        self.composition = {
            'proton': ['u', 'u', 'd'],
            'neutron': ['u', 'd', 'd'],
            'pi+': ['u', 'anti_d'],
            'pi0': ['u', 'anti_u'],
            'pi-': ['d', 'anti_u'],
            'K+': ['u', 'anti_s'],
            'K0': ['d', 'anti_s'],
            'K-': ['s', 'anti_u'],
            'eta': ['mixed'],  # Особый случай
            'Lambda0': ['u', 'd', 's'],
        }
        
        # Инициализация параметров (будем оптимизировать все)
        self.params = {
            # Эффективные массы (как в v6.1)
            'm_u_eff': 2.203806,
            'm_d_eff': 4.583020,
            'm_s_eff': 13.0,
            
            # Базовые coupling для типов частиц
            'C_proton': 1.613565,    # Из v6.1
            'C_neutron': 0.285395,   # Из v6.1
            'C_pion': 4.273121,      # Из v6.1
            'C_kaon': 4.5,           # Подбираем
            'C_eta': 4.0,            # Подбираем
            'C_lambda': 2.0,         # Подбираем
            
            # Поправочные коэффициенты
            'k_color_meson': 1.0,    # Цветовой фактор для мезонов
            'k_phase_neutron': 0.18, # Фазовый фактор нейтрона/протона
            'k_strange': 0.7,        # Фактор странности
            'k_mixing_pi0': 0.95,    # Смешивание для π⁰
            'k_mixing_eta': 0.4,     # Смешивание для η
        }
    
    def calculate_base_mass(self, particle):
        """Базовая масса из эффективных масс"""
        comp = self.composition[particle]
        total = 0
        
        if particle == 'eta':
            # η: (uū + dđ + sš)/√3 упрощённо
            return (2*self.params['m_u_eff'] + 
                    2*self.params['m_d_eff'] + 
                    self.params['m_s_eff']) / 3.0
        
        for quark in comp:
            if quark in ['u', 'anti_u']:
                total += self.params['m_u_eff']
            elif quark in ['d', 'anti_d']:
                total += self.params['m_d_eff']
            elif quark in ['s', 'anti_s']:
                total += self.params['m_s_eff']
        
        return total
    
    def calculate_coupling(self, particle):
        """Вычисление coupling с поправками"""
        if particle == 'proton':
            return self.params['C_proton']
        
        elif particle == 'neutron':
            # coupling нейтрона = coupling протона * k_phase
            return self.params['C_proton'] * self.params['k_phase_neutron']
        
        elif particle in ['pi+', 'pi-']:
            return self.params['C_pion'] * self.params['k_color_meson']
        
        elif particle == 'pi0':
            # π⁰ немного отличается из-за смешивания
            return self.params['C_pion'] * self.params['k_color_meson'] * self.params['k_mixing_pi0']
        
        elif particle in ['K+', 'K0', 'K-']:
            # Странные мезоны
            return self.params['C_kaon'] * self.params['k_color_meson'] * self.params['k_strange']
        
        elif particle == 'eta':
            # η-мезон со сложным смешиванием
            return self.params['C_eta'] * self.params['k_color_meson'] * self.params['k_mixing_eta']
        
        elif particle == 'Lambda0':
            # Λ⁰ барион с s-кварком
            return self.params['C_lambda'] * self.params['k_strange']
        
        else:
            return 1.0
    
    def calculate_mass(self, particle):
        """Основная формула: M = (base ± coupling) * scale"""
        base = self.calculate_base_mass(particle)
        coupling = self.calculate_coupling(particle)
        
        # Барионы: +coupling, мезоны: -coupling
        if particle in ['proton', 'neutron', 'Lambda0']:
            mass = (base + coupling) * self.scale
        else:
            mass = (base - coupling) * self.scale
        
        return mass
    
    def error_function(self, params_array):
        """Функция ошибки с штрафами"""
        # Обновляем параметры из массива
        param_names = [
            'm_u_eff', 'm_d_eff', 'm_s_eff',
            'C_proton', 'C_neutron', 'C_pion', 'C_kaon', 'C_eta', 'C_lambda',
            'k_color_meson', 'k_phase_neutron', 'k_strange', 'k_mixing_pi0', 'k_mixing_eta'
        ]
        
        for i, name in enumerate(param_names):
            self.params[name] = params_array[i]
        
        total_error = 0.0
        mass_errors = []
        
        # Вычисляем массы и ошибки
        for particle, target in self.targets.items():
            mass = self.calculate_mass(particle)
            
            # Жёсткий штраф за отрицательные массы
            if mass <= 0:
                total_error += 10000.0
                mass_errors.append(10000.0)
                continue
            
            # Квадратичная относительная ошибка
            rel_error = (mass - target) / target
            mass_errors.append(rel_error ** 2)
            total_error += rel_error ** 2
        
        # Дополнительные физические штрафы
        
        # 1. coupling нейтрона должен быть меньше coupling протона
        if self.params['C_neutron'] >= self.params['C_proton']:
            total_error += 100.0 * (self.params['C_neutron'] - self.params['C_proton'])
        
        # 2. Массы кварков должны быть в порядке: m_s > m_d > m_u
        if self.params['m_s_eff'] <= self.params['m_d_eff']:
            total_error += 50.0 * (self.params['m_d_eff'] - self.params['m_s_eff'] + 0.1)
        
        if self.params['m_d_eff'] <= self.params['m_u_eff']:
            total_error += 50.0 * (self.params['m_u_eff'] - self.params['m_d_eff'] + 0.1)
        
        # 3. Разность масс n-p должна быть положительной ~1.293 МэВ
        mass_n = self.calculate_mass('neutron')
        mass_p = self.calculate_mass('proton')
        diff_np = mass_n - mass_p
        target_diff = 1.293
        diff_error = abs(diff_np - target_diff) / target_diff
        total_error += 10.0 * diff_error
        
        # 4. Массы π⁺, π⁰, π⁻ должны быть близки
        mass_pi_plus = self.calculate_mass('pi+')
        mass_pi_zero = self.calculate_mass('pi0')
        pi_diff_error = abs(mass_pi_plus - mass_pi_zero) / 4.6  # Ожидаемая разница ~4.6 МэВ
        if pi_diff_error > 2.0:  # Если разница слишком большая
            total_error += 5.0 * pi_diff_error
        
        return total_error
    
    def run_annealing(self, iterations=3000000, initial_temp=5.0, 
                     cooling_rate=0.999997, save_interval=200000):
        """Метод отжига (имитации закалки) как в v6.1"""
        print("="*80)
        print("ЗАПУСК МЕТОДА ОТЖИГА v15.0")
        print(f"Итераций: {iterations:,}")
        print("="*80)
        
        import time
        import random
        
        # Инициализация параметров
        param_names = [
            'm_u_eff', 'm_d_eff', 'm_s_eff',
            'C_proton', 'C_neutron', 'C_pion', 'C_kaon', 'C_eta', 'C_lambda',
            'k_color_meson', 'k_phase_neutron', 'k_strange', 'k_mixing_pi0', 'k_mixing_eta'
        ]
        
        # Начальная точка: параметры из v6.1 + разумные значения для новых
        current_params = np.array([
            # Эффективные массы
            2.203806,  # m_u_eff из v6.1
            4.583020,  # m_d_eff из v6.1
            13.0,      # m_s_eff (оценка)
            
            # Базовые coupling
            1.613565,  # C_proton из v6.1
            0.285395,  # C_neutron из v6.1
            4.273121,  # C_pion из v6.1
            4.5,       # C_kaon (оценка)
            4.0,       # C_eta (оценка)
            2.0,       # C_lambda (оценка)
            
            # Поправочные коэффициенты
            1.0,       # k_color_meson
            0.18,      # k_phase_neutron (0.285/1.614 ≈ 0.177)
            0.7,       # k_strange
            0.95,      # k_mixing_pi0
            0.4,       # k_mixing_eta
        ])
        
        # Границы параметров
        bounds = [
            # m_u_eff, m_d_eff, m_s_eff
            (1.5, 3.0), (3.0, 6.0), (8.0, 20.0),
            # C_proton, C_neutron, C_pion, C_kaon, C_eta, C_lambda
            (1.0, 2.5), (0.1, 0.8), (3.0, 6.0), (3.0, 6.0), (3.0, 6.0), (1.0, 3.0),
            # k_color_meson, k_phase_neutron, k_strange, k_mixing_pi0, k_mixing_eta
            (0.5, 2.0), (0.05, 0.3), (0.3, 1.2), (0.8, 1.2), (0.1, 1.0)
        ]
        
        # Текущая ошибка
        current_error = self.error_function(current_params)
        best_params = current_params.copy()
        best_error = current_error
        
        temperature = initial_temp
        start_time = time.time()
        
        # Статистика
        stats = {'accepts': 0, 'improves': 0, 'rejects': 0}
        
        for i in range(1, iterations + 1):
            # Генерация нового решения
            new_params = current_params.copy()
            
            for j in range(len(new_params)):
                if random.random() < 0.3:  # Вероятность мутации каждого параметра
                    min_val, max_val = bounds[j]
                    range_width = max_val - min_val
                    
                    # Адаптивный шаг мутации
                    mutation = np.random.normal(0, range_width * 0.05 * temperature)
                    new_val = current_params[j] + mutation
                    
                    # Отражение от границ
                    while new_val < min_val or new_val > max_val:
                        if new_val < min_val:
                            new_val = 2 * min_val - new_val
                        if new_val > max_val:
                            new_val = 2 * max_val - new_val
                    
                    new_params[j] = new_val
            
            # Оценка нового решения
            new_error = self.error_function(new_params)
            
            # Критерий Метрополиса
            delta_error = new_error - current_error
            
            if delta_error < 0:
                # Улучшение - всегда принимаем
                current_params = new_params
                current_error = new_error
                stats['accepts'] += 1
                stats['improves'] += 1
            else:
                # Ухудшение - принимаем с вероятностью exp(-ΔE/T)
                prob = np.exp(-delta_error / temperature)
                if random.random() < prob:
                    current_params = new_params
                    current_error = new_error
                    stats['accepts'] += 1
                else:
                    stats['rejects'] += 1
            
            # Обновление лучшего решения
            if new_error < best_error:
                best_params = new_params.copy()
                best_error = new_error
            
            # Охлаждение
            temperature *= cooling_rate
            
            # Вывод прогресса
            if i % 100000 == 0:
                elapsed = time.time() - start_time
                progress = i / iterations * 100
                
                # Вычисляем некоторые массы для мониторинга
                self.error_function(best_params)  # Обновляем параметры
                mass_p = self.calculate_mass('proton')
                mass_n = self.calculate_mass('neutron')
                mass_pi = self.calculate_mass('pi+')
                
                print(f"\rИтерация {i:,}/{iterations:,} ({progress:.1f}%) | "
                      f"Ошибка: {best_error:.4f} | "
                      f"Темп: {temperature:.4f} | "
                      f"p={mass_p:.0f}, n={mass_n:.0f}, π={mass_pi:.0f} | "
                      f"Время: {elapsed:.0f}с", end='', flush=True)
            
            # Сохранение контрольной точки
            if i % save_interval == 0 and i > 0:
                self.save_checkpoint(i, best_params, best_error)
        
        # Финальный отчёт
        elapsed = time.time() - start_time
        print(f"\n\n{'='*80}")
        print("ОТЖИГ ЗАВЕРШЁН")
        print(f"Всего итераций: {iterations:,}")
        print(f"Время выполнения: {elapsed:.1f} сек")
        print(f"Лучшая ошибка: {best_error:.6f}")
        print(f"Принято решений: {stats['accepts']:,}")
        print(f"Улучшений: {stats['improves']:,}")
        print(f"Отклонено: {stats['rejects']:,}")
        
        return best_params, best_error
    
    def save_checkpoint(self, iteration, params, error):
        """Сохранение контрольной точки"""
        checkpoint = {
            'iteration': iteration,
            'error': error,
            'params': params.tolist(),
            'timestamp': datetime.now().isoformat()
        }
        
        # Создаём директорию для результатов
        if not hasattr(self, 'results_dir'):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.results_dir = f"annealing_v15_{timestamp}"
            os.makedirs(self.results_dir, exist_ok=True)
        
        filename = f"{self.results_dir}/checkpoint_{iteration:08d}.json"
        with open(filename, 'w') as f:
            json.dump(checkpoint, f, indent=2)
    
    def evaluate_final(self, best_params):
        """Оценка финальных параметров"""
        # Обновляем параметры модели
        param_names = [
            'm_u_eff', 'm_d_eff', 'm_s_eff',
            'C_proton', 'C_neutron', 'C_pion', 'C_kaon', 'C_eta', 'C_lambda',
            'k_color_meson', 'k_phase_neutron', 'k_strange', 'k_mixing_pi0', 'k_mixing_eta'
        ]
        
        for i, name in enumerate(param_names):
            self.params[name] = best_params[i]
        
        # Вычисляем все массы
        results = {}
        for particle in self.targets:
            results[particle] = self.calculate_mass(particle)
        
        return results

def main():
    """Основная функция"""
    print("="*80)
    print("ТОПОЛОГИЧЕСКАЯ МОДЕЛЬ АДРОНОВ v15.0")
    print("МЕТОД ОТЖИГА С ФИЗИЧЕСКИМИ ОГРАНИЧЕНИЯМИ")
    print("="*80)
    
    # Создаём модель
    model = TopologicalModelV15()
    
    # Запускаем отжиг
    try:
        best_params, best_error = model.run_annealing(
            iterations=1000000,  # 1 миллион итераций для начала
            initial_temp=5.0,
            cooling_rate=0.999995,
            save_interval=100000
        )
        
        # Оценка финальных результатов
        final_masses = model.evaluate_final(best_params)
        
        # Вывод результатов
        print("\n" + "="*80)
        print("ФИНАЛЬНЫЕ РЕЗУЛЬТАТЫ")
        print("="*80)
        
        # Параметры
        param_names = [
            'm_u_eff', 'm_d_eff', 'm_s_eff',
            'C_proton', 'C_neutron', 'C_pion', 'C_kaon', 'C_eta', 'C_lambda',
            'k_color_meson', 'k_phase_neutron', 'k_strange', 'k_mixing_pi0', 'k_mixing_eta'
        ]
        
        print("\nОПТИМАЛЬНЫЕ ПАРАМЕТРЫ:")
        for i, name in enumerate(param_names):
            print(f"  {name}: {best_params[i]:.6f}")
        
        # Эффективные массы в МэВ
        m_u_mev = model.params['m_u_eff'] * model.scale
        m_d_mev = model.params['m_d_eff'] * model.scale
        m_s_mev = model.params['m_s_eff'] * model.scale
        
        print(f"\nЭФФЕКТИВНЫЕ МАССЫ КВАРКОВ (в МэВ):")
        print(f"  u: {m_u_mev:.1f} МэВ")
        print(f"  d: {m_d_mev:.1f} МэВ")
        print(f"  s: {m_s_mev:.1f} МэВ")
        print(f"  Отношение m_d/m_u: {m_d_mev/m_u_mev:.3f}")
        print(f"  Отношение m_s/m_u: {m_s_mev/m_u_mev:.3f}")
        
        # Массы частиц
        print(f"\nМАССЫ ЧАСТИЦ:")
        print(f"{'Частица':<10} {'Расчёт':<10} {'Цель':<10} {'Ошибка %':<10}")
        print("-"*50)
        
        total_error = 0
        for particle, target in model.targets.items():
            mass = final_masses[particle]
            error_pct = abs(mass - target) / target * 100
            total_error += error_pct
            
            status = "✓" if error_pct < 1.0 else "⚠" if error_pct < 5.0 else "✗"
            print(f"{status} {particle:<8} {mass:<10.3f} {target:<10.3f} {error_pct:<10.3f}")
        
        avg_error = total_error / len(model.targets)
        print(f"\nСредняя ошибка: {avg_error:.2f}%")
        
        # Ключевые физические проверки
        print(f"\nФИЗИЧЕСКИЕ ПРОВЕРКИ:")
        
        # 1. Разность масс n-p
        diff_np = final_masses['neutron'] - final_masses['proton']
        print(f"  Разность масс n-p: {diff_np:.3f} МэВ (цель 1.293 МэВ)")
        
        # 2. coupling нейтрона / протона
        ratio_coupling = model.params['C_neutron'] / model.params['C_proton']
        print(f"  Отношение coupling нейтрон/протон: {ratio_coupling:.3f}")
        
        # 3. Массы пионов
        diff_pi = final_masses['pi+'] - final_masses['pi0']
        print(f"  Разность масс π⁺-π⁰: {diff_pi:.3f} МэВ (ожидается ~4.6 МэВ)")
        
        # Сохранение полных результатов
        results_data = {
            'model': 'v15.0_topological_annealing',
            'timestamp': datetime.now().isoformat(),
            'error': float(best_error),
            'parameters': {name: float(best_params[i]) for i, name in enumerate(param_names)},
            'masses': {p: float(final_masses[p]) for p in final_masses},
            'quark_masses_mev': {
                'u': float(m_u_mev),
                'd': float(m_d_mev),
                's': float(m_s_mev)
            }
        }
        
        results_dir = model.results_dir if hasattr(model, 'results_dir') else 'results_v15'
        final_file = f"{results_dir}/final_results.json"
        
        with open(final_file, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        print(f"\nПолные результаты сохранены в: {final_file}")
        
    except KeyboardInterrupt:
        print("\n\nОптимизация прервана пользователем.")
    except Exception as e:
        print(f"\nОшибка при выполнении: {e}")
        import traceback
        traceback.print_exc()
    
    print("="*80)

if __name__ == "__main__":
    main()
```

## Ключевые улучшения v15.0:

### 1. **Настоящий метод отжига (имитации закалки):**
- 1-3 миллиона итераций (как в v6.1)
- Температурный режим с охлаждением
- Критерий Метрополиса для принятия решений
- Адаптивный шаг мутации

### 2. **Физические ограничения и штрафы:**
- **Запрет отрицательных масс** (штраф 10,000)
- **Правильный порядок масс кварков:** m_s > m_d > m_u
- **Разность масс n-p ≈ 1.293 МэВ**
- **coupling нейтрона < coupling протона** (как в v6.1)

### 3. **Структура параметров:**
- **Эффективные массы:** m_u_eff, m_d_eff, m_s_eff
- **Базовые coupling:** отдельно для каждого типа частиц
- **Поправочные коэффициенты:** цвет, фазы, странность, смешивание

### 4. **Мониторинг прогресса:**
- Вывод каждые 100,000 итераций
- Контрольные точки сохранения
- Отслеживание ключевых масс (p, n, π)

## Ожидаемое поведение:

1. **Первые 100,000 итераций:** Быстрый поиск в широком диапазоне
2. **500,000-1,000,000 итераций:** Тонкая настройка параметров
3. **Финал:** Ошибка должна уменьшиться до <10%, а для p, n, π⁺ — до <1%

## Что мы проверяем:

✅ **Воспроизводимость v6.1:** Можем ли мы получить те же результаты для p, n, π⁺?  
✅ **Расширяемость:** Работает ли модель для K⁺, η, Λ⁰?  
✅ **Физическая осмысленность:** Имеют ли найденные параметры физический смысл?

## Если v15.0 сработает хорошо, следующий шаг:

1. **Добавить больше частиц** (Σ, Ξ, Ω, D-мезоны)
2. **Ввести матрицу смешивания** для нейтральных мезонов
3. **Связать параметры с топологией узлов** (полиномы Джонса)
4. **Вывести coupling из первых принципов** (цвет, спин, изоспин)

**Давайте запустим v15.0 и посмотрим, сможем ли мы достичь точности v6.1 для всего набора частиц!** Отжиг займёт время (минуты или часы), но это необходимо для серьёзной оптимизации.