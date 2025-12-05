# Исправленная модель v13.1 - Исправляем ошибки и упрощаем подход

Ошибка в индексации связана с тем, что мы пытаемся обновить 7 параметров из массива из 6 элементов. Давайте исправим это и упростим модель, чтобы она работала стабильно.

```python
import numpy as np
from scipy.optimize import minimize
import json
from datetime import datetime
import os
import sys

class SimplifiedTopologicalModel:
    """Упрощённая топологическая модель с фиксированными узлами"""
    
    def __init__(self):
        # Целевые массы (МэВ) - ограничимся ключевыми частицами
        self.target_masses = {
            'proton': 938.272,
            'neutron': 939.565,
            'pi+': 139.570,
            'pi0': 134.9768,
            'K+': 493.677,
            'eta': 547.862,
            'Lambda0': 1115.683,
        }
        
        # Топологические сложности узлов (гипотеза)
        # Число = сложность узла * 100
        self.knot_complexity = {
            'proton': 2.0,    # Трилистник (сложный)
            'neutron': 1.5,   # Восьмёрка (менее сложный)
            'pi+': 1.0,       # Зацепление Хопфа
            'pi0': 1.0,       # Зацепление Хопфа, но с другим смешиванием
            'K+': 1.0,        # Зацепление Хопфа с возбуждением
            'eta': 0.5,       # Тривиальный узел, но сложное смешивание
            'Lambda0': 2.0,   # Трилистник с возбуждением
        }
        
        # Начальные параметры
        self.params = {
            'm_u': 2.2,      # Эффективная масса u
            'm_d': 3.5,      # Эффективная масса d  
            'm_s': 10.0,     # Эффективная масса s
            'alpha': 1.0,    # Коэффициент топологической энергии
            'beta': 0.2,     # Коэффициент для странных частиц
            'gamma': 0.05,   # Электромагнитная поправка
            'mix_eta': 4.0,  # Параметр смешивания для η
            'scale': 100.0,
        }
        
        # Состав частиц
        self.composition = {
            'proton': ['u', 'u', 'd'],
            'neutron': ['u', 'd', 'd'],
            'pi+': ['u', 'anti_d'],
            'pi0': ['u', 'anti_u'],  # Упрощённо
            'K+': ['u', 'anti_s'],
            'eta': ['mixed'],  # Специальная обработка
            'Lambda0': ['u', 'd', 's'],
        }
    
    def effective_mass(self, quark):
        """Эффективная масса кварка"""
        if quark in ['u', 'anti_u']:
            return self.params['m_u']
        elif quark in ['d', 'anti_d']:
            return self.params['m_d']
        elif quark in ['s', 'anti_s']:
            return self.params['m_s']
        else:
            return 0.0
    
    def calculate_base_mass(self, particle):
        """Базовая масса из состава"""
        if particle == 'eta':
            # Для η особая формула: смесь u, d, s
            return (2*self.params['m_u'] + 2*self.params['m_d'] + 
                    self.params['m_s']) / 2.0
        
        comp = self.composition[particle]
        return sum(self.effective_mass(q) for q in comp)
    
    def calculate_topological_energy(self, particle):
        """Топологическая энергия связи"""
        complexity = self.knot_complexity[particle]
        
        # Базовая топологическая энергия
        energy = complexity * self.params['alpha']
        
        # Поправка для странных частиц
        if particle in ['K+', 'Lambda0']:
            energy *= (1 + self.params['beta'])
        
        # Для η добавляем параметр смешивания
        if particle == 'eta':
            energy = self.params['mix_eta']
        
        return energy
    
    def calculate_mass(self, particle):
        """Расчёт массы частицы"""
        base_mass = self.calculate_base_mass(particle)
        topo_energy = self.calculate_topological_energy(particle)
        
        # Для мезонов энергия вычитается, для барионов - прибавляется
        if particle in ['proton', 'neutron', 'Lambda0']:
            mass = (base_mass + topo_energy) * self.params['scale']
        else:
            mass = (base_mass - topo_energy) * self.params['scale']
        
        # Электромагнитная поправка для заряженных частиц
        if particle in ['proton', 'pi+', 'K+']:
            mass += self.params['gamma'] * 100.0  # Масштабируем
        
        return max(mass, 1.0)
    
    def calculate_all_masses(self):
        """Расчёт всех масс"""
        results = {}
        for particle in self.target_masses:
            results[particle] = self.calculate_mass(particle)
        return results
    
    def error_function(self, params_array):
        """Функция ошибки для оптимизации"""
        # Параметры для оптимизации в порядке:
        # m_u, m_d, m_s, alpha, beta, gamma, mix_eta
        param_names = ['m_u', 'm_d', 'm_s', 'alpha', 'beta', 'gamma', 'mix_eta']
        
        # Обновляем только оптимизируемые параметры
        for i, name in enumerate(param_names):
            self.params[name] = params_array[i]
        
        total_error = 0.0
        
        # Рассчитываем массы и накапливаем ошибку
        masses = self.calculate_all_masses()
        
        for particle, target in self.target_masses.items():
            calculated = masses[particle]
            
            # Квадратичная ошибка с весом
            error = (calculated - target) / target
            total_error += error ** 2
            
            # Дополнительный штраф за отрицательные массы
            if calculated <= 0:
                total_error += 100.0
        
        # Физические ограничения
        if self.params['m_d'] <= self.params['m_u']:
            total_error += 10.0
        
        if self.params['m_s'] <= self.params['m_d']:
            total_error += 10.0
        
        # Штраф за нереалистичные параметры
        if self.params['alpha'] < 0:
            total_error += 50.0
        
        return total_error
    
    def run_optimization(self, method='Nelder-Mead', max_iter=10000):
        """Запуск оптимизации"""
        print(f"\nЗапуск оптимизации методом {method}...")
        
        # Начальные значения
        initial_guess = [
            self.params['m_u'],
            self.params['m_d'], 
            self.params['m_s'],
            self.params['alpha'],
            self.params['beta'],
            self.params['gamma'],
            self.params['mix_eta']
        ]
        
        # Границы параметров
        bounds = [
            (1.5, 3.5),    # m_u
            (2.5, 5.0),    # m_d
            (8.0, 15.0),   # m_s
            (0.5, 3.0),    # alpha
            (0.0, 1.0),    # beta
            (0.0, 0.2),    # gamma
            (2.0, 6.0),    # mix_eta
        ]
        
        # Запускаем оптимизацию
        result = minimize(
            self.error_function,
            initial_guess,
            method=method,
            bounds=bounds,
            options={
                'maxiter': max_iter,
                'disp': True,
                'xatol': 1e-8,
                'fatol': 1e-8
            }
        )
        
        if result.success:
            print(f"Оптимизация успешна за {result.nit} итераций")
            print(f"Финальная ошибка: {result.fun:.6f}")
        else:
            print(f"Оптимизация завершилась: {result.message}")
        
        return result
    
    def print_results(self, final_params=None):
        """Вывод результатов"""
        if final_params is not None:
            # Обновляем параметры финальными значениями
            param_names = ['m_u', 'm_d', 'm_s', 'alpha', 'beta', 'gamma', 'mix_eta']
            for i, name in enumerate(param_names):
                self.params[name] = final_params[i]
        
        masses = self.calculate_all_masses()
        
        print("\n" + "="*80)
        print("РЕЗУЛЬТАТЫ МОДЕЛИ")
        print("="*80)
        
        print("\nПАРАМЕТРЫ:")
        for name, value in self.params.items():
            print(f"  {name}: {value:.6f}")
        
        # Рассчитываем эффективные массы в МэВ
        m_u_mev = self.params['m_u'] * self.params['scale']
        m_d_mev = self.params['m_d'] * self.params['scale']
        m_s_mev = self.params['m_s'] * self.params['scale']
        
        print(f"\nЭФФЕКТИВНЫЕ МАССЫ КВАРКОВ:")
        print(f"  u: {m_u_mev:.1f} МэВ")
        print(f"  d: {m_d_mev:.1f} МэВ")
        print(f"  s: {m_s_mev:.1f} МэВ")
        print(f"  Отношение m_d/m_u: {m_d_mev/m_u_mev:.3f}")
        print(f"  Отношение m_s/m_u: {m_s_mev/m_u_mev:.3f}")
        
        print("\nМАССЫ ЧАСТИЦ:")
        print(f"{'Частица':<10} {'Расчёт':<10} {'Цель':<10} {'Ошибка %':<10}")
        print("-"*50)
        
        total_error = 0
        for particle in self.target_masses:
            target = self.target_masses[particle]
            calculated = masses[particle]
            error_pct = abs(calculated - target) / target * 100
            total_error += error_pct
            
            status = "✓" if error_pct < 1.0 else "⚠" if error_pct < 5.0 else "✗"
            print(f"{status} {particle:<8} {calculated:<10.3f} {target:<10.3f} {error_pct:<10.3f}")
        
        avg_error = total_error / len(self.target_masses)
        print(f"\nСредняя ошибка: {avg_error:.2f}%")
        
        # Специальные проверки
        print(f"\nФИЗИЧЕСКИЕ ПРОВЕРКИ:")
        print(f"  Разность масс n-p: {masses['neutron'] - masses['proton']:.3f} МэВ (цель 1.293 МэВ)")
        print(f"  Разность масс π⁺-π⁰: {masses['pi+'] - masses['pi0']:.3f} МэВ (ожидается ~4.6 МэВ)")
        
        return masses

def main():
    print("="*80)
    print("УПРОЩЁННАЯ ТОПОЛОГИЧЕСКАЯ МОДЕЛЬ v13.1")
    print("="*80)
    
    # Создаём модель
    model = SimplifiedTopologicalModel()
    
    print("\nНачальные параметры:")
    for name, value in model.params.items():
        if name != 'scale':
            print(f"  {name}: {value}")
    
    print("\nНачальные оценки масс:")
    initial_masses = model.calculate_all_masses()
    for particle in ['proton', 'neutron', 'pi+', 'pi0', 'K+', 'eta', 'Lambda0']:
        target = model.target_masses[particle]
        calc = initial_masses[particle]
        err = abs(calc - target) / target * 100
        print(f"  {particle}: {calc:.1f} МэВ (цель {target:.1f}) - ошибка {err:.1f}%")
    
    # Запускаем оптимизацию
    try:
        result = model.run_optimization(method='Nelder-Mead', max_iter=5000)
        
        if result.success:
            print("\n" + "="*80)
            print("ОПТИМИЗАЦИЯ УСПЕШНА!")
            print("="*80)
            
            # Выводим финальные результаты
            final_masses = model.print_results(result.x)
            
            # Сохраняем результаты
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_dir = f"simplified_model_v131_{timestamp}"
            os.makedirs(results_dir, exist_ok=True)
            
            results_data = {
                'model': 'v13.1_simplified_topological',
                'timestamp': datetime.now().isoformat(),
                'parameters': model.params,
                'masses': final_masses,
                'optimization_result': {
                    'success': result.success,
                    'message': result.message,
                    'iterations': result.nit,
                    'final_error': result.fun
                }
            }
            
            with open(f"{results_dir}/results.json", 'w') as f:
                json.dump(results_data, f, indent=2, default=str)
            
            print(f"\nРезультаты сохранены в директории: {results_dir}")
            
            # Создаём текстовый отчёт
            report_file = f"{results_dir}/report.txt"
            with open(report_file, 'w') as f:
                f.write("="*80 + "\n")
                f.write("ОТЧЁТ ПО МОДЕЛИ v13.1\n")
                f.write("="*80 + "\n\n")
                
                f.write("ФИНАЛЬНЫЕ ПАРАМЕТРЫ:\n")
                for name, value in model.params.items():
                    f.write(f"  {name}: {value:.6f}\n")
                
                f.write("\nМАССЫ ЧАСТИЦ:\n")
                for particle in model.target_masses:
                    target = model.target_masses[particle]
                    calc = final_masses[particle]
                    err = abs(calc - target) / target * 100
                    f.write(f"  {particle}: {calc:.3f} МэВ (цель {target:.3f}) - {err:.3f}%\n")
            
            print(f"Текстовый отчёт создан: {report_file}")
            
        else:
            print("\nОптимизация не удалась. Пробуем альтернативный метод...")
            
            # Пробуем другой метод оптимизации
            result2 = model.run_optimization(method='Powell', max_iter=3000)
            
            if result2.success:
                final_masses = model.print_results(result2.x)
            else:
                print("\nОба метода оптимизации не удались.")
                print("Используем начальные параметры для анализа...")
                model.print_results()
                
    except Exception as e:
        print(f"\nОшибка при оптимизации: {e}")
        print("Выводим начальные результаты...")
        model.print_results()
    
    print("\n" + "="*80)
    print("АНАЛИЗ МОДЕЛИ")
    print("="*80)
    
    print("\nТОПОЛОГИЧЕСКИЕ ИНВАРИАНТЫ:")
    for particle, complexity in model.knot_complexity.items():
        print(f"  {particle}: сложность узла = {complexity}")
    
    print("\nФИЗИЧЕСКАЯ ИНТЕРПРЕТАЦИЯ:")
    print("  1. Каждый адрон соответствует определённому топологическому узлу")
    print("  2. Сложность узла определяет энергию синхронизации")
    print("  3. s-кварк представляет собой возбуждённую нить")
    print("  4. Электромагнитные поправки учитываются отдельно")
    
    print("\nСЛЕДУЮЩИЕ ШАГИ:")
    print("  1. Уточнить топологические инварианты для каждой частицы")
    print("  2. Добавить больше частиц (Ξ, Ω, Σ)")
    print("  3. Ввести матрицу смешивания для нейтральных мезонов")
    print("  4. Связать модель с концепцией нитей времени")
    
    print("="*80)

if __name__ == "__main__":
    main()
```

## Ключевые изменения в v13.1:

### 1. **Упрощение модели:**
- Вместо полиномов Джонса используем простые вещественные числа для сложности узлов
- Фиксированные топологические инварианты для каждой частицы

### 2. **Стабильная оптимизация:**
- Исправлена ошибка индексации
- Используем метод Нелдера-Мида (более устойчивый)
- Добавлены разумные границы параметров

### 3. **Физически осмысленные параметры:**
- `alpha` - общий масштаб топологической энергии
- `beta` - параметр возбуждения для странных частиц
- `gamma` - электромагнитные поправки
- `mix_eta` - специальный параметр для η-мезона

### 4. **Ограничение набора частиц:**
- Начинаем с ключевых 7 частиц для отладки
- Постепенно будем расширять

## Физическая интерпретация:

### Для протона (узел трилистник):
- Сложность = 2.0
- Топологическая энергия = 2.0 × alpha
- Базовая масса = 2×m_u + m_d
- Итог: (база + топология) × 100

### Для пиона (зацепление Хопфа):
- Сложность = 1.0  
- Топологическая энергия = 1.0 × alpha
- Базовая масса = m_u + m_d
- Итог: (база - топология) × 100 (знак минус для мезонов)

### Особенности:
1. **Нейтрон** чуть менее сложный узел, чем протон
2. **Странные частицы** имеют дополнительное возбуждение (×1+beta)
3. **η-мезон** имеет особую обработку из-за смешивания состояний

## Ожидаемые результаты:

1. **Эффективные массы кварков** в диапазоне:
   - u: ~200-300 МэВ
   - d: ~300-400 МэВ  
   - s: ~400-600 МэВ

2. **Топологическая энергия** alpha ≈ 1.0-2.0

3. **Точность модели** ≈ 5-10% для начальной версии

## Дальнейшее развитие:

После того как базовая модель заработает, мы можем:
1. Ввести **матрицу смешивания** для π⁰, η, η'
2. Добавить **спиновые поправки**
3. Связать сложность узлов с **цветовыми степенями свободы**
4. Расширить на **тяжёлые кварки** (c, b, t)

Эта модель сохраняет дух нашей оригинальной идеи о нитях времени, но реализует её через конкретную математику топологических инвариантов.

**Запускаем v13.1 и смотрим на результаты!**