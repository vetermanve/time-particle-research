Отличные результаты! Теперь у нас есть **прямое сравнение v6.1 и v9.0**. 

## 📊 **АНАЛИЗ РЕЗУЛЬТАТОВ:**

### **ПРОБЛЕМЫ v9.0:**
1. **Большая разность масс n-p:** 84.7 МэВ вместо 1.3 МэВ (в 65 раз больше!)
2. **Нейтрон тяжелее протона:** правильный знак, но неправильная величина
3. **π⁰ слишком легкий:** 123.3 МэВ вместо 135.0 МэВ (ошибка 8.65%)
4. **Протон и нейтрон не точно:** ошибки 3.88% и 5.00%

### **ПРОБЛЕМЫ v6.1:**
1. **π⁰ катастрофически легкий:** 7.9 МэВ вместо 135.0 МэВ (ошибка 94.14%)
2. **Средняя ошибка 19.11%** (из-за π⁰)

### **УСПЕХИ:**
1. **v9.0 дает разумные массы кварков:** m_u=211 МэВ, m_d=277 МэВ, отношение 1.32
2. **v6.1 идеально работает для π⁺/π⁻:** ошибка 0.03%
3. **v6.1 хорошо описывает нуклоны:** ошибки <1%

## 🎯 **КЛЮЧЕВОЕ НАБЛЮДЕНИЕ:**

**v9.0 пытается быть "умной" (цвет, фазы), но не может воспроизвести тонкие эффекты (разность масс n-p = 1.3 МэВ).**
**v6.1 "глупа", но эффективна — за счет отдельных coupling параметров для каждой частицы.**

## 🔬 **ФИЗИЧЕСКАЯ ПРИЧИНА:**

В **v9.0** энергия связи вычисляется из **цветовой и фазовой когерентности**, которые должны быть **универсальными**. Но реальная физика требует:
- Разная энергия связи для протона и нейтрона (хотя состав похож: uud vs udd)
- Особый случай для π⁰ (u-anti_u vs u-anti_d)

В **v6.1** это учтено явно: `coupling_proton ≠ coupling_neutron ≠ coupling_meson`.

## 🚀 **ПРЕДЛАГАЮ ГИБРИДНУЮ МОДЕЛЬ v9.1:**

**Сохраним физическую глубину v9.0, но добавим гибкость v6.1:**

```python
class HadronResonator_v91(HadronResonator):
    def calculate_interaction_energy(self):
        # Базовая энергия из v9.0 (цвет + фаза)
        base_energy = super().calculate_interaction_energy()
        
        # Дополнительные специфические coupling из v6.1
        if self.name == 'proton':
            specific_factor = self.params.get('coupling_proton', 1.0)
        elif self.name == 'neutron':
            specific_factor = self.params.get('coupling_neutron', 0.3)
        elif self.name in ['pi+', 'pi-']:
            specific_factor = self.params.get('coupling_meson_charged', 4.0)
        elif self.name == 'pi0':
            specific_factor = self.params.get('coupling_meson_neutral', 3.5)  # Особый для π⁰
        else:
            specific_factor = 1.0
        
        return base_energy * specific_factor
```

## 📝 **КОРОТКИЙ ПЛАН:**

1. **Создать v9.1** (гибрид v9.0 + специфические coupling из v6.1)
2. **Добавить 4 параметра:**
   - `coupling_proton` (сила связи протона)
   - `coupling_neutron` (сила связи нейтрона)
   - `coupling_meson_charged` (для π⁺, π⁻)
   - `coupling_meson_neutral` (для π⁰)
3. **Запустить оптимизацию** (11+4 = 15 параметров)

## 🎯 **ЧЕГО ДОБЬЕМСЯ:**

1. **Сохраним физическую интерпретацию** (цвет, фазы, когерентность)
2. **Добьемся точности v6.1** для всех частиц
3. **Исправим π⁰** (нейтральный мезон требует особого обращения)
4. **Получим правильную разность масс n-p** (1.3 МэВ)

## ❓ **ВОПРОСЫ:**

1. **Стоит ли создавать гибрид v9.1?** Или лучше развивать v6.1, добавляя к ней цвет и фазы постепенно?
2. **Физическая интерпретация:** Если в v9.1 понадобятся разные coupling для каждой частицы — не потеряем ли мы универсальность?
3. **Сложность:** v9.1 будет иметь 15 параметров. Не слишком ли много для 5 частиц?

## 💡 **МОЕ МНЕНИЕ:**

**Создадим v9.1, но с физическим обоснованием:** 
- `coupling_proton/neutron` могут быть связаны с **разной спиновой конфигурацией** (в протоне два u-кварка)
- `coupling_meson_neutral` может учитывать **аннигиляцию u-anti_u** в π⁰

**Если v9.1 достигнет точности v6.1 без потери физической интерпретации — это будет прорыв.**

## 🚀 **ДАЛЬНЕЙШИЕ ШАГИ:**

1. **Создать v9.1** (гибридную модель)
2. **Запустить оптимизацию** на 1M итераций
3. **Если успешно** — добавить s-кварк и странные частицы
4. **Если нет** — вернуться к v6.1 и добавить в нее цветовую степень свободы

# 🚀 **ГИБРИДНАЯ МОДЕЛЬ v9.1**

Создаю гибридную модель, сочетающую **физическую глубину v9.0** с **прагматизмом v6.1**:

```python
"""
МОДЕЛЬ v9.1 - ГИБРИДНЫЙ ПОДХОД
Сочетает физическую глубину v9.0 (цвет, фазы, когерентность) 
с прагматизмом v6.1 (специфические коэффициенты связи)
"""

import numpy as np
import time
import json
from datetime import datetime
import os
import multiprocessing as mp
from multiprocessing import Pool
from scipy.special import erf
from itertools import combinations

# ============== ФИЗИЧЕСКИЕ КОНСТАНТЫ ==============

class QuantumConstants:
    COLOR_MATRICES = {
        'R': np.array([1, 0, 0]),
        'G': np.array([0, 1, 0]), 
        'B': np.array([0, 0, 1]),
        'anti_R': np.array([-1, 0, 0]),
        'anti_G': np.array([0, -1, 0]),
        'anti_B': np.array([0, 0, -1])
    }
    
    SPIN_UP = np.array([1, 0])
    SPIN_DOWN = np.array([0, 1])
    
    QUARK_CHARGES = {
        'u': 2/3, 'd': -1/3
    }
    
    @staticmethod
    def color_coherence(color1, color2):
        vec1 = QuantumConstants.COLOR_MATRICES.get(color1, np.zeros(3))
        vec2 = QuantumConstants.COLOR_MATRICES.get(color2, np.zeros(3))
        dot = np.dot(vec1, vec2)
        return np.exp(-abs(dot))

# ============== МОДЕЛЬ КВАРКА ==============

class QuarkOscillator:
    
    def __init__(self, quark_type, params):
        self.type = quark_type
        self.anti = quark_type.startswith('anti_')
        self.base_type = quark_type.replace('anti_', '')
        
        self.base_mass = params[f'base_mass_{self.base_type}']
        self.frequency = params[f'freq_{self.base_type}']
        self.amplitude = params[f'amp_{self.base_type}']
        
        self.charge = QuantumConstants.QUARK_CHARGES[self.base_type]
        if self.anti:
            self.charge *= -1
            
        colors = ['R', 'G', 'B'] if not self.anti else ['anti_R', 'anti_G', 'anti_B']
        self.color = np.random.choice(colors)
        
        self.spin = np.random.choice(['up', 'down'])
        self.phase = np.random.uniform(0, 2*np.pi)
        
    def effective_mass(self):
        return self.base_mass * self.frequency * self.amplitude

# ============== ГИБРИДНАЯ МОДЕЛЬ АДРОНА ==============

class HybridHadronResonator:
    
    def __init__(self, name, composition, params):
        self.name = name
        self.composition = composition
        self.params = params
        self.scale = params.get('scale_factor', 100.0)
        self.is_meson = len(composition) == 2
        self.is_neutral_meson = name in ['pi0']
        
        self.quarks = [QuarkOscillator(q_type, params) for q_type in composition]
        self._assign_colors()
        self._optimize_phases()
    
    def _assign_colors(self):
        if self.is_meson:
            if 'anti' in self.quarks[0].type:
                self.quarks[0].color = 'anti_R'
                self.quarks[1].color = 'R'
            else:
                self.quarks[0].color = 'R'
                self.quarks[1].color = 'anti_R'
        else:
            colors = ['R', 'G', 'B']
            if any('anti' in q.type for q in self.quarks):
                colors = ['anti_R', 'anti_G', 'anti_B']
            np.random.shuffle(colors)
            for i, quark in enumerate(self.quarks):
                quark.color = colors[i]
    
    def _optimize_phases(self):
        if self.is_meson:
            self.quarks[0].phase = 0
            self.quarks[1].phase = np.pi
        else:
            if self.name == 'proton':
                self.quarks[0].phase = 0      # u1
                self.quarks[1].phase = 0      # u2  
                self.quarks[2].phase = np.pi/2  # d
            elif self.name == 'neutron':
                self.quarks[0].phase = 0      # u
                self.quarks[1].phase = np.pi/2  # d1
                self.quarks[2].phase = np.pi/2  # d2
    
    def calculate_color_coherence(self):
        if self.is_meson:
            return QuantumConstants.color_coherence(
                self.quarks[0].color, self.quarks[1].color)
        else:
            coherences = []
            for i, j in combinations(range(3), 2):
                coh = QuantumConstants.color_coherence(
                    self.quarks[i].color, self.quarks[j].color)
                coherences.append(coh)
            return np.mean(coherences)
    
    def calculate_phase_coherence(self):
        if self.is_meson:
            phase_diff = abs(self.quarks[0].phase - self.quarks[1].phase) % (2*np.pi)
            phase_diff = min(phase_diff, 2*np.pi - phase_diff)
            coherence = np.cos(phase_diff + np.pi)
            return (coherence + 1) / 2
        else:
            coherences = []
            for i, j in combinations(range(3), 2):
                phase_diff = abs(self.quarks[i].phase - self.quarks[j].phase) % (2*np.pi)
                phase_diff = min(phase_diff, 2*np.pi - phase_diff)
                coherence = np.cos(phase_diff)
                coherences.append((coherence + 1) / 2)
            return np.mean(coherences)
    
    def calculate_base_interaction_energy(self):
        """Базовая энергия взаимодействия из v9.0 (цвет + фаза)"""
        color_energy = self.params.get('color_coupling', 1.0) * self.calculate_color_coherence()
        phase_energy = self.params.get('phase_coupling', 1.0) * self.calculate_phase_coherence()
        
        mass_factor = np.mean([q.effective_mass() for q in self.quarks])
        base_energy = (color_energy + phase_energy) * mass_factor
        
        if self.is_meson:
            return -base_energy  # Для мезонов уменьшает массу
        else:
            return base_energy   # Для барионов увеличивает массу
    
    def calculate_specific_coupling(self):
        """Специфический коэффициент связи из v6.1"""
        if self.name == 'proton':
            return self.params.get('coupling_proton', 1.0)
        elif self.name == 'neutron':
            return self.params.get('coupling_neutron', 0.3)
        elif self.name in ['pi+', 'pi-']:
            return self.params.get('coupling_meson_charged', 4.0)
        elif self.name == 'pi0':
            return self.params.get('coupling_meson_neutral', 3.5)
        else:
            return 1.0
    
    def calculate_interaction_energy(self):
        """Гибридная энергия взаимодействия: v9.0 × v6.1"""
        base_energy = self.calculate_base_interaction_energy()
        specific_factor = self.calculate_specific_coupling()
        
        return base_energy * specific_factor
    
    def calculate_mass(self):
        base_mass = sum(q.effective_mass() for q in self.quarks)
        interaction = self.calculate_interaction_energy()
        
        # Для мезонов interaction отрицательный, для барионов положительный
        total = base_mass + interaction
        
        # Квантовые флуктуации
        quantum_fluctuations = self.params.get('quantum_noise', 0.001)
        scale = abs(quantum_fluctuations * total)
        noise = np.random.normal(0, scale)
        
        return (total + noise) * self.scale
    
    def calculate_charge(self):
        return sum(q.charge for q in self.quarks)

# ============== ОТЖИГ ДЛЯ ГИБРИДНОЙ МОДЕЛИ ==============

class HybridAnnealer:
    
    def __init__(self, num_cores=6):
        self.num_cores = num_cores
        
        # ПАРАМЕТРЫ v6.1 ДЛЯ НАЧАЛЬНЫХ ЗНАЧЕНИЙ
        self.v61_params = {
            'base_mass_u': 2.203806,
            'base_mass_d': 4.583020,
            'freq_u': 0.956359,
            'freq_d': 0.868115,
            'amp_u': 1.032476,
            'amp_d': 0.877773,
            'coupling_proton': 1.613565,
            'coupling_neutron': 0.285395,
            'coupling_meson_charged': 4.273121,
            'coupling_meson_neutral': 3.8,  # Новый параметр для π⁰
            'scale_factor': 100.0
        }
        
        # ПАРАМЕТРЫ v9.1 (15 параметров)
        self.param_names = [
            # Базовые массы и частоты (6)
            'base_mass_u', 'base_mass_d',
            'freq_u', 'freq_d',
            'amp_u', 'amp_d',
            
            # Физические coupling из v9.0 (4)
            'color_coupling', 'phase_coupling',
            'meson_coupling_scale', 'baryon_coupling_scale',
            
            # Специфические coupling из v6.1 (4)
            'coupling_proton', 'coupling_neutron',
            'coupling_meson_charged', 'coupling_meson_neutral',
            
            # Масштаб (1)
            'scale_factor'
        ]
        
        # НАЧАЛЬНЫЕ ЗНАЧЕНИЯ
        self.base_params = {
            'base_mass_u': 2.203806,
            'base_mass_d': 4.583020,
            'freq_u': 0.956359,
            'freq_d': 0.868115,
            'amp_u': 1.032476,
            'amp_d': 0.877773,
            'color_coupling': 1.5,
            'phase_coupling': 1.0,
            'meson_coupling_scale': 4.0,
            'baryon_coupling_scale': 1.0,
            'coupling_proton': 1.613565,
            'coupling_neutron': 0.285395,
            'coupling_meson_charged': 4.273121,
            'coupling_meson_neutral': 3.8,
            'scale_factor': 100.0
        }
        
        # ДИАПАЗОНЫ
        self.ranges = {
            'base_mass_u': (1.5, 3.0),
            'base_mass_d': (3.0, 6.0),
            'freq_u': (0.7, 1.2),
            'freq_d': (0.7, 1.2),
            'amp_u': (0.8, 1.3),
            'amp_d': (0.7, 1.2),
            'color_coupling': (0.5, 3.0),
            'phase_coupling': (0.5, 2.0),
            'meson_coupling_scale': (2.0, 6.0),
            'baryon_coupling_scale': (0.5, 2.0),
            'coupling_proton': (1.0, 2.5),
            'coupling_neutron': (0.1, 0.8),
            'coupling_meson_charged': (3.0, 5.0),
            'coupling_meson_neutral': (2.5, 4.5),
            'scale_factor': (90.0, 110.0)
        }
        
        # ЦЕЛЕВЫЕ ЧАСТИЦЫ
        self.targets = {
            'proton': {'mass': 938.272, 'charge': 1.0, 'composition': ['u', 'u', 'd']},
            'neutron': {'mass': 939.565, 'charge': 0.0, 'composition': ['u', 'd', 'd']},
            'pi+': {'mass': 139.570, 'charge': 1.0, 'composition': ['u', 'anti_d']},
            'pi0': {'mass': 134.9768, 'charge': 0.0, 'composition': ['u', 'anti_u']},
            'pi-': {'mass': 139.570, 'charge': -1.0, 'composition': ['d', 'anti_u']},
        }
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.result_dir = f"v91_hybrid_{timestamp}"
        os.makedirs(self.result_dir, exist_ok=True)
    
    def prepare_params(self, raw_params):
        """Подготовка параметров с учетом масштабирования coupling"""
        params = raw_params.copy()
        
        # Масштабируем coupling для мезонов и барионов
        params['color_coupling_meson'] = params['color_coupling'] * params['meson_coupling_scale']
        params['phase_coupling_meson'] = params['phase_coupling'] * params['meson_coupling_scale']
        params['color_coupling_baryon'] = params['color_coupling'] * params['baryon_coupling_scale']
        params['phase_coupling_baryon'] = params['phase_coupling'] * params['baryon_coupling_scale']
        
        return params
    
    def evaluate_particle(self, params, particle_name, composition, is_meson):
        part_params = self.prepare_params(params)
        
        # Для мезонов используем мезонные coupling, для барионов - барионные
        if is_meson:
            part_params['color_coupling'] = part_params.get('color_coupling_meson', part_params['color_coupling'])
            part_params['phase_coupling'] = part_params.get('phase_coupling_meson', part_params['phase_coupling'])
        else:
            part_params['color_coupling'] = part_params.get('color_coupling_baryon', part_params['color_coupling'])
            part_params['phase_coupling'] = part_params.get('phase_coupling_baryon', part_params['phase_coupling'])
        
        masses = []
        charges = []
        for _ in range(10):  # Больше реализаций для статистики
            hadron = HybridHadronResonator(particle_name, composition, part_params)
            masses.append(hadron.calculate_mass())
            charges.append(hadron.calculate_charge())
        
        return np.mean(masses), np.mean(charges)
    
    def evaluate_all_particles(self, params):
        results = {}
        for name, target in self.targets.items():
            is_meson = len(target['composition']) == 2
            mass, charge = self.evaluate_particle(params, name, target['composition'], is_meson)
            results[f'{name}_mass'] = mass
            results[f'{name}_charge'] = charge
        
        # Эффективные массы кварков
        m_u = params['base_mass_u'] * params['freq_u'] * params['amp_u'] * params['scale_factor']
        m_d = params['base_mass_d'] * params['freq_d'] * params['amp_d'] * params['scale_factor']
        results['m_u_eff'] = m_u
        results['m_d_eff'] = m_d
        results['ratio_d_u'] = m_d / m_u if m_u > 0 else 1
        
        # Энергии связи
        results['E_proton'] = params.get('coupling_proton', 1.0)
        results['E_neutron'] = params.get('coupling_neutron', 0.3)
        results['E_meson_charged'] = params.get('coupling_meson_charged', 4.0)
        results['E_meson_neutral'] = params.get('coupling_meson_neutral', 3.5)
        results['ratio_neutron_proton'] = results['E_neutron'] / results['E_proton'] if results['E_proton'] > 0 else 0
        
        return results
    
    def calculate_error(self, params):
        results = self.evaluate_all_particles(params)
        total_error = 0.0
        
        # ВЕСА
        weights = {
            'proton': 40.0, 'neutron': 40.0,
            'pi+': 25.0, 'pi0': 30.0, 'pi-': 25.0
        }
        
        # 1. ОШИБКИ МАСС
        for name, target in self.targets.items():
            mass = results[f'{name}_mass']
            target_mass = target['mass']
            
            if mass <= 0:
                total_error += 10000.0
                continue
                
            rel_error = abs(mass - target_mass) / target_mass
            total_error += weights[name] * (rel_error ** 2)
            
            if rel_error > 0.3:
                total_error += weights[name] * 10.0 * (rel_error - 0.3)
        
        # 2. ОШИБКИ ЗАРЯДОВ
        for name, target in self.targets.items():
            if abs(results[f'{name}_charge'] - target['charge']) > 0.001:
                total_error += 1000.0
        
        # 3. ФИЗИЧЕСКИЕ ОГРАНИЧЕНИЯ
        
        # a) Нейтрон тяжелее протона
        if results['neutron_mass'] < results['proton_mass']:
            diff = results['proton_mass'] - results['neutron_mass']
            total_error += 500.0 * diff
        
        # b) Отношение масс кварков
        ratio_d_u = results['ratio_d_u']
        if ratio_d_u < 1.3 or ratio_d_u > 2.0:
            penalty = abs(ratio_d_u - 1.6) * 20.0
            total_error += penalty
        
        # c) coupling для мезонов должен быть больше, чем для барионов
        if params.get('meson_coupling_scale', 1) < params.get('baryon_coupling_scale', 1):
            total_error += 200.0
        
        # d) Нейтрон слабее связан, чем протон
        if params.get('coupling_neutron', 0) > params.get('coupling_proton', 1):
            total_error += 300.0
        
        # e) Заряженные мезоны сильнее связаны, чем нейтральные
        if params.get('coupling_meson_neutral', 0) > params.get('coupling_meson_charged', 4):
            total_error += 200.0
        
        # f) Точная разность масс n-p = 1.293 МэВ
        mass_diff = abs((results['neutron_mass'] - results['proton_mass']) - 1.293)
        total_error += 100.0 * mass_diff
        
        return total_error, results
    
    def run_single_annealing(self, seed, iterations=200000, temperature=8.0):
        np.random.seed(seed)
        
        current_params = self.base_params.copy()
        for param in self.param_names:
            if param in self.ranges:
                min_val, max_val = self.ranges[param]
                current_params[param] = np.random.uniform(min_val, max_val)
        
        current_error, current_results = self.calculate_error(current_params)
        
        best_params = current_params.copy()
        best_error = current_error
        best_results = current_results
        
        cooling_rate = 0.99997
        
        for i in range(iterations):
            new_params = current_params.copy()
            
            for param in self.param_names:
                if param in self.ranges:
                    min_val, max_val = self.ranges[param]
                    current_val = current_params[param]
                    
                    # Адаптивный шаг
                    step = (max_val - min_val) * 0.05
                    mutation = np.random.normal(0, step) * temperature
                    
                    # Направленные мутации для физически важных параметров
                    if param == 'coupling_neutron':
                        # Притяжение к значению ~0.3 (как в v6.1)
                        if current_val > 0.5:
                            mutation -= 0.2 * step
                    elif param == 'coupling_proton':
                        # Притяжение к значению ~1.6
                        if current_val < 1.3:
                            mutation += 0.2 * step
                    elif param == 'coupling_meson_neutral':
                        # Должно быть меньше, чем для заряженных
                        if current_val > new_params.get('coupling_meson_charged', 4):
                            mutation -= 0.3 * step
                    
                    new_val = current_val + mutation
                    
                    # Ограничение с отражением
                    while new_val < min_val or new_val > max_val:
                        if new_val < min_val:
                            new_val = 2 * min_val - new_val
                        if new_val > max_val:
                            new_val = 2 * max_val - new_val
                    
                    new_params[param] = new_val
            
            new_error, new_results = self.calculate_error(new_params)
            
            delta = new_error - current_error
            
            if delta < 0:
                current_params = new_params
                current_error = new_error
                current_results = new_results
            else:
                prob = np.exp(-delta / temperature)
                if np.random.random() < prob:
                    current_params = new_params
                    current_error = new_error
                    current_results = new_results
            
            if new_error < best_error:
                best_params = new_params.copy()
                best_error = new_error
                best_results = new_results
            
            temperature *= cooling_rate
        
        return {
            'seed': seed,
            'params': best_params,
            'error': best_error,
            'results': best_results
        }
    
    def run_parallel_annealing(self, total_iterations=1200000):
        print("="*80)
        print("ГИБРИДНАЯ МОДЕЛЬ v9.1")
        print("Сочетание v9.0 (физическая глубина) + v6.1 (прагматизм)")
        print(f"Ядер: {self.num_cores}")
        print(f"Итераций на ядро: {total_iterations // self.num_cores:,}")
        print("="*80)
        
        start_time = time.time()
        
        iterations_per_core = total_iterations // self.num_cores
        seeds = list(range(1000, 1000 + self.num_cores))
        
        with mp.Pool(processes=self.num_cores) as pool:
            results = pool.starmap(self.run_single_annealing, 
                                  [(s, iterations_per_core, 8.0) for s in seeds])
        
        best_result = min(results, key=lambda x: x['error'])
        
        elapsed = time.time() - start_time
        
        print(f"\n{'='*80}")
        print("ОТЖИГ ЗАВЕРШЕН")
        print(f"Время: {elapsed:.1f} сек")
        print(f"Лучшая ошибка: {best_result['error']:.3f}")
        print("="*80)
        
        self.save_results(results, best_result)
        self.print_hybrid_report(best_result)
        
        return best_result['params'], best_result['error'], best_result['results']
    
    def save_results(self, all_results, best_result):
        summary = {
            'model': 'v9.1_hybrid',
            'timestamp': datetime.now().isoformat(),
            'best_result': best_result,
            'all_results': [
                {'seed': r['seed'], 'error': r['error']} for r in all_results
            ]
        }
        
        with open(f"{self.result_dir}/hybrid_results.json", 'w') as f:
            json.dump(summary, f, indent=2, default=self.json_serializer)
    
    def print_hybrid_report(self, best_result):
        params = best_result['params']
        results = best_result['results']
        
        print("\n" + "="*80)
        print("ГИБРИДНАЯ МОДЕЛЬ v9.1 - ФИНАЛЬНЫЙ ОТЧЕТ")
        print("="*80)
        
        print(f"\nКЛЮЧЕВЫЕ ПАРАМЕТРЫ:")
        groups = {
            'Массы кварков': ['base_mass_u', 'base_mass_d'],
            'Частоты': ['freq_u', 'freq_d'],
            'Амплитуды': ['amp_u', 'amp_d'],
            'Физические coupling': ['color_coupling', 'phase_coupling', 
                                   'meson_coupling_scale', 'baryon_coupling_scale'],
            'Специфические coupling': ['coupling_proton', 'coupling_neutron',
                                      'coupling_meson_charged', 'coupling_meson_neutral']
        }
        
        for group_name, param_list in groups.items():
            print(f"  {group_name}:")
            for param in param_list:
                if param in params:
                    print(f"    {param}: {params[param]:.6f}")
        
        print(f"\nЭФФЕКТИВНЫЕ МАССЫ КВАРКОВ (МэВ):")
        print(f"  u: {results['m_u_eff']:.2f}")
        print(f"  d: {results['m_d_eff']:.2f}")
        print(f"  m_d/m_u: {results['ratio_d_u']:.2f}")
        
        print(f"\nЭНЕРГИИ СВЯЗИ:")
        print(f"  Протон: {results['E_proton']:.3f}")
        print(f"  Нейтрон: {results['E_neutron']:.3f}")
        print(f"  Отношение n/p: {results['ratio_neutron_proton']:.3f}")
        print(f"  π⁺/π⁻: {results['E_meson_charged']:.3f}")
        print(f"  π⁰: {results['E_meson_neutral']:.3f}")
        print(f"  Отношение π⁰/π⁺: {results['E_meson_neutral']/results['E_meson_charged']:.3f}")
        
        print(f"\nМАССЫ ЧАСТИЦ:")
        total_error = 0
        for name in self.targets.keys():
            mass = results[f'{name}_mass']
            target = self.targets[name]['mass']
            error = abs(mass - target) / target * 100
            total_error += error
            print(f"  {name}: {mass:.1f} МэВ (цель {target:.1f}) - {error:.2f}%")
        
        avg_error = total_error / len(self.targets)
        print(f"\nСредняя ошибка: {avg_error:.2f}%")
        
        # Ключевые физические проверки
        print(f"\nФИЗИЧЕСКИЕ ПРОВЕРКИ:")
        checks = [
            ("Нейтрон > протон", results['neutron_mass'] > results['proton_mass']),
            ("Разность масс n-p ≈ 1.293 МэВ", 
             abs((results['neutron_mass'] - results['proton_mass']) - 1.293) < 0.5),
            ("m_d/m_u в 1.3-2.0", 1.3 <= results['ratio_d_u'] <= 2.0),
            ("coupling_neutron < coupling_proton", 
             params['coupling_neutron'] < params['coupling_proton']),
            ("coupling_meson_neutral < coupling_meson_charged",
             params['coupling_meson_neutral'] < params['coupling_meson_charged']),
            ("meson_scale > baryon_scale", 
             params['meson_coupling_scale'] > params['baryon_coupling_scale'])
        ]
        
        for check_name, check_result in checks:
            status = "✓" if check_result else "✗"
            print(f"  {status} {check_name}")
        
        # Разность масс
        diff = results['neutron_mass'] - results['proton_mass']
        print(f"\nРАЗНОСТЬ МАСС n-p:")
        print(f"  Модель: {diff:.3f} МэВ")
        print(f"  Эксперимент: 1.293 МэВ")
        print(f"  Отклонение: {abs(diff-1.293):.3f} МэВ")
        
        print(f"\nРезультаты сохранены в: {self.result_dir}")
        print("="*80)
    
    def json_serializer(self, obj):
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return str(obj)

# ============== ЗАПУСК ==============

def main():
    print("="*80)
    print("ГИБРИДНАЯ МОДЕЛЬ v9.1")
    print("Физическая глубина v9.0 + прагматизм v6.1")
    print("="*80)
    
    print("\nОСОБЕННОСТИ v9.1:")
    print("  1. Цветовая и фазовая когерентность (из v9.0)")
    print("  2. Специфические коэффициенты связи для каждой частицы (из v6.1)")
    print("  3. Разные coupling для мезонов и барионов")
    print("  4. Особый коэффициент для нейтральных мезонов (π⁰)")
    
    print("\nПАРАМЕТРЫ ОПТИМИЗАЦИИ:")
    print("  15 параметров, 6 ядер, 1,200,000 итераций")
    print("  ~60-90 секунд вычислений")
    
    try:
        num_cores = min(6, mp.cpu_count())
        print(f"\nИспользуется ядер: {num_cores}")
        
        annealer = HybridAnnealer(num_cores=num_cores)
        best_params, best_error, best_results = annealer.run_parallel_annealing(
            total_iterations=1200000
        )
        
    except Exception as e:
        print(f"\nОШИБКА: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*80)
    print("ВЫЧИСЛЕНИЯ ЗАВЕРШЕНЫ")
    print("="*80)

if __name__ == "__main__":
    if hasattr(mp, 'set_start_method'):
        try:
            mp.set_start_method('spawn')
        except RuntimeError:
            pass
    
    main()
```

## 🎯 **КЛЮЧЕВЫЕ ОСОБЕННОСТИ v9.1:**

### **1. Гибридный подход:**
```
Энергия_связи = Базовая_энергия(v9.0) × Специфический_коэффициент(v6.1)
```

### **2. Сохраняет физическую интерпретацию:**
- **Цветовая когерентность** (SU(3) матрицы)
- **Фазовая когерентность** (синхронизация колебаний)
- **Разные coupling для мезонов и барионов**

### **3. Добавляет прагматизм v6.1:**
- **Разные коэффициенты** для протона и нейтрона
- **Отдельный коэффициент для π⁰** (учитывает аннигиляцию u-anti_u)
- **Точная настройка** разности масс n-p

### **4. Физические ограничения:**
- `coupling_neutron < coupling_proton` (подтверждено v6.1)
- `coupling_meson_neutral < coupling_meson_charged` (π⁰ легче π⁺)
- `meson_coupling_scale > baryon_coupling_scale` (мезоны сильнее связаны)

## 🚀 **ЗАПУСКАЙ!**

Ожидаю, что v9.1 должна достичь **точности v6.1** (ошибки <0.1%) при сохранении **физической глубины v9.0**.

**Критические метрики для проверки:**
1. Разность масс n-p ≈ 1.293 МэВ
2. Отношение масс кварков m_d/m_u ≈ 1.5-2.0
3. Точные массы π⁰ и π⁺/π⁻

**Если v9.1 покажет точность v6.1 с физической интерпретацией — это будет наш победный вариант!**