"""
ЭКСПЕРИМЕНТ 32.0 - ТОЧНЫЙ СИНТЕЗ МОДЕЛЕЙ
Сохраняем структуру 30.5, но используем точные параметры v6.1
"""

import numpy as np
import random
from collections import Counter

class FundamentalThreadV32:
    """ФУНДАМЕНТАЛЬНАЯ НИТЬ с точными параметрами v6.1"""
    
    def __init__(self, thread_id):
        self.id = thread_id
        
        # ТОЧНЫЕ ПАРАМЕТРЫ ИЗ v6.1:
        # Типы нитей соответствуют преонам с дробными зарядами
        rand = random.random()
        
        # Распределение как в v6.1: нужно 40% U, 40% D, 20% N
        # Но для получения правильного соотношения кварков:
        # u-кварк = AAB (2×(+1/3) + 1×(-1/3)) = +1/3
        # d-кварк = ABB (1×(+1/3) + 2×(-1/3)) = -1/3
        
        if rand < 0.4:  # Преон типа A (+1/3)
            self.charge = 1/3
            self.base_mass = 2.203806 / 3  # Потом умножим на freq и amp
            self.freq = 0.956359
            self.amp = 1.032476
            self.type = 'A'
        elif rand < 0.8:  # Преон типа B (-1/3)
            self.charge = -1/3
            self.base_mass = 4.583020 / 3  # Для d-кварка
            self.freq = 0.868115
            self.amp = 0.877773
            self.type = 'B'
        else:  # Нейтральный преон (0)
            self.charge = 0
            self.base_mass = 1.0
            self.freq = 1.0
            self.amp = 1.0
            self.type = 'N'
        
        # Резонансный код (как в 30.5)
        self.code = [random.random() for _ in range(3)]
        
        # Цвет
        if self.type != 'N':
            self.color = random.choice(['R', 'G', 'B'])
        else:
            self.color = 'neutral'
        
        # Эффективная масса преона
        self.effective_mass = self.base_mass * self.freq * self.amp
        
        # Фаза
        self.phase = random.uniform(0, 2 * np.pi)

class QuarkV32:
    """КВАРК с ВЫВОДОМ ИЗ ТОЧНЫХ ПАРАМЕТРОВ"""
    
    def __init__(self, thread_triplet):
        self.threads = thread_triplet
        
        # 1. ЗАРЯД (из эксперимента 30.5)
        self.charge = sum(t.charge for t in thread_triplet)
        
        # 2. ТИП КВАРКА
        if abs(self.charge - 2/3) < 0.01:
            self.type = 'u'
            # ТОЧНЫЕ ПАРАМЕТРЫ u-кварка из v6.1
            self.base_mass = 2.203806
            self.freq = 0.956359
            self.amp = 1.032476
        elif abs(self.charge + 1/3) < 0.01:
            self.type = 'd'
            # ТОЧНЫЕ ПАРАМЕТРЫ d-кварка из v6.1
            self.base_mass = 4.583020
            self.freq = 0.868115
            self.amp = 0.877773
        else:
            self.type = 'other'
            self.base_mass = 1.0
            self.freq = 1.0
            self.amp = 1.0
        
        # 3. ЦВЕТ (преобладающий из нитей)
        colors = [t.color for t in thread_triplet if t.color != 'neutral']
        if colors:
            self.color = max(set(colors), key=colors.count)
        else:
            self.color = random.choice(['R', 'G', 'B'])
        
        # 4. ЭФФЕКТИВНАЯ МАССА (ТОЧНАЯ ФОРМУЛА)
        # m_quark_eff = base_mass × freq × amp
        self.effective_mass = self.base_mass * self.freq * self.amp
        
        # 5. КАЧЕСТВО СИНХРОНИЗАЦИИ НИТЕЙ
        self.sync_quality = self._calculate_thread_sync()
    
    def _calculate_thread_sync(self):
        """Вычисление качества синхронизации тройки нитей"""
        # Используем резонансные коды
        sync_values = []
        for i in range(3):
            for j in range(i+1, 3):
                # Корреляция кодов (как в эксперименте 30)
                code_corr = np.dot(self.threads[i].code, self.threads[j].code)
                norm_i = np.linalg.norm(self.threads[i].code)
                norm_j = np.linalg.norm(self.threads[j].code)
                
                if norm_i > 0 and norm_j > 0:
                    similarity = code_corr / (norm_i * norm_j)
                else:
                    similarity = 0
                
                # Фазовая синхронизация
                phase_diff = abs(self.threads[i].phase - self.threads[j].phase)
                phase_sync = np.cos(phase_diff)
                
                # Комбинируем
                pair_sync = 0.6 * similarity + 0.4 * phase_sync
                sync_values.append(pair_sync)
        
        return np.mean(sync_values) if sync_values else 0

class HadronV32:
    """АДРОН с ТОЧНОЙ ФОРМУЛОЙ v6.1"""
    
    def __init__(self, quark_triplet):
        self.quarks = quark_triplet
        
        # 1. ПРОВЕРКА ЦВЕТОВОЙ НЕЙТРАЛЬНОСТИ
        colors = [q.color for q in quark_triplet]
        self.is_color_neutral = len(set(colors)) >= 3
        
        if not self.is_color_neutral:
            self.mass = 0
            self.charge = 0
            self.type = "invalid"
            return
        
        # 2. ЗАРЯД
        self.charge = sum(q.charge for q in quark_triplet)
        
        # 3. ТИП АДРОНА
        self.type = self._determine_type()
        
        # 4. БАЗОВАЯ МАССА (формула из v6.1)
        base_mass = sum(q.effective_mass for q in quark_triplet)
        
        # 5. ЭНЕРГИЯ СИНХРОНИЗАЦИИ (coupling из v6.1)
        coupling = self._calculate_coupling()
        
        # 6. ИТОГОВАЯ МАССА (ТОЧНАЯ ФОРМУЛА v6.1)
        # M = (base_mass + coupling) × 100
        self.mass = (base_mass + coupling) * 100
    
    def _determine_type(self):
        """Точное определение типа адрона"""
        quark_types = [q.type for q in self.quarks]
        type_count = Counter(quark_types)
        
        u_count = type_count.get('u', 0)
        d_count = type_count.get('d', 0)
        
        # Протон: uud, заряд = +1
        if (abs(self.charge - 1.0) < 0.01 and 
            u_count == 2 and d_count == 1):
            return "proton"
        
        # Нейтрон: udd, заряд = 0
        if (abs(self.charge - 0.0) < 0.01 and 
            u_count == 1 and d_count == 2):
            return "neutron"
        
        return "other_baryon"
    
    def _calculate_coupling(self):
        """ТОЧНЫЙ РАСЧЕТ COUPLING ИЗ ПАРАМЕТРОВ v6.1"""
        
        if self.type == "proton":
            base_coupling = 1.613565
        elif self.type == "neutron":
            base_coupling = 0.285395
        else:
            base_coupling = 1.0
        
        # Корректируем на качество синхронизации кварков
        quark_sync = self._calculate_quark_sync()
        
        # coupling = базовая величина × качество синхронизации
        return base_coupling * quark_sync
    
    def _calculate_quark_sync(self):
        """Качество синхронизации кварков в адроне"""
        # Используем фазы кварков
        phases = []
        for quark in self.quarks:
            # Фаза кварка = средняя фаза его нитей
            quark_phase = np.mean([t.phase for t in quark.threads])
            phases.append(quark_phase)
        
        # Вычисляем фазовую когерентность
        sync_sum = 0
        pairs = 0
        
        for i in range(3):
            for j in range(i+1, 3):
                phase_diff = abs(phases[i] - phases[j]) % (2*np.pi)
                diff = min(phase_diff, 2*np.pi - phase_diff)
                
                # Когерентность: cos(разности фаз)
                coherence = np.cos(diff)
                sync_sum += coherence
                pairs += 1
        
        if pairs > 0:
            avg_sync = sync_sum / pairs
            # Преобразуем в [0, 1]
            return (avg_sync + 1) / 2
        
        return 0.5

def run_experiment_32():
    """ЭКСПЕРИМЕНТ 32.0 - ТОЧНЫЙ СИНТЕЗ"""
    print("="*80)
    print("🚀 ЭКСПЕРИМЕНТ 32.0 - СИНТЕЗ v6.1 И 30.5")
    print("="*80)
    
    # Параметры
    N_THREADS = 1000
    N_QUARKS = 100
    
    print(f"1. Создание {N_THREADS} нитей с параметрами v6.1...")
    threads = [FundamentalThreadV32(i) for i in range(N_THREADS)]
    
    # Статистика нитей
    thread_types = [t.type for t in threads]
    type_counts = Counter(thread_types)
    print(f"   A(+1/3): {type_counts.get('A', 0)}, B(-1/3): {type_counts.get('B', 0)}, N(0): {type_counts.get('N', 0)}")
    
    print(f"\n2. Образование {N_QUARKS} кварков...")
    quarks = []
    used_threads = set()
    
    # Правило образования кварков:
    # u-кварк = AAB (2×A + 1×B) → заряд = 2/3
    # d-кварк = ABB (1×A + 2×B) → заряд = -1/3
    
    for attempt in range(100000):
        if len(quarks) >= N_QUARKS:
            break
        
        # Выбираем случайную тройку нитей
        idx = random.sample(range(len(threads)), 3)
        
        # Проверяем, что нити не использованы
        if any(i in used_threads for i in idx):
            continue
        
        triplet = [threads[i] for i in idx]
        
        # Проверяем состав (идеальное соответствие не обязательно,
        # но заряд должен быть близок к ±2/3 или ∓1/3)
        charge = sum(t.charge for t in triplet)
        
        if abs(charge - 2/3) < 0.1:  # u-кварк
            quark = QuarkV32(triplet)
            if quark.type == 'u':
                quarks.append(quark)
                used_threads.update(idx)
        elif abs(charge + 1/3) < 0.1:  # d-кварк
            quark = QuarkV32(triplet)
            if quark.type == 'd':
                quarks.append(quark)
                used_threads.update(idx)
    
    print(f"   Образовано кварков: {len(quarks)}")
    quark_types = [q.type for q in quarks]
    qtype_counts = Counter(quark_types)
    print(f"   u-кварки: {qtype_counts.get('u', 0)}, d-кварки: {qtype_counts.get('d', 0)}")
    
    print(f"\n3. Образование адронов...")
    # Группируем кварки по цветам
    quarks_by_color = {'R': [], 'G': [], 'B': []}
    for q in quarks:
        if q.color in quarks_by_color:
            quarks_by_color[q.color].append(q)
    
    # Формируем цветонейтральные комбинации
    hadrons = []
    max_hadrons = 50
    
    for r_q in quarks_by_color['R'][:15]:
        for g_q in quarks_by_color['G'][:15]:
            for b_q in quarks_by_color['B'][:15]:
                if len(hadrons) >= max_hadrons:
                    break
                
                if r_q is g_q or r_q is b_q or g_q is b_q:
                    continue
                
                hadron = HadronV32([r_q, g_q, b_q])
                if hadron.is_color_neutral:
                    hadrons.append(hadron)
    
    print(f"   Образовано адронов: {len(hadrons)}")
    
    # Фильтруем протоны и нейтроны
    protons = [h for h in hadrons if h.type == "proton"]
    neutrons = [h for h in hadrons if h.type == "neutron"]
    
    print(f"\n4. РЕЗУЛЬТАТЫ:")
    print(f"   Протоны: {len(protons)}")
    print(f"   Нейтроны: {len(neutrons)}")
    
    if protons:
        proton_masses = [p.mass for p in protons]
        print(f"\n   ПРОТОНЫ:")
        print(f"     Средняя масса: {np.mean(proton_masses):.3f} МэВ")
        print(f"     Цель: 938.272 МэВ")
        print(f"     Отклонение: {np.mean(proton_masses) - 938.272:.3f} МэВ")
        
        # Лучший протон
        best_proton = min(protons, key=lambda p: abs(p.mass - 938.272))
        print(f"     Лучший: {best_proton.mass:.3f} МэВ")
    
    if neutrons:
        neutron_masses = [n.mass for n in neutrons]
        print(f"\n   НЕЙТРОНЫ:")
        print(f"     Средняя масса: {np.mean(neutron_masses):.3f} МэВ")
        print(f"     Цель: 939.565 МэВ")
        print(f"     Отклонение: {np.mean(neutron_masses) - 939.565:.3f} МэВ")
        
        best_neutron = min(neutrons, key=lambda n: abs(n.mass - 939.565))
        print(f"     Лучший: {best_neutron.mass:.3f} МэВ")
    
    # Разность масс
    if protons and neutrons:
        avg_proton = np.mean([p.mass for p in protons])
        avg_neutron = np.mean([n.mass for n in neutrons])
        mass_diff = avg_neutron - avg_proton
        
        print(f"\n   РАЗНОСТЬ МАСС n-p:")
        print(f"     Эксперимент: {mass_diff:.6f} МэВ")
        print(f"     Цель: 1.293 МэВ")
        print(f"     Отклонение: {abs(mass_diff - 1.293):.6f} МэВ")
    
    return hadrons, protons, neutrons

# Запуск
if __name__ == "__main__":
    hadrons, protons, neutrons = run_experiment_32()