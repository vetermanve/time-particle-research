"""
ЭКСПЕРИМЕНТ 30.6 — КАЛИБРОВКА МАСС И СИММЕТРИИ
"""

import numpy as np
import random
import math
from collections import Counter

class PreonV6:
    """ПРЕОН с нарушением симметрии материи/антиматерии"""
    
    def __init__(self, preon_id):
        self.id = preon_id
        
        # НАРУШЕНИЕ СИММЕТРИИ: 55% материя, 45% антиматерия
        is_antiparticle = random.random() > 0.55
        
        if is_antiparticle:
            # Антиматерия: обратные заряды
            self.charge_options = [-1/3, 2/3, 0]
            self.charge = random.choice(self.charge_options)
            self.antiparticle = True
            self.color_prefix = 'anti_'
        else:
            # Материя
            self.charge_options = [1/3, -1/3, 0]
            self.charge = random.choice(self.charge_options)
            self.antiparticle = False
            self.color_prefix = ''
        
        # Цвет (R, G, B)
        self.color = self.color_prefix + random.choice(['R', 'G', 'B'])
        
        # Масса преона (с небольшим разбросом)
        self.base_mass = random.uniform(0.9, 1.1)
        
        # Резонансный код
        self.code = [random.random() for _ in range(3)]

class QuarkV6:
    """КВАРК с калиброванными свойствами"""
    
    def __init__(self, preon_triplet):
        self.preons = preon_triplet
        
        # Заряд и тип
        self.charge = sum(p.charge for p in preon_triplet)
        
        if abs(self.charge - 2/3) < 0.1:
            self.type = 'u'
        elif abs(self.charge + 1/3) < 0.1:
            self.type = 'd'
        else:
            self.type = 'other'
        
        # Определяем, античастица ли это
        antiparticle_count = sum(1 for p in preon_triplet if p.antiparticle)
        if antiparticle_count >= 2:  # Если 2+ преона - античастицы
            self.antiparticle = True
            self.type = 'anti_' + self.type if self.type != 'other' else 'other'
        else:
            self.antiparticle = False
        
        # Цвет
        colors = [p.color for p in preon_triplet if not p.color.startswith('anti_')]
        if colors:
            self.color = max(set(colors), key=colors.count)
        else:
            # Все преоны - античастицы
            anti_colors = [p.color.replace('anti_', '') for p in preon_triplet]
            self.color = 'anti_' + max(set(anti_colors), key=anti_colors.count)
        
        # МАССА КВАРКА (калиброванная)
        base_mass = sum(p.base_mass for p in preon_triplet)
        
        # u-кварк легче d-кварка
        if self.type == 'u':
            self.mass = base_mass * 2.0
        elif self.type == 'd':
            self.mass = base_mass * 4.0
        else:
            self.mass = base_mass * 3.0

class HadronV6:
    """АДРОН с калиброванными массами"""
    
    def __init__(self, quark_triplet):
        self.quarks = quark_triplet
        
        # Проверка цветовой нейтральности
        colors = []
        for q in quark_triplet:
            color = q.color
            if color.startswith('anti_'):
                colors.append(color.replace('anti_', ''))
            else:
                colors.append(color)
        
        self.is_color_neutral = (len(set(colors)) >= 3)
        
        if self.is_color_neutral:
            # Заряд
            self.charge = sum(q.charge for q in quark_triplet)
            
            # Тип
            self.type = self._determine_type()
            
            # МАССА (калиброванная)
            self.mass = self._calculate_mass()
        else:
            self.type = "invalid"
            self.mass = 0
    
    def _determine_type(self):
        """Определяем тип адрона"""
        quark_types = [q.type for q in self.quarks]
        
        # Подсчет u и d кварков
        u_count = sum(1 for t in quark_types if t == 'u')
        d_count = sum(1 for t in quark_types if t == 'd')
        anti_u_count = sum(1 for t in quark_types if t == 'anti_u')
        anti_d_count = sum(1 for t in quark_types if t == 'anti_d')
        
        # Протон: uud
        if u_count == 2 and d_count == 1:
            return 'proton'
        # Нейтрон: udd
        elif u_count == 1 and d_count == 2:
            return 'neutron'
        # Антипротон: anti_u anti_u anti_d
        elif anti_u_count == 2 and anti_d_count == 1:
            return 'anti_proton'
        # Антинейтрон: anti_u anti_d anti_d
        elif anti_u_count == 1 and anti_d_count == 2:
            return 'anti_neutron'
        # Другие
        elif self.charge > 0:
            return 'positive_baryon'
        elif self.charge < 0:
            return 'negative_baryon'
        else:
            return 'neutral_baryon'
    
    def _calculate_mass(self):
        """КАЛИБРОВАННАЯ МАССА АДРОНА"""
        # Базовые массы кварков (калиброванные)
        quark_masses = {
            'u': 2.16,      # u-кварк (лёгкий)
            'd': 4.67,      # d-кварк (тяжелее)
            'anti_u': 2.16,
            'anti_d': 4.67
        }
        
        # Базовая масса из кварков
        base_mass = 0
        for q in self.quarks:
            base_mass += quark_masses.get(q.type, 3.0)
        
        # ЭНЕРГИЯ СВЯЗИ (разная для разных частиц)
        binding_factors = {
            'proton': 0.32,        # Сильная связь
            'neutron': 0.30,       # Немного слабее
            'anti_proton': 0.30,   # Слабее чем протон
            'anti_neutron': 0.28,  # Еще слабее
            'positive_baryon': 0.25,
            'negative_baryon': 0.22,
            'neutral_baryon': 0.20
        }
        
        binding_energy = base_mass * binding_factors.get(self.type, 0.25)
        
        # ИТОГОВАЯ МАССА (×100 для перевода в "МэВ")
        final_mass = (base_mass - binding_energy) * 310
        
        return max(final_mass, 1.0)
    
    def __repr__(self):
        if self.is_color_neutral:
            return f"Hadron[{self.type}, charge={self.charge:.1f}, mass={self.mass:.1f} MeV]"
        else:
            return "Hadron[INVALID]"

def experiment_30_6():
    """Запуск калиброванного эксперимента"""
    print("=" * 80)
    print("🚀 ЭКСПЕРИМЕНТ 30.6 — КАЛИБРОВКА МАСС И СИММЕТРИИ")
    print("=" * 80)
    
    # Параметры
    NUM_PREONS = 1000
    NUM_QUARKS_TARGET = 200
    
    # 1. Создаем преоны
    print("Создание преонов с нарушением симметрии...")
    preons = [PreonV6(i) for i in range(NUM_PREONS)]
    
    # Статистика преонов
    matter_count = sum(1 for p in preons if not p.antiparticle)
    antimatter_count = sum(1 for p in preons if p.antiparticle)
    print(f"  Материя: {matter_count} ({matter_count/NUM_PREONS*100:.1f}%)")
    print(f"  Антиматерия: {antimatter_count} ({antimatter_count/NUM_PREONS*100:.1f}%)")
    
    # 2. Ищем кварки
    print("\nПоиск кварков...")
    quarks = []
    used_preons = set()
    
    for _ in range(100000):
        if len(quarks) >= NUM_QUARKS_TARGET:
            break
        
        # Выбираем случайную тройку
        idxs = random.sample(range(len(preons)), 3)
        
        if any(idx in used_preons for idx in idxs):
            continue
        
        triplet = [preons[idx] for idx in idxs]
        quark = QuarkV6(triplet)
        
        # Принимаем только u и d кварки
        if quark.type in ['u', 'd', 'anti_u', 'anti_d']:
            quarks.append(quark)
            used_preons.update(idxs)
    
    print(f"Найдено {len(quarks)} кварков")
    
    # Статистика кварков
    quark_types = [q.type for q in quarks]
    type_counts = {t: quark_types.count(t) for t in set(quark_types)}
    
    for qtype, count in type_counts.items():
        print(f"  {qtype}: {count}")
    
    # 3. Строим адроны
    print("\nПостроение адронов...")
    
    # Группируем кварки по цвету
    quarks_by_color = {'R': [], 'G': [], 'B': [], 
                       'anti_R': [], 'anti_G': [], 'anti_B': []}
    
    for quark in quarks:
        if quark.color in quarks_by_color:
            quarks_by_color[quark.color].append(quark)
    
    # Строим цветонейтральные комбинации
    hadrons = []
    max_combinations = 200
    
    # Материальные комбинации (R+G+B)
    for r_q in quarks_by_color['R'][:20]:
        for g_q in quarks_by_color['G'][:20]:
            for b_q in quarks_by_color['B'][:20]:
                if len(hadrons) >= max_combinations:
                    break
                
                if r_q is g_q or r_q is b_q or g_q is b_q:
                    continue
                
                hadron = HadronV6([r_q, g_q, b_q])
                if hadron.is_color_neutral:
                    hadrons.append(hadron)
    
    # Антиматериальные комбинации (anti_R+anti_G+anti_B)
    for ar_q in quarks_by_color['anti_R'][:10]:
        for ag_q in quarks_by_color['anti_G'][:10]:
            for ab_q in quarks_by_color['anti_B'][:10]:
                if len(hadrons) >= max_combinations * 1.5:
                    break
                
                if ar_q is ag_q or ar_q is ab_q or ag_q is ab_q:
                    continue
                
                hadron = HadronV6([ar_q, ag_q, ab_q])
                if hadron.is_color_neutral:
                    hadrons.append(hadron)
    
    print(f"Построено {len(hadrons)} адронов")
    
    # 4. Анализ
    print("\n📊 АНАЛИЗ РЕЗУЛЬТАТОВ:")
    
    hadron_types = [h.type for h in hadrons]
    type_counts = {t: hadron_types.count(t) for t in set(hadron_types)}
    
    for htype, count in sorted(type_counts.items()):
        print(f"  {htype}: {count}")
    
    # Протоны и нейтроны
    protons = [h for h in hadrons if h.type == 'proton']
    neutrons = [h for h in hadrons if h.type == 'neutron']
    antiprotons = [h for h in hadrons if h.type == 'anti_proton']
    antineutrons = [h for h in hadrons if h.type == 'anti_neutron']
    
    if protons:
        print(f"\n🎯 ПРОТОНЫ: {len(protons)}")
        proton_masses = [p.mass for p in protons]
        print(f"  Средняя масса: {np.mean(proton_masses):.1f} МэВ")
        print(f"  Диапазон: {min(proton_masses):.1f} - {max(proton_masses):.1f} МэВ")
        
        # Ближайший к 938 МэВ
        closest = min(protons, key=lambda p: abs(p.mass - 938.3))
        print(f"  Ближайший к 938.3 МэВ: {closest.mass:.1f} МэВ")
    
    if neutrons:
        print(f"\n🎯 НЕЙТРОНЫ: {len(neutrons)}")
        neutron_masses = [n.mass for n in neutrons]
        print(f"  Средняя масса: {np.mean(neutron_masses):.1f} МэВ")
        print(f"  Диапазон: {min(neutron_masses):.1f} - {max(neutron_masses):.1f} МэВ")
        
        # Ближайший к 940 МэВ
        closest = min(neutrons, key=lambda n: abs(n.mass - 939.6))
        print(f"  Ближайший к 939.6 МэВ: {closest.mass:.1f} МэВ")
    
    # Соотношение материя/антиматерия
    matter_hadrons = [h for h in hadrons if not h.type.startswith('anti_')]
    antimatter_hadrons = [h for h in hadrons if h.type.startswith('anti_')]
    
    print(f"\n⚖️ СООТНОШЕНИЕ МАТЕРИЯ/АНТИМАТЕРИЯ:")
    print(f"  Материя: {len(matter_hadrons)} адронов")
    print(f"  Антиматерия: {len(antimatter_hadrons)} адронов")
    if antimatter_hadrons:
        print(f"  Отношение: {len(matter_hadrons)/len(antimatter_hadrons):.2f}:1")
    
    # Соотношение нейтрон/протон
    if protons and neutrons:
        print(f"\n⚖️ СООТНОШЕНИЕ НЕЙТРОН/ПРОТОН:")
        print(f"  Протоны: {len(protons)}, Нейтроны: {len(neutrons)}")
        print(f"  Отношение n/p: {len(neutrons)/len(protons):.2f}")
    
    # Разность масс нейтрон-протон
    if protons and neutrons:
        avg_proton = np.mean([p.mass for p in protons])
        avg_neutron = np.mean([n.mass for n in neutrons])
        mass_diff = avg_neutron - avg_proton
        print(f"\n📏 РАЗНОСТЬ МАСС НЕЙТРОН-ПРОТОН:")
        print(f"  ΔM = {mass_diff:.3f} МэВ (цель: 1.293 МэВ)")
        print(f"  Ошибка: {abs(mass_diff - 1.293):.3f} МэВ")
    
    return hadrons

# Запуск
if __name__ == "__main__":
    hadrons = experiment_30_6()