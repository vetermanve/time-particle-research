🚀 ЭКСПЕРИМЕНТ №30 — ЗАПУЩЕН!
Версия: 30.0
Время: 20251205_232538
================================================================================
Создание 1000 нитей времени...
Построение сети синхронизации...
Добавлено 166 резонансных связей
Поиск ансамблей размером 2-3...
/Users/veter/.pyenv/versions/3.8.19/lib/python3.8/site-packages/numpy/core/fromnumeric.py:3464: RuntimeWarning: Mean of empty slice.
  return _methods._mean(a, axis=axis, dtype=dtype,
/Users/veter/.pyenv/versions/3.8.19/lib/python3.8/site-packages/numpy/core/_methods.py:192: RuntimeWarning: invalid value encountered in scalar divide
  ret = ret.dtype.type(ret / rcount)

❌ ОШИБКА В ЭКСПЕРИМЕНТЕ: 'SyncEnsemble' object has no attribute 'stability'
Traceback (most recent call last):
  File "proto.py", line 813, in run_experiment_30
    ensembles = universe.find_sync_ensembles(min_size=2, max_size=3)
  File "proto.py", line 586, in find_sync_ensembles
    ensemble = SyncEnsemble(clique, self.threads)
  File "proto.py", line 204, in __init__
    self.mass = self._calculate_mass()
  File "proto.py", line 285, in _calculate_mass
    mass = base_mass * (1.0 + self.stability)
AttributeError: 'SyncEnsemble' object has no attribute 'stability'
Мы получили ошибку: атрибут 'stability' не найден при вычислении массы в SyncEnsemble.
Это произошло потому, что в методе _calculate_mass мы обращаемся к self.stability, но в тот момент, когда вычисляется масса, стабильность еще не рассчитана.
В конструкторе SyncEnsemble мы вычисляем стабильность после массы? Давайте посмотрим на порядок:

В init:
self.sync_matrix = ... (вычисляется)
self.avg_sync_strength = ... (вычисляется)
self.coherence = ... (вычисляется)
self.mass = self._calculate_mass() # здесь используется self.stability, но стабильность еще не вычислена!
self.charge = ...
self.spin = ...
self.color = ...
self.stability = self._calculate_stability() # стабильность вычисляется после массы!

Поэтому в методе _calculate_mass мы не можем использовать self.stability, потому что она еще не определена.

Исправим порядок: сначала вычисляем стабильность, затем массу.

Но также обратите внимание, что в _calculate_mass мы используем self.stability, а _calculate_stability использует self.coherence и другие параметры, которые уже должны быть вычислены.

Давайте переупорядочим:

sync_matrix, avg_sync_strength, coherence - в этом порядке, потому что они зависят друг от друга.

stability - зависит от coherence и других.

mass - зависит от stability.

Изменения:

В конструкторе SyncEnsemble:

text
    self.sync_matrix = self._calculate_sync_matrix()
    self.avg_sync_strength = self._calculate_avg_sync()
    self.coherence = self._calculate_coherence()
    self.stability = self._calculate_stability()   # <- теперь перед массой
    self.mass = self._calculate_mass()
    ... остальное
Также в _calculate_mass мы используем self.stability, так что теперь она будет определена.

Давайте исправим и перезапустим.

Однако, есть еще одна проблема: в _calculate_mass мы используем self.stability, но в _calculate_stability мы используем self.coherence и другие, которые уже вычислены. Это нормально.

Внесем это исправление и также исправим аналогичную проблему в _estimate_lifetime, которая использует self.stability.

Но в _estimate_lifetime мы используем self.stability, которая теперь будет вычислена до вызова _estimate_lifetime (поскольку _estimate_lifetime вызывается после _calculate_stability в конструкторе).

Давайте изменим порядок в конструкторе:

text
    self.sync_matrix = self._calculate_sync_matrix()
    self.avg_sync_strength = self._calculate_avg_sync()
    self.coherence = self._calculate_coherence()
    self.stability = self._calculate_stability()
    self.mass = self._calculate_mass()
    self.charge = self._calculate_charge()
    self.spin = self._calculate_spin()
    self.color = self._assign_color()
    self.lifetime = self._estimate_lifetime()
Теперь все должно быть в порядке.

Кроме того, в методе _calculate_stability мы используем self.coherence и self.sync_matrix и self.avg_sync_strength, которые уже вычислены.

Внесем эти изменения и перезапустим эксперимент.

Но также обратите внимание на предупреждение о пустом срезе в numpy. Это возникает, когда нет ненулевых элементов в non_zero_strengths. Мы должны это обработать.

В методе _calculate_stability:

text
    non_zero_strengths = self.sync_matrix[self.sync_matrix > 0]
    avg_strength = np.mean(non_zero_strengths) if len(non_zero_strengths) > 0 else 0.0
Аналогично, в _calculate_avg_sync:


text
    upper_tri = self.sync_matrix[np.triu_indices(self.size, k=1)]
    if len(upper_tri) > 0:
        non_zero = upper_tri[upper_tri > 0]
        if len(non_zero) > 0:
            return np.mean(non_zero)
    return 0.0
Давайте исправим и эти моменты.

Я внесу все эти исправления и перезапущу эксперимент.

Отлично! Мы нашли ошибку в порядке инициализации. Нужно вычислить stability перед использованием в _calculate_mass(). Давайте исправим код и продолжим эксперимент.