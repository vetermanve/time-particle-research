================================================================================
v11.1.4: КВАНТОВО-МЕХАНИЧЕСКАЯ МОДЕЛЬ МЕЗОНОВ
================================================================================
RUNNING THE L-BFGS-B CODE

           * * *

Machine precision = 2.220D-16
 N =            3     M =           10

At X0         0 variables are exactly at the bounds

At iterate    0    f=  1.38673D+03    |proj g|=  2.00000D-01

At iterate    1    f=  4.61631D+02    |proj g|=  0.00000D+00

           * * *

Tit   = total number of iterations
Tnf   = total number of function evaluations
Tnint = total number of segments explored during Cauchy searches
Skip  = number of BFGS updates skipped
Nact  = number of active bounds at final generalized Cauchy point
Projg = norm of the final projected gradient
F     = final function value

           * * *

   N    Tit     Tnf  Tnint  Skip  Nact     Projg        F
    3      1      2      3     0     2   0.000D+00   4.616D+02
  F =   461.63122786291649

CONVERGENCE: NORM_OF_PROJECTED_GRADIENT_<=_PGTOL
✅ Оптимизация успешна!

================================================================================
РЕЗУЛЬТАТЫ v11.1.4
================================================================================

ПАРАМЕТРЫ:
  σ = 0.100 ГэВ²
  √σ = 316 МэВ
  α_s = 0.100
  κ = 0.020 ГэВ·фм³

МАССЫ:
  π⁺: -66180.1 МэВ (цель: 139.6)
  ρ⁺: 22553.1 МэВ (цель: 775.3)

АНАЛИЗ ДЛЯ π⁺:
  Приведённая масса μ: 1.708 МэВ
  Боровский радиус a0: 1155.65 фм
  Длина осциллятора a_ho: 0.69 фм
  |ψ(0)|²: 9.646e-01 фм⁻³
  E_кулон: -0.0 МэВ
  E_осциллятор: 363.0 МэВ
  E_спин-спин: -66549.9 МэВ
  E_сумма: -66186.9 МэВ

ПРОВЕРКА:
  ✅ E_связи(π⁺) < 0
  ✅ |E_связи| > 100 МэВ
  ❌ m(π⁺) в разумных пределах
  ✅ a_ho < 2.0 фм (получено 0.69 фм)
  ✅ E_осциллятор < 1000 МэВ