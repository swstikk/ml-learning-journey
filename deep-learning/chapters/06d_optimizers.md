# 1.6: Optimizers — Weights Kaise Update Hote Hain (SGD se Adam tak)

> **Backprop se gradients mil gaye. Ab un gradients se weights UPDATE kaise karein?**
> **Ye "kaise update karein" wala decision = OPTIMIZER.**

---

## Basic: SGD (Stochastic Gradient Descent)

Tu ye jaanta hai:
```
w_new = w_old - learning_rate × gradient

Example:
  w = 0.5, gradient = -0.04, lr = 0.1
  w_new = 0.5 - 0.1 × (-0.04) = 0.5 + 0.004 = 0.504
```

**Ye hai vanilla SGD. Simple. Lekin 2 problems hain:**

### Problem 1: Noisy Path

```
SGD ek sample ya chota batch dekh ke update karta hai.
Har sample alag direction mein le jaata hai:

  Loss surface:
       *
      ╱ ╲          ← optimal minimum yahan
     ╱   ╲
    *─→*←─*        ← SGD zigzag karta hai!
       │
       * ←─ kabhi kabhi GALAT direction bhi jaata hai!

Ye "noisy" hai — converge hota hai but SLOWLY, zigzag path se.
```

### Problem 2: Same Learning Rate for All Weights

```
Kuch features bahut SPARSE hain (rarely non-zero):
  has_bpr: Mostly 0, kabhi kabhi 1
  
Kuch features bahut DENSE hain (hamesha non-zero):
  sl_distance: Hamesha koi value hai

SAME lr dono ke liye? 
  Sparse feature ke rare gradient ko BADA step chahiye!
  Dense feature ke frequent gradient ko CHOTA step chahiye!

SGD ye difference nahi samajhta.
```

---

## Momentum — "Gend Rolling Down Hill"

**Idea:** Agar gradient baar baar SAME direction mein point kar raha hai, toh us direction mein TEZI se jaao!

```
Physics se: Ek ball slope pe ludo.
  Pehle dhire chalti hai.
  Phir momentum pakadti hai.
  Same direction → faster and faster!
  Direction change → slow down first.

Mathematically:
  v = beta × v_prev + gradient        ← velocity (momentum builds up)
  w = w - lr × v                      ← update with velocity

  beta = 0.9 (typically)
  Matlab: 90% purana momentum + 10% naya gradient
```

**Effect:**
```
Without momentum (SGD):         With momentum:
     *                              *
    ╱ ╲                            ╱│╲
   *   *                          * │ *
  ╱ ╲ ╱ ╲                          │
 *   *   *                         * ← STRAIGHT path!
 zigzag path                    momentum smooths it
```

---

## RMSProp — "Har Weight Ka Apna Learning Rate"

**Idea:** Agar ek weight ke gradients hamesha bade hain → lr chota kar do.
Agar gradients chhote hain → lr bada kar do. ADAPTIVE!

```
s = beta × s_prev + (1-beta) × gradient²    ← running avg of squared gradients
w = w - lr × gradient / sqrt(s + epsilon)    ← divide by sqrt(s)

epsilon = 1e-8 (taaki divide by zero na ho)
```

**Effect:**
- Features with large gradients → s bada → divide bada → chota step
- Features with small gradients → s chota → divide chota → bada step
- **Automatically adjusts per weight!**

---

## Adam — THE KING (SGD + Momentum + RMSProp)

**Adam = Adaptive Moment Estimation = Momentum + RMSProp combined!**

```
m = beta1 × m_prev + (1-beta1) × gradient       ← momentum (1st moment)
v = beta2 × v_prev + (1-beta2) × gradient²       ← RMSProp  (2nd moment)

m_hat = m / (1 - beta1^t)    ← bias correction (early steps fix)
v_hat = v / (1 - beta2^t)

w = w - lr × m_hat / (sqrt(v_hat) + epsilon)
```

**Defaults (almost NEVER change these):**
```
lr    = 0.001    ← learning rate
beta1 = 0.9      ← momentum decay
beta2 = 0.999    ← RMSProp decay
eps   = 1e-8     ← numerical stability
```

### Kyun Adam best hai?

```
╔═══════════════╦════════════════════════════════════════════╗
║ Optimizer     ║ Pros / Cons                                ║
╠═══════════════╬════════════════════════════════════════════╣
║ SGD           ║ Simple, but zigzag, same lr for all        ║
║ SGD+Momentum  ║ Smoother, but still same lr for all        ║
║ RMSProp       ║ Adaptive lr, but no momentum               ║
║ ADAM          ║ Adaptive lr + momentum = BEST OF BOTH      ║
╚═══════════════╩════════════════════════════════════════════╝

Rule: Use Adam. Default lr=0.001. 
      99% cases mein ye best ya near-best hoga.
      Kuch rare cases mein SGD+Momentum better hota hai 
      (very large datasets), but start Adam se karo HAMESHA.
```

---

## Learning Rate — Sabse Important Hyperparameter

```
lr = 0.1:    Bade steps → fast but can OVERSHOOT minimum → DIVERGE!
lr = 0.001:  Medium steps → good balance (Adam default)
lr = 0.00001: Tiny steps → bahut slow, time waste

  Loss                      Loss                     Loss
   |  *   *                  |  *                      |  * * * * * * * *
   | * * * *                 |   *                     |               *
   |*       *                |    *                    |
   └──────── epoch           |     * * *               └──────── epoch
   lr too high!              └──────── epoch            lr too low!
   DIVERGING                 lr just right!             TOO SLOW
                             CONVERGING!
```

**PyTorch mein:**
```python
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # bas!
```

---

## Summary — Kya Yaad Rakhna Hai

```
1. SGD = basic, zigzag, works but slow
2. Momentum = SGD + velocity, smoother path
3. RMSProp = per-weight adaptive learning rate
4. Adam = Momentum + RMSProp = USE THIS!
5. lr = 0.001 default, most important hyperparameter
6. PyTorch: torch.optim.Adam(model.parameters(), lr=0.001)
```
