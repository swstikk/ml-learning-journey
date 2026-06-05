# Deep Dive into CNN: Convolutional Neural Networks (Part 1)
===========================================================

> **Phase 2.1: CNN ka Foundation**
> **Style:** Hinglish, slow-paced, clear math, visualization, from scratch.

Abhi humne do models train kiye:
1. **MNIST (MLP):** 97.8% Accuracy
2. **DNA Promoter (MLP):** 99.8% Accuracy

Dono projects mein humne **Flattening** ki. Humne 2D image ($28 \times 28$) ya 2D DNA encoding ($60 \times 4$) ko ek lambi 1D list mein convert kiya. 
Lekin biology aur real-world problems mein ye approach fail ho jati hai. 

---

## 1. MLP ki Kamiyan: Kyun Chahiye CNN?

Maan lo aapke paas ek billi (cat) ki photo hai:
* **MLP problem:** Agar billi photo ke center mein hai, to MLP use seekh lega. Lekin agar billi photo ke corner mein chali gayi, to flatten karne pe uske pixels ki positions change ho jayengi! MLP use bilkul nayi cheez samajhne lagega (No translation invariance).
* **Weight Explosion:** Agar hamari image badi ho (jaise $256 \times 256 \times 3$ channels = 196,608 pixels), to sirf pehli hidden layer (with 512 neurons) mein hi **100 Million weights** ho jayenge! Model crash ho jayega aur seekh nahi payega.

### Solution: CNN (Convolutional Neural Networks)
CNN do ideas pe kaam karta hai:
1. **Local Connectivity:** Har neuron puri image se connect hone ki jagah sirf ek chhote region (jaise $3 \times 3$ pixels) ko dekhta hai.
2. **Weight Sharing:** Pura network ek hi filter (weights) ko poori image par slide karke features dhoondhta hai. Isse parameters millions se drop hokar thousands mein aa jate hain!

---

## 2. The Core: Convolution Operation (Ghisatna / Sliding Window)

Convolution ka matlab hai ek chhota **Filter** (ya **Kernel**) lena aur use input image ke upar slide karna.

### Example se samjho (The Math):
Maan lo hamari image $5 \times 5$ pixels ki hai aur filter $3 \times 3$ ka hai.

**Input Image ($X$):**
```
1  0  1  0  0
0  1  1  1  0
0  0  1  1  1
0  0  0  1  0
0  1  0  0  1
```

**Filter ($W$) — Vertial Edge Detector:**
```
 1  0 -1
 1  0 -1
 1  0 -1
```

#### Step 1: Filter ko Top-Left corner pe rakho:
Input ka sub-matrix ($3 \times 3$):
```
1  0  1
0  1  1
0  0  1
```

Hum **Element-wise multiplication** karenge aur sabko add karenge:

$$\text{Output} = (1 \times 1) + (0 \times 0) + (1 \times -1) + (0 \times 1) + (1 \times 0) + (1 \times -1) + (0 \times 1) + (0 \times 0) + (1 \times -1)$$

$$\text{Output} = 1 + 0 - 1 + 0 + 0 - 1 + 0 + 0 - 1 = -2$$

Ye value humari output feature map ka pehla element banegi.

#### Step 2: Filter ko 1 pixel right slide karo (Stride = 1):
Ab sub-matrix hoga:
```
0  1  0
1  1  1
0  1  1
```
Multiplication:

$$\text{Output} = (0 \times 1) + (1 \times 0) + (0 \times -1) + (1 \times 1) + (1 \times 0) + (1 \times -1) + (0 \times 1) + (1 \times 0) + (1 \times -1)$$

$$\text{Output} = 0 + 0 + 0 + 1 + 0 - 1 + 0 + 0 - 1 = -1$$

Aise hi filter poori image pe slide karta hai. 

$5 \times 5$ input pe jab $3 \times 3$ filter slide karega, to output shape $3 \times 3$ ho jayegi!

---

## 3. The 3 Pillars: Stride, Padding, Pooling

CNN ki shapes ko control karne ke liye 3 main concepts hote hain:

### A. Padding (Zero Border)
Jab filter slide karta hai, to edges wale pixels bahut kam baar touch hote hain, aur image ki shape chhoti ho jati hai ($5 \times 5 \rightarrow 3 \times 3$).
Isse bachne ke liye hum image ke borders pe **Zeros (0)** laga dete hain.

* **Valid Padding (No Padding):** Border pe kuch nahi lagate. Image shape chhoti hoti jati hai.
* **Same Padding:** Zeros is tarah add karte hain ki output shape exact input shape ke barabar rahe ($5 \times 5$ remains $5 \times 5$).

```
0  0  0  0  0  0  0
0  1  0  1  0  0  0
0  0  1  1  1  0  0
0  0  0  1  1  1  0   <-- Padding = 1 (Zeros added around)
0  0  0  0  1  0  0
0  0  1  0  0  1  0
0  0  0  0  0  0  0
```

### B. Stride (Jump Size)
Filter ek baar mein kitne pixels aage jump karega use Stride bolte hain.
* `Stride = 1`: Ek-ek pixel aage badhega (Default).
* `Stride = 2`: Do-do pixels jump karega (Output shape half ho jayegi!).

### C. Max Pooling (Downsampling)
Convolution ke baad hum image ke features ko summarise karte hain. 
**Max Pooling** ek window (jaise $2 \times 2$) leta hai aur usme se **MAXIMUM value** ko choose karta hai.

```
Input (4 x 4):
1  3 │ 2  1
0  5 │ 4  0
─────┼─────
0  1 │ 3  5
2  1 │ 0  2

MaxPool (with 2x2 window & Stride=2):
[1,3,0,5] ka max = 5  │  [2,1,4,0] ka max = 4
─────────────────────┼─────────────────────
[0,1,2,1] ka max = 2  │  [3,5,0,2] ka max = 5

Output (2 x 2):
5  4
2  5
```
**Benefits:** 
1. Spatial size half ho gayi (RAM/GPU bach gaya).
2. Features robust ho gaye (Chhote moti position changes se farak nahi padta).

---

## 4. 1D CNN: AlphaGenome & Sequence Modelling

Aapne standard tutorial mein 2D CNN (images ke liye) dekha hoga. Lekin genetics (DNA sequences) aur trading (time-series prices) mein **1D CNN** use hota hai!

### 1D CNN Kaise Kaam Karta Hai?
* **Input DNA sequence:** one-hot encoded shape $60 \times 4$. Yahan 60 base pairs hain, aur 4 channels (A, C, G, T) hain.
* **Filter size:** Maan lo $5 \times 4$. 
  * Filter ki width exact **4** (channels) hoti hai.
  * Filter ki length **5** hoti hai (jo 5 base pairs ke pattern ko cover karegi).
* **Sliding:** Filter sirf **horizontal direction** (DNA ki length ke along) slide karega, vertical nahi!

```text
Sequence (Length 60, Channels 4):
[A C G T A T A A T ...]
  │ │ │ │ │ │
 └───┬───┘
   1D Filter (Length 6) slides only along the length!
```

Jab 1D filter slide karta hai:
1. Agar uske weights `TATAAT` ke features se match karte hain, to wo biological motif detect hone par **High Output Value (Activation)** deta hai!
2. Ye pattern detect karne ka process dynamic hai, chahe motif position 10 pe ho ya 30 pe. CNN use easily pakad leta hai!

---

## 5. Homework: Visualization & Formula Check

Kal hum CNN ka CIFAR-10 image classifier aur DNA 1D CNN code likhenge. Aaj sone se pehle ek important formula dimaag mein fit kar lo:

### Shape Formula:
Agar input height $H_{in}$ hai, padding $P$ hai, kernel size $K$ hai, aur stride $S$ hai, to output height $H_{out}$ kya hogi?

$$H_{out} = \left\lfloor \frac{H_{in} + 2P - K}{S} \right\rfloor + 1$$

*Note: $\lfloor \cdot \rfloor$ ka matlab hai Floor function (round down to integer).*

### Chalo Check Karein:
* Input = $5$, Padding = $0$, Kernel = $3$, Stride = $1$:
  $$H_{out} = \frac{5 + 0 - 3}{1} + 1 = 2 + 1 = 3 \quad (\text{Correct!})$$
* Input = $60$, Padding = $2$, Kernel = $5$, Stride = $2$:
  $$H_{out} = \lfloor \frac{60 + 4 - 5}{2} \rfloor + 1 = \lfloor \frac{59}{2} \rfloor + 1 = 29 + 1 = 30$$

Aap is formula ko achhe se samajh lijiye. Jab aap tayyar honge, tab batana, hum code level pe PyTorch CNN banana seekhenge!
