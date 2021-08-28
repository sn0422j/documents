---
title: 現代数理統計学の基礎 04 多次元確率変数の分布
---

## 同時確率分布と周辺分布

### 離散分布
$\mathcal{X}, \mathcal{Y}$ 上の2つの離散型確率変数 $X,Y$ において、 $X=x$ かつ $Y=y$ である確率 $P(\{X=x\} \cap \{Y=y\})$ を $P(X=x, Y=y)$ で表し、

$$
P(X=x, Y=y) = f_{X,Y}(x,y), \quad (x,y) \in \mathcal{X} \times \mathcal{Y}
$$

と表す。このとき、$C \subset \R^2$ に対して、$(X,Y) \in C$ となる確率は

$$
P((X,Y) \in C) = \sum_{(x,y) \in C} f_{X,Y}(x,y)
$$

となる。これを**同時分布**という。離散型確率変数の場合、$f_{X,Y}(x,y)$ を**同時確率関数**という。

$\mathcal{X}$ 上の集合 $A$ に対して $\{X \in A\}$ という事象は $\{ X \in A\} \cap \{ Y \in \mathcal{Y}\}$ もしくは $\{(X,Y) \in A \times \mathcal{Y}\}$ と同等なので、$P(X \in A)$ は

$$
\begin{aligned}
    P(X \in A) &= P((X,Y) \in A \times \mathcal{Y}) \\
    &= \sum_{(x,y) \in A \times \mathcal{Y}} f_{X,Y}(x,y)
\end{aligned}
$$

と書ける。これを $X$ の**周辺分布**といい、

$$
f_X(x) = \sum_{y=0}^\infty f_{X,Y}(x,y)
$$

を $X$ の**周辺確率関数**という。

関数 $g(X,Y)$ の同時確率関数に関する期待値は

$$
\mathbb{E} [g(X,Y)] = \sum_{x=0}^\infty \sum_{y=0}^\infty g(x,y) f_{X,Y}(x,y)
$$

で定義される。これを $\mathbb{E}^{X,Y} [\cdot]$ と書いたりする。

<details>
    <summary>例4.1</summary>
    
$$
\begin{aligned}
    (1)\ & \mathbb{E}[X] = 1 f_X(1) + 2 f_X(2) = 1.5 \\
    & \text{Var}[X] = (1-1.5)^2 f_X(1) + (2-1.5)^2 f_X(2) = 0.25 \\
    & \mathbb{E}[X] = -1 f_Y(-1) + 1 f_Y(1) = 0 \\
    & \text{Var}[Y] = (-1)^2 f_Y(-1) + 1^2 f_Y(1) = 0.6 \\
    (2)\ & P(X \ge Y+2) = P((X,Y) \in \{ (1,-1),(2,-1),(2,0)\}) \\
    & = f_{X,Y} (1,-1) + f_{X,Y} (2,-1) + f_{X,Y} (2,0) = 0.4 \\
    (3)\ & \mathbb{E}[XY] = \sum xy f_{X,Y}(x,y)\\
    & = -0.1+0.1-0.4+0.4 = 0
\end{aligned}
$$

</details><br />

### 連続分布
連続型確率変数 $X,Y \in \R$ に対して

$$
P((X,Y) \in C) = \iint_{(x,y) \in C} f_{X,Y}(x,y) dxdy
$$

と表されるとき、$f_{X,Y}(x,y)$ を**同時確率密度関数**という。

離散分布の時と同様に**周辺確率密度関数**が定義される。

$$
f_X(x) = \int_{-\infty}^\infty f_{X,Y}(x,y) dy
$$

2次元の分布関数とその微分が定義される。

$$
F_{X,Y}(x,y) = \int_{-\infty}^x \int_{-\infty}^y f_{X,Y}(s,t) dsdt \\[0.5em]
f_{X,Y}(x,y) = \frac{\partial^2}{\partial x \partial y} F_{X,Y}(x,y)
$$

また、関数 $g(X,Y)$ の同時確率密度関数に関する期待値は

$$
\mathbb{E} [g(X,Y)] = \int_{-\infty}^\infty \int_{-\infty}^\infty g(x,y) f_{X,Y}(x,y) dxdy
$$

<details>
    <summary>例4.2</summary>
    
$$
\begin{aligned}
    (1)\ & f_X(x) = \int_0^1 f_{X,Y}(x,y) dy = 0.5 + x \\
    & f_Y(y) = \int_0^1 f_{X,Y}(x,y) dx = 0.5 + y \\
    (2)\ & \mathbb{E}[Y] = \int_0^1  y f_{Y}(y) dy = \frac{7}{12} \\
    & \mathbb{E}[XY] = \int_0^1 \int_0^1 xy f_{X,Y}(x,y) dxdy \\
    &= \int_0^1 (\frac{1}{4}y+\frac{2}{3}y^2) dy = \frac{25}{72}
\end{aligned}
$$

</details><br />



## 条件付き確率分布と独立性
### 条件付き確率分布と条件付き期待値

<div style="border: 1px solid #000000; padding: 0.5em">

#### 定義 4.3
離散型確率変数 $(X,Y)$ が与えられているとする。  
$f_X(x)\ne0$ となる $x$ に対して、$X=x$ を与えた時の $Y=y$ の**条件付き確率関数**を
$$
f_{Y|X}(y|x) = P(Y=y|X=x) = \frac{f_{X,Y}(x,y)}{f_X(x)}
$$
で定義する。
</div><br>

$X=x$ を与えた時の $Y$ の**条件付き平均**（**期待値**）は

$$
\mathbb{E} [Y|X=x] = \sum_{y=0}^\infty y f_{Y|X}(y|x)
$$

で定義される。これを $\mathbb{E}^{Y|X} [\cdot|X=x]$ と表記する。また**条件付き分散**は

$$
\text{Var} [Y|X=x] = \mathbb{E}^{Y|X} [Y^2|X=x] - (\mathbb{E}^{Y|X} [Y|X=x])^2
$$

<details>
    <summary>途中式</summary>
    
$$
\begin{aligned}
    &\text{Var}[(Y-\mathbb{E}^{Y|X} [Y|X=x])^2|X=x]\\
    &= \text{Var}[Y^2 - 2Y\mathbb{E}_Y +  (\mathbb{E}_Y)^2|X=x] \quad (\because \mathbb{E}_Y = \mathbb{E}^{Y|X} [Y|X=x])\\
    &= \sum_y (y^2 - 2y\mathbb{E}_Y +  (\mathbb{E}_Y)^2)f_{Y|X} (y|x) \\
    &= \sum_y y^2 f_{Y|X} (y|x) - 2 \mathbb{E}_Y \sum_y y f_{Y|X} (y|x) + (\mathbb{E}^{Y|X})^2 \\
    &= \mathbb{E}^{Y|X} [Y^2|X=x] - 2 \mathbb{E}_Y \mathbb{E}_Y + (\mathbb{E}_Y)^2 \\
    &= \mathbb{E}^{Y|X} [Y^2|X=x] - (\mathbb{E}^{Y|X} [Y|X=x])^2
\end{aligned}
$$

</details><br />


条件付き期待値の条件に関してさらに期待値をとると、同時確率に関する期待値となる。

$$
\begin{aligned}
    \mathbb{E}^{X}[\mathbb{E}^{Y|X} [g(X(w),Y)|X(w)]]&= \sum_{x} \left(\sum_{y} g(x,y)  \frac{f_{X,Y}(x,y)}{f_X(x)}\right) f_X(x) \\
    &= \sum_{x} \sum_{y} g(x,y) f_{X,Y}(x,y) \\
    &= \mathbb{E}^{X,Y}[g(X,Y)]
\end{aligned}
$$

<details>
    <summary>例4.4</summary>
    
$$
\begin{aligned}
    (1)\ & P(Y=1|X=2) = 0.2 / 0.5 = 0.4 \\
    & P(X \gt Y+2|Y=0) = 0.1 / 0.4 = 0.25 \\
    (2)\ & \mathbb{E}[Y|X=2] = (-1) f_{Y|X} (-1|2) + 1 f_{Y|X} (1|2) \\
    &= -(0.2/0.5) + (0.2/0.5) = 0 \\
    & \text{Var}[Y|X=2] = \mathbb{E}^{Y|X} [Y^2|X=2] - (\mathbb{E}^{Y|X} [Y|X=2])^2 \\
    &= 2 \times (0.2/0.5) - 0^2 = 0.8
\end{aligned}
$$

</details><br />



<div style="border: 1px solid #000000; padding: 0.5em">

#### 定義 4.5
連続型確率変数 $(X,Y)$ が与えられているとする。  
$f_X(x)\gt0$ となる $x$ に対して、$X=x$ を与えた時の $Y=y$ の**条件付き確率密度関数**を
$$
f_{Y|X}(y|x) =\frac{f_{X,Y}(x,y)}{f_X(x)}
$$
で定義する。
</div><br>

条件付き期待値と条件付き分散は離散分布の時と同様に定義される。

<details>
    <summary>例4.6</summary>
    
$$
\begin{aligned}
    (1)\ & f_{Y|X} (y|x) = \frac{f_{X,Y}(x,y)}{\int f_{X,Y}(x,y) dy} \\
    &= (0.5+2xy)/(0.5+x) = (1+4xy)/(1+2x) \\
    (2)\ & \mathbb{E}[Y|X=x] = \int_0^1 y f_{Y|X} (y|x) dy \\
    &= (0.5+(4/3)x)/(1+2x) = (3+8x)/(6+12x) \\
    & \text{Var}[Y|X=x] = \int_0^1 (y- \mathbb{E}[Y|X=x])^2 f_{Y|X} (y|x) dy \\
    &= \frac{0.25+x+(2/3)x^2}{2(1+2x)^2} \\
    (3)\ & \mathbb{E}[\mathbb{E}[Y|X]] = \mathbb{E}[(3+8X)/(6+12X)] = \frac{7}{12} = \mathbb{E}[Y]
\end{aligned}
$$

</details><br />


### 確率変数の独立性
<div style="border: 1px solid #000000; padding: 0.5em">

#### 定義 4.7
確率変数 $(X,Y)$ の同時確率関数 $f_{X,Y}(x,y)$ と周辺確率関数 $f_{X}(x), f_{Y}(y)$ が与えられているとする。全ての $x \in \mathcal{X}$ と $x \in \mathcal{Y}$ に対して、
$$
f_{X,Y}(x,y) = f_{X}(x) f_{Y}(y)
$$
であるとき、$X$ と $Y$ は**独立**であるという。
</div><br>

$X$ と $Y$ が独立であれば、次が成立する。

$$
\mathbb{E} [g(X) h(Y)] = \mathbb{E} [g(X)] \cdot \mathbb{E} [h(Y)] 
$$

### 共分散と相関係数
2つの確率変数が独立でないとき、それらの関係を捉えるのに共分散と相関係数が役立つ。

$(X-\mu_X)(Y-\mu_Y)$ の期待値は変数の関係を捉えることができる。これを**共分散**といい、

$$
\sigma_{XY} = \text{Cov}[X,Y] = \mathbb{E} [(X-\mu_X)(Y-\mu_Y)]
$$

と書く。これに関連して**相関係数**を

$$
\rho_{XY} = \text{Corr}[X,Y] = \frac{\text{Cov}[X,Y]}{\sqrt{\text{Var}[X]}\sqrt{\text{Var}[Y]}} = \frac{\sigma_{XY}}{\sigma_{X}\sigma_{Y}}
$$

で定義する。共分散は尺度に依存するが、相関係数の絶対値は尺度に依存しない。$\| \text{Corr}[X,Y] \| \le 1$はコーシーシュバルツの不等式から導かれる。

<details>
    <summary>尺度不変性</summary>
    
$$
\begin{aligned}
    \text{Cov}[aX+b, cY+d] &= \mathbb{E} [a(X-\mu_X)c(Y-\mu_Y)+\alpha] = ac \text{Cov}[X,Y] \\
    \text{Corr}[aX+b, cY+d] &= \frac{ac\text{Cov}[X,Y]}{\sqrt{a^2\text{Var}[X]}\sqrt{c^2\text{Var}[Y]}} = \frac{ac}{\|ac\|} \text{Corr}[X, Y]
\end{aligned}
$$

</details><br />




$\text{Corr}[X,Y] \gt 0$ のとき**正の相関**、$\text{Corr}[X,Y] \lt 0$ のとき**負の相関**といい、$\text{Corr}[X,Y] = 0$ のとき**無相関**という。

> 「**独立ならば無相関**」は成り立つが、逆は必ずしも成り立たない。  
> $\text{Cov}[X,Y] = \mathbb{E}[XY] - \mathbb{E}[X]\mathbb{E}[Y] = \mathbb{E}[X]\mathbb{E}[Y] - \mathbb{E}[X]\mathbb{E}[Y] =0$

<details>
    <summary>例4.8</summary>
    
$X \sim \mathcal{N}(0,1), Y=X^2$ のとき独立では無いが無相関になる。

$\mathbb{E}[X^2] = M_X^{2}(0) = (\frac{d^2}{dt^2} \exp[t^2/2] )_{t=0} = 1, \ \mathbb{E}[X^3] = M_X^{3}(0) = 0$ を利用して

$$
\begin{aligned}
    \text{Cov}[X,Y] &= \mathbb{E}[X(Y-\mathbb{E}[Y])]= \mathbb{E}[X(X^2-\mathbb{E}[X^2])]\\
    &= \mathbb{E}[X^3-X] = \mathbb{E}[X^3] - \mathbb{E}[X] = 0
\end{aligned}
$$

</details><br />

$X$ と $Y$ の線形結合の分散は次のように表せる。

$$
\text{Var} [aX + bY] = a^2 \text{Var} [X] + b^2 \text{Var} [Y] + 2ab \text{Cov} [X,Y]
$$

### 階層モデルと混合分布
$$
(X|Y=y) \sim f_{X|Y} (x|y), \quad Y \sim f_Y(y)
$$

なる形で階層的に表すとき、このような構造をモデルを**階層モデル**という。

$X$ の周辺確率関数は、

$$
f_X(x) = \int f_{X|Y} (x|y) f_Y(y) d\mu_Y(y)
$$

となり、これを**混合分布**と呼ぶ。

<details>
    <summary>例4.9</summary>

$Y$ を離散型確率変数とし、$f_Y(i)=p_i, \ \sum_{i=1}^k p_i=1$ を満たすものとする。$f_i(x) = P(X|Y=i)$ とするとき、$X$ の周辺確率関数は

$$
f_X(x) = \sum_{i=1}^k f_Y(i) f_i(x) = \sum_{i=1}^k p_i f_i(x)
$$

という混合分布になる。$Y$ をグループとし、$p_i$ をあるグループ $i$ に所属する確率、$f_i(x)$ をそのグループにおける $X$ の分布として考えることに相当する。このとき、全体の分布は $f_X(x)$ となるわけである。

</details><br />

<details>
    <summary>例4.10</summary>

$Y$ を正の連続型確率変数とし、$X|Y = y \sim \mathcal{N}(\mu,y), \ Y \sim g(y), \ y \gt 0$ という階層モデルを考えると、$X$ の周辺確率密度関数 $f_X(x)$ は**正規尺度混合分布**と呼ばれる。

$$
\begin{aligned}
    f_X(x) &= \int f_{X|Y}(x|y) f_Y(y) dy \\
    &= \int_0^\infty \frac{1}{\sqrt{2 \pi y}} \exp \left[-\frac{(x-\mu)^2}{2y} \right] g(y) dy
\end{aligned}
$$

</details><br />

<div style="border: 1px solid #000000; padding: 0.5em">

#### 命題 4.11
$\text{Var}[X]$ は条件付き平均と条件付き分散を用いて次のように分解できる。
$$
\text{Var}[X] = \mathbb{E}[\text{Var}[X|Y]]+\text{Var}[\mathbb{E}[X|Y]]
$$



<details>
    <summary>証明（Ref.2）</summary>

$$
\begin{aligned}
    \text{Var}[X] &= \mathbb{E}[X^2] - (\mathbb{E}[X])^2 \\
    &= \mathbb{E}[\mathbb{E}^{X|Y}[X^2|Y]]-(\mathbb{E}[\mathbb{E}^{X|Y}[X|Y]])^2 &*1\\
    &= \mathbb{E}[\text{Var}[X|Y]+(\mathbb{E}^{X|Y}[X|Y])^2]-(\mathbb{E}[\mathbb{E}^{X|Y}[X|Y]])^2 &*2\\
    &= \mathbb{E}[\text{Var}[X|Y]]+\mathbb{E}[(\mathbb{E}^{X|Y}[X|Y])^2]-(\mathbb{E}[\mathbb{E}^{X|Y}[X|Y]])^2 \\
    &=  \mathbb{E}[\text{Var}[X|Y]]-\text{Var}[\mathbb{E}^{X|Y}[X|Y]]&*3
\end{aligned}\\[1em]
\begin{aligned}
    *1:\quad & \mathbb{E}^{X}[g(X)] = \mathbb{E}^{X}[\mathbb{E}^{X|Y}[g(X)|Y]]\\
    *2:\quad & \text{Var}[X|Y] = \mathbb{E}^{X|Y}[X^2|Y] - (\mathbb{E}^{X|Y}[X|Y])^2 \\
    *3:\quad & \text{Var}[\mathbb{E}^{X|Y}[X|Y]]=\mathbb{E}[(\mathbb{E}^{X|Y}[X|Y])^2]-(\mathbb{E}[\mathbb{E}^{X|Y}[X|Y]])^2 
\end{aligned}
$$

</details><br />
</div><br>


<div style="border: 1px dashed #32a1ce; padding: 0.5em">

#### 例 4.12
$$
(X|Y=i) \sim \mathcal{N}(\mu_i, \sigma_i^2), \quad Y \sim k^{-1}, \quad i=1, \dots , k
$$
という階層モデルにおいて、混合分布 $f_X(x)$ の平均と分散は
$$
\begin{aligned}
    \mathbb{E}[X] &= \sum_{i} Y \cdot \mathbb{E}[X|Y=i] = \sum_{i} k^{-1} \mu_i \equiv \bar{\mu} \\
    \text{Var}[X] &= \mathbb{E}[\text{Var}[X|Y]]+\text{Var}[\mathbb{E}[X|Y]]\\
    &= \sum_{i} Y \cdot \text{Var}[X|Y=i] + \text{Var}[\mu_i] \\
    &= \sum_{i} k^{-1} \sigma_i^2 + \sum_{i} k^{-1} (\mu_i - \bar{\mu})^2 \\
    &= k^{-1} \sum_{i} \{ \sigma_i^2 + (\mu_i - \bar{\mu})^2 \}
\end{aligned}
$$
と求められる。
</div><br>

<div style="border: 1px dashed #32a1ce; padding: 0.5em">

#### 例 4.14 ベータ・2項分布
$$
(X|Y) \sim Bin(n,Y), \quad Y \sim Beta(\alpha, \beta)
$$
という階層モデルにおいて、$X$ の周辺分布は**ベータ・2項分布**と呼ばれ、周辺確率関数は
$$
f_X(x) = \begin{pmatrix} n \\ x \end{pmatrix} \frac{B(x+\alpha,n-x+\beta)}{B(\alpha,\beta)}
$$
と書ける。
</div><br>

<div style="border: 1px dashed #32a1ce; padding: 0.5em">

#### 例 4.15 ガンマ・ポアソン分布
$$
(X|Y) \sim Po(Y), \quad Y \sim Ga(\alpha, \beta)
$$
という階層モデルにおいて、$X$ の周辺分布は**ガンマ・ポアソン分布**と呼ばれ、周辺確率関数は
$$
f_X(x) = \frac{\Gamma(x+\alpha)}{\Gamma(\alpha)x!} \frac{\beta^{x}}{(1+\beta)^{x+\alpha}}
$$
と書ける。
</div><br>

## 変数変換
### 変数変換の公式
$S=g_1(X,Y), T=g_2(X,Y)$ となる変数変換を考え、$(S,T)$ となる同時確率関数は

$$
P((S,T) \in D)  = P ((X,Y) \in C), \quad C= \{ (x,y)\ | (g_1(x,y), g_2(x,y)) \in D \}
$$

と表せる。

$(X,Y) \leftrightarrow (S,T)$ の対応が1対1であるときには、$X=h_1(S,T), Y=h_2(S,T)$ と陽にかける。このとき、**ヤコビアン**を利用して $(S,T)$ の同時確率密度関数は

$$
f_{S,T}(s,t) = f_{X,Y} (h_1(s,t),h_2(s,t)) \| J(s,t) \| \\[0.5em]
J = \det \begin{pmatrix}
    \frac{\partial }{\partial s} h_1(s,t) & \frac{\partial }{\partial t} h_1(s,t) \\
    \frac{\partial }{\partial s} h_2(s,t) & \frac{\partial }{\partial t} h_2(s,t)
\end{pmatrix}
$$

> ちなみに $J((s,t) \to (x,y)) = 1 / J((x,y) \to (s,t))$ となるため、やりやすい方向で計算すればよい。

<div style="border: 1px dashed #32a1ce; padding: 0.5em">

#### 例 4.17
独立な確率変数 $X \sim Ga(a,1), Y \sim Ga(b,1)$ に対し、$Z=X+Y, W=X/(X+Y)$ となる変数変換を考える。$x=zw, y=z(1-w)$ となるので、ヤコビアンは
$$
J((z,w) \to (x,y)) = \det \begin{pmatrix} w & z \\ 1-w & -z \end{pmatrix} = -z
$$
となるため、
$$
\begin{aligned}
    f_{Z,W}(z,w) &= f_X(zw) f_Y(z(1-w)) z \\
    &= \frac{1}{\Gamma(a+b)} (zw)^{a-1} (z(1-w))^{b-1} \exp [-z] z\\
    &= \frac{1}{\Gamma(a+b)} z^{a+b-1} \exp [-z] \times \frac{\Gamma(a+b)}{\Gamma(a)\Gamma(b)} w^{a-1} (1-w)^{b-1}
\end{aligned}
$$
と書けるため、$Z \sim Ga(a+b,1), W \sim Beta(a,b)$ に従う（独立なので積になる）。
</div><br>

<div style="border: 1px dashed #32a1ce; padding: 0.5em">

#### 例 4.18 ボックス - ミュラー変換
$U_1, U_2$ を $(0,1)$ 上の一様分布からの独立な確率変数とし、$r=\sqrt{-2\log U_1}, \theta = 2 \pi U_2$ とおく。このとき、$X=r\cos \theta, Y=r \sin \theta$ は独立に分布し、それぞれ標準正規分布に従う。これは、正規分布に従う乱数の発生に応用されている（Ref.3）。



<details>
    <summary>証明</summary>

書き換えると $u_1=\exp(-r^2/2),u_2=u_2/(2\pi)$ となるので、ヤコビアンは

$$
J = \det \begin{pmatrix}
    -r \exp(-r^2/2) & 0 \\ 0 & 1/(2\pi)
\end{pmatrix} = -\frac{r}{2\pi} \exp(-r^2/2)
$$

であるから、同時確率密度関数 $f_{r,\theta}(r,\theta)$ は

$$
f_{r,\theta}(r,\theta) = f_{U_1,U_2}(u_1,u_2) \|J\| = \frac{r}{2\pi} \exp(-r^2/2)
$$

となる。さらに、$x=r\cos \theta, y=r \sin \theta$ とおくと、極座標のヤコビアンは

$$
J = \frac{1}{r}
$$

であるから、同時確率密度関数 $f_{X,Y}(x,y)$ は

$$
\begin{aligned}
    f_{X,Y}(x,y) &= f_{r,\theta}(r,\theta) \|J\| = \frac{1}{2\pi} \exp(-r^2/2) = \frac{1}{2\pi} \exp \left[-\frac{x^2+y^2}{2}\right] \\[0.5em]
    &= \frac{1}{\sqrt{2\pi}} \exp \left[-\frac{x^2}{2}\right] \times \frac{1}{\sqrt{2\pi}} \exp \left[-\frac{y^2}{2}\right] 
\end{aligned}
$$    

</details><br>
</div><br>


### 確率変数の和の分布
$X \sim f_X(x), Y \sim f_Y(y)$ に対して、$Z=X+Y$ の分布を求めたい。**変数変換を用いる方法**と、**積率母関数・特性関数を用いる方法**がある。

1. 変数変換を用いる方法  
$Z=X+Y, T=Y$ となる変数変換を考えればよい。
$$
\begin{aligned}
    f_{Z,T} (z,t) &= f_X(z-t) f_Y(t) \\
    f_Z(z) &= \int f_X(z-t) f_Y(t) dt = f_X * f_Y \leftarrow たたみこみ
\end{aligned}
$$
2. 積率母関数・特性関数を用いる方法  
特性関数が $\phi_X(t) \phi_Y(t)$ となるような分布を見つけることができる。
$$
\varphi_Z(t) = \mathbb{E}[e^{itZ}] = \mathbb{E}[e^{itX}] \mathbb{E}[e^{itY}] = \phi_X(t) \phi_Y(t)
$$

<details>
    <summary>例4.19</summary>

$X,Y, i.i.d. \sim \mathcal{N}(0,1)$ のとき $Z=X+Y$ とすると、たたみこみを利用して

$$
\begin{aligned}
    f_Z(z) &= \int f_X(z-t) f_Y(t) dt\\
    &= \int \frac{1}{\sqrt{2\pi}} \exp \left[ - \frac{(z-t)^2}{2} \right] \frac{1}{\sqrt{2\pi}} \exp \left[ - \frac{t^2}{2} \right] dt \\[0.5em]
    &= \frac{1}{2\pi} \int \exp \left[ -\frac{(z-t)^2+t^2}{2} \right] dt \\[0.5em]
    &= \frac{1}{2\pi} \int \exp \left[ -\frac{2(t-z/2)^2 + z^2/2}{2} \right] dt \\[0.5em]
    &= \int \frac{\sqrt{2}}{\sqrt{2\pi}}  \exp \left[ -(t-z/2)^2 \right] dt \times \frac{1}{\sqrt{2\pi}\sqrt{2}} \exp \left[ -\frac{z^2}{4} \right] \\[0.5em]
    &= \int f_{\mathcal{N}} \left(t \middle|\frac{z}{2},\frac{1}{2} \right) dt \times \frac{1}{\sqrt{2\pi}\sqrt{2}} \exp \left[ -\frac{z^2}{4} \right] \\[0.5em]
    &= \frac{1}{\sqrt{2\pi}\sqrt{2}} \exp \left[ -\frac{z^2}{4} \right]
\end{aligned}
$$

したがって $Z \sim \mathcal{N}(0,2)$ となる。
</details><br>



<div style="border: 1px solid #000000; padding: 0.5em">

#### 命題 4.20
確率変数の和に関して、次の関係が成り立つ。ただし、左辺はそれぞれの分布に従う独立な2つの確率変数の和を意味し、右辺はその和の分布を意味するものとする。
$$
\begin{aligned}
    \mathcal{N}(\mu_X, \sigma_X^2) + \mathcal{N}(\mu_Y, \sigma_Y^2) &= \mathcal{N}(\mu_X + \mu_Y, \sigma_X^2 + \sigma_Y^2) \\
    Bin(m,p) + Bin(n,p) &= Bin(m+n,p) \\
    Po(\lambda_1) + Po(\lambda_2) &= Po(\lambda_1 + \lambda_2) \\
    Ga(\alpha_1, \beta) + Ga(\alpha_2, \beta) &= Ga(\alpha_1 + \alpha_2, \beta) \\
    \chi_m^2 + \chi_n^2 &= \chi_{m+n}^2  
\end{aligned}
$$
</div><br>

<div style="border: 1px solid #000000; padding: 0.5em">

#### 命題 4.21
互いに独立な確率変数 $Z_i \sim \mathcal{N}(0,1)$ について、$Z_1, \dots, Z_k$ の2乗和は自由度 $k$ のカイ2乗分布に従う。
すなわち、
$$
Z_1^2 + \cdots +  Z_k^2 \sim \chi_k^2
$$
となる。
</div><br>

## 多次元確率分布
### 多次元確率変数の分布
$k$ 個の確率変数の組を $(X_1, \dots, X_k)$ とし、実現値を $(x_1, \dots, x_k)$ とする。$i=1, \dots, k$ に対して、$X_i$ の標本空間の $\mathcal{X}_i$ とするとき、$(X_1, \dots, X_k)$ の標本空間は $\mathcal{X}_1 \times \cdots \times \mathcal{X}_k$ となる。

まず、離散型確率変数のとき、同時確率関数は

$$
f_{1, \dots k} (x_1, \dots, x_k) = P(X_1=x_1, \dots, X_k=x_k)
$$

であり、分布関数は

$$
F_{1, \dots k} (x_1, \dots, x_k) = \sum_{t_1 \ge x_1} \cdots \sum_{t_k \ge x_k} f_{1, \dots k} (x_1, \dots, x_k)
$$

である。周辺確率や期待値は総和を取ればよい。

$$
f_{1,2,3} (x_1, x_2, x_3) = \sum_{x_4 \in \mathcal{X}_4} \cdots \sum_{x_k \in \mathcal{X}_k} f_{1, \dots, k} (x_1, \dots, x_k) \\[0.5em]
\mathbb{E}[g(X_1, \dots, X_k)] = \sum_{x_1 \in \mathcal{X}_1} \cdots \sum_{x_k \in \mathcal{X}_k} g(x_1, \dots, x_k) f_{1, \dots, k} (x_1, \dots, x_k)
$$

<div style="border: 1px solid #000000; padding: 0.5em">

#### 定義 4.22
$k$ 次元離散型確率変数 $(X_1, \dots, X_k)$ の各周辺確率関数を $f_1(x_1), \dots, f_k(x_k)$ とする。全ての $x_1, \dots, x_k$ に対して
$$
f_{1, \dots, k} (x_1, \dots, x_k) = f_1(x_1) \times \cdots \times f_k(x_k)
$$
と書けるとき、$X_1, \dots, X_k$ は**互いに独立である**という。
</div><br>

連続型確率変数も同様に考える。

$$
F_{1, \dots, k} (x_1, \dots, x_k) = \int_{-\infty}^{x_1} \cdots \int_{-\infty}^{x_k} f_{1, \dots, k} (t_1, \dots, t_k) dt_1 \dots dt_k \\[0.5em]
\mathbb{E}[g(X_1, \dots, X_k)] = \int_{-\infty}^{\infty} \cdots \int_{-\infty}^{\infty} g(x_1, \dots, x_k) f_{1, \dots, k} (x_1, \dots, x_k) dx_1 \dots dx_k \\[0.5em]
f_1(x_1) = \int_{-\infty}^{\infty} \cdots \int_{-\infty}^{\infty} f_{1, \dots, k} (x_1, \dots, x_k) dx_2 \dots dx_k 
$$

互いに独立な場合は、次が成立する。

$$
f_{1, \dots, k} (x_1, \dots, x_k) = f_1(x_1) \times \cdots \times f_k(x_k)
$$

$k$ 次元の連続型独立変数に関する変数変換も同様に定義できる。

$$
J (\boldsymbol{y} \to \boldsymbol{x}) = \det \begin{pmatrix}
    (\partial / \partial y_1) h_1 (\boldsymbol{y}) & \cdots & (\partial / \partial y_k) h_1 (\boldsymbol{y}) \\
    \vdots & \ddots & \vdots \\
    (\partial / \partial y_1) h_k (\boldsymbol{y}) & \cdots & (\partial / \partial y_k) h_k (\boldsymbol{y}) \\
\end{pmatrix} \\[1em]
f_{Y_1, \dots, Y_k} (\boldsymbol{y}) = f_{X_1, \dots, X_k} (h_1(\boldsymbol{y}), \dots, h_k(\boldsymbol{y})) \| J(\boldsymbol{y} \to \boldsymbol{x}) \|
$$

### 多項分布

<div style="border: 1px solid #000000; padding: 0.5em">

#### 定義 4.23
$(X_1, \dots, X_k)$ の $(x_1, \dots, x_k)$ における同時確率関数が
$$
f_{1, \dots, k} (x_1, \dots, x_k | n, p_1, \dots, p_{k-1}) = \frac{n!}{x_1! \cdots x_k!} p_1^{x_1} \cdots p_k^{x_k}
$$
となる形で与えられるとき、これを**多項分布** $Multin_k(n,p_1, \dots, p_{k-1})$ という。
</div><br>

> 確率分布になることは、多項定理より確かめられる。  
> $1=(p_1 + \cdots +p_k)^n = \sum_{\mathcal{X}} \frac{n!}{x_1! \cdots x_k!}p_1^{x_1}\cdots p_k^{x_k}$

### 多変量正規分布

多次元分布として最もよく使われるのは多変量正規分布である。平均のベクトル $\boldsymbol{\mu}$ と分散共分散行列（共分散行列）$\Sigma$ を

$$
\begin{aligned}
    \boldsymbol{\mu} &= \begin{pmatrix}
    \mathbb{E}[X_i] \\ \vdots \\ \mathbb{E}[X_k]
    \end{pmatrix} = \begin{pmatrix}
        \mu_1 \\ \vdots \\ \mu_k
    \end{pmatrix} \\[2em]
    \Sigma &= \begin{pmatrix}
        \text{Var}[X_1] & \cdots & \text{Cov}[X_1, X_k] \\
        \vdots & \ddots & \vdots \\
        \text{Cov}[X_k, X_1] & \cdots & \text{Var}[X_k] \\
    \end{pmatrix} = \begin{pmatrix}
        \sigma_{11} & \cdots & \sigma_{1k} \\
        \vdots & \ddots & \vdots \\
        \sigma_{k1} & \cdots & \sigma_{kk} \\
    \end{pmatrix}
\end{aligned}

$$

<div style="border: 1px solid #000000; padding: 0.5em">

#### 定義 4.24
連続型確率変数 $\boldsymbol{X} = (X_1, \dots, X_k)^\top$ の $\boldsymbol{x} = (x_1, \dots, x_k)^\top$ における同時確率密度関数が
$$
f_{\boldsymbol{X}} (\boldsymbol{x} | \boldsymbol{\mu}, \Sigma) = \frac{1}{(2\pi)^{k/2}} \frac{1}{\|\Sigma\|^{1/2}} \exp \left[-\frac{1}{2} (\boldsymbol{x}-\boldsymbol{\mu})^\top \Sigma^{-1} (\boldsymbol{x}-\boldsymbol{\mu}) \right]
$$
となる形で与えられるとき、$\boldsymbol{X}$ は平均 $\boldsymbol{\mu}$、共分散行列 $\Sigma$ の**多変量正規分布** $\mathcal{N}_k(\boldsymbol{\mu}, \Sigma)$ に従うという
</div><br>

多変量正規分布では、無相関であることが独立であることの必要十分条件となる。

$k$ 次元の確率変数 $\boldsymbol{X}$ が多変量正規分布 $\mathcal{N}_k(\boldsymbol{\mu}, \Sigma)$ に従うとき、$\boldsymbol{X}$ の確率母関数 $M_X(\boldsymbol{t})$ と特性関数 $\varphi(\boldsymbol{t})$ は、$\boldsymbol{t}=(t_1, \dots, t_k)^\top$ に対して

$$
M_X(\boldsymbol{t}) = \mathbb{E}[e^{\boldsymbol{t}\boldsymbol{X}}] = \exp \left[\boldsymbol{\mu}^\top \boldsymbol{t} + \frac{1}{2}\boldsymbol{t}^\top \Sigma \boldsymbol{t}\right] \\[0.5em]
\varphi(\boldsymbol{t}) = \mathbb{E}[e^{i\boldsymbol{t}\boldsymbol{X}}] = \exp \left[i\boldsymbol{\mu}^\top \boldsymbol{t} - \frac{1}{2}\boldsymbol{t}^\top \Sigma \boldsymbol{t}\right] 
$$


## Reference
1. [久保川達也著 『現代数理統計学の基礎』 共立出版 2017年04月](https://www.kyoritsu-pub.co.jp/bookdetail/9784320111660)
2. [第1講 確率と確率変数 条件付き分散](http://www.data-arts.jp/course/probability/characteristic_values/conditional_variance.html)
3. [分析ノート - ボックス=ミュラー法](https://analytics-note.xyz/statistics/box-muller-method/)
