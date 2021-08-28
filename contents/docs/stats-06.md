---
title: 現代数理統計学の基礎 06 標本分布とその近似 (6.3,6.4)
---

## 推定量の評価
どの推定法を用いたらよいかを評価する方法を導入する。

### 不偏性
点推定量の望ましい性質の1つの不偏性がある。

<div style="border: 1px solid #000000; padding: 0.5em">

#### 定義 6.14 
推定量 $\hat{\theta}$ が $\theta$ の**不偏推定量**であるとは、すべての $\theta$ に対して次が成立することである。

$$
\mathbb{E}_{\theta}[\hat{\theta}(\boldsymbol{X})] = \theta
$$
</div><br>

不偏性とは、$\hat{\theta}$ が平均的に $\theta$ の周りに分布していることを指す。一般に $\hat{\theta}$ の**バイアス** $\text{Bias}$ は次のように定義され、不偏であるときにゼロとなる。

$$
\text{Bias}[\hat{\theta}] = \mathbb{E}_{\theta}[\hat{\theta}] - \theta
$$

バイアスが小さくて、$\theta$ の周りにバランス良く分布していても、散らばりが大きいと使いにくい。散らばりを評価するには**分散**を評価するとよい。

$$
\text{Var}[\hat{\theta}] = \mathbb{E}[\{ \hat{\theta} - \mathbb{E}[\hat{\theta}]\}^2]
$$

ここで $\hat{\theta}$ と $\theta$ の距離を**平均2乗誤差** MSE で測ることを考える。

$$
\text{MSE}[\hat{\theta},\theta] = \mathbb{E}[\{ \hat{\theta} - \theta\}^2]
$$

MSEは次のように変形できる。したがって、平均2乗誤差は分散とバイアスの2乗の平均であり、両方を評価する基準である。また、不偏推定量の平均2乗誤差は分散と一致する。

$$
\begin{aligned}
    \text{MSE}[\hat{\theta},\theta] &= \mathbb{E}[\{ (\hat{\theta} - \mathbb{E}[\hat{\theta}]) + (\mathbb{E}[\hat{\theta}] - \theta)\}^2]\\
    &= \mathbb{E}[(\hat{\theta} - \mathbb{E}[\hat{\theta}])^2]+ \mathbb{E}[(\mathbb{E}[\hat{\theta}] - \theta)^2]+2\mathbb{E}[(\hat{\theta} - \mathbb{E}[\hat{\theta}])(\mathbb{E}[\hat{\theta}] - \theta)]\\
    &= \text{Var}[\hat{\theta}] + ( \text{Bias}[\hat{\theta}] )^2 + 0
\end{aligned}
$$

<div style="border: 1px dashed #32a1ce; padding: 0.5em">

#### 例 6.15 

平均 $\mu$、分散 $\sigma^2$ の母集団からの無作為標本 $\{X_i\}$ を考える。$\mu$ の推定量として線形推定量 $\hat{\mu}_c=\sum_{i=1}^nc_iX_i$ を利用する。その期待値は

$$
\mathbb{E}[\hat{\mu}_c]=\mathbb{E}\left[\sum_{i=1}^nc_iX_i\right] = \sum_{i=1}^nc_i\mathbb{E}\left[X_i\right] = \sum_{i=1}^n c_i \mu
$$

と書けるので、バイアスは 

$$
\text{Bias}[\hat{\mu}_c] = \mathbb{E}[\hat{\mu}_c]- \mu = \left(\sum_{i=1}^n c_i -1\right)\mu
$$

となるため、線形推定量が不偏になるためには $\sum_{i=1}^n c_i=1$ を満たす必要がある。また、分散は

$$
\text{Var}[\hat{\mu}_c] = \mathbb{E}[(\hat{\mu}_c-\mathbb{E}[\hat{\mu}_c])^2]= \mathbb{E}\left[ \left\{\sum_{i=1}^n c_i (X_i-\mu)\right\}^2\right] = \sum_{i=1}^n c_i^2 \sigma^2
$$

> 1乗の項は期待値とって消える。

となる。不偏な線形推定量のうち、分散を最小にするものを**最良線形不偏推定量**（**BLUE**）といい、$\sum_{i=1}^n c_i=1$ の条件下で $\sum_{i=1}^n c_i^2$ を最小化することになる。

これは制約付き最適化問題であり、ラグランジュの未定乗数法を用いて解ける。

$$
H = \sum_{i=1}^n c_i^2 - \lambda \left(\sum_{i=1}^n c_i-1\right) \\[0.5em]
\frac{\partial H}{\partial c_i} = 2c_i - \lambda = 0 \quad (i=1,\dots,n) \\[0.5em]
\therefore \ c_1 = \cdots = c_n =\frac{\lambda}{2} =const \\[0.5em]
\therefore \ c_i = \frac{1}{n} \quad (i=1,\dots,n) \\
$$

したがって、$\mu$ の BLEU は $\hat{\mu}_{BLEU} = \sum_{i=1}^n \frac{1}{n} X_i = \bar{X}$ と平均に一致する。

</div><br>

<div style="border: 1px dashed #32a1ce; padding: 0.5em">

#### 例 6.16 

正規母集団 $\mathcal{N}(\mu,\sigma^2)$ からの無作為標本 $X_1,\dots,X_n$ を考える。$Q=\sum_{i=1}^n(X_i-\bar{X})^2$ とおくとき、不偏分散 $\hat{\sigma}_U^2=\frac{1}{n-1}Q$ は $\sigma^2$ の不偏推定量となる。一方、$\sigma^2$ の最尤推定量は標本分散 $\hat{\sigma}_M^2=\frac{1}{n}Q$ で与えられる。これらの違いを推定する。定理 5.1 より $Q/\sigma^2 \sim \chi_{n-1}^2$ であるから、カイ二乗分布の期待値について $\mathbb{E}[Q/\sigma^2]=n-1, \mathbb{E}[(Q/\sigma^2)^2]=(n-1)(n+1)$ となる。したがって、バイアス・分散・平均2乗誤差を計算できる。

$$
\begin{aligned}
    \text{Bias}[\hat{\sigma}_U^2] &= \mathbb{E}[\hat{\sigma}_U^2] - \sigma^2 = 0 \\
    \text{Bias}[\hat{\sigma}_M^2] &= \mathbb{E}[\hat{\sigma}_M^2] - \sigma^2 = \frac{n-1}{n} \sigma^2 -\sigma^2 = -\sigma^2/n \\[0.7em]
    \text{Var}[\hat{\sigma}_U^2] &= \mathbb{E}[\{\hat{\sigma}_U^2\}^2] - \{ \mathbb{E}[\hat{\sigma}_U^2]\}^2 \\
    &= \mathbb{E}\left[Q^2/(n-1)^2\right] - (\sigma^2)^2 \\
    &= \sigma^4(n-1)(n+1)/(n-1)^2 - \sigma^4 \\
    &= \frac{2\sigma^4}{(n-1)} \\
    \text{Var}[\hat{\sigma}_M^2] &= \mathbb{E}[\{\hat{\sigma}_M^2\}^2] - \{ \mathbb{E}[\hat{\sigma}_M^2]\}^2 \\
    &= \mathbb{E}\left[Q^2/n^2\right] - (\sigma^2)^2 \\
    &= \sigma^4(n-1)(n+1)/n^2 - \sigma^4 \\
    &= \frac{2\sigma^4(n-1)}{n^2} = \frac{2\sigma^4}{(n-1)} \left(\frac{n-1}{n}\right)^2 \lt \text{Var}[\hat{\sigma}_U^2] \\[0.7em]
    \text{MSE}[\hat{\sigma}_U^2] &= \text{Var}[\hat{\sigma}_U^2] + ( \text{Bias}[\hat{\sigma}_U^2] )^2 = \frac{2\sigma^4}{(n-1)} \\
    \text{MSE}[\hat{\sigma}_M^2] &= \text{Var}[\hat{\sigma}_M^2] + ( \text{Bias}[\hat{\sigma}_M^2] )^2 \\
    &= \frac{2\sigma^4(n-1)}{n^2} +\frac{\sigma^4}{n^2} = \frac{(2n-1)\sigma^4}{n^2} \\
    &= \frac{2\sigma^4}{(n-1)} \frac{2n^2-3n+1}{2n^2} \lt \text{MSE}[\hat{\sigma}_U^2]
\end{aligned}
$$

まとめると、$\hat{\sigma}_U^2$ は不偏推定量（バイアスがゼロ）であるが、分散・平均2乗誤差は $\hat{\sigma}_M^2$ より大きくなってしまうといえる。
</div><br>

### フィッシャー情報量とクラメール・ラオの不等式
不偏推定量における分散には下限が存在し、フィッシャー情報量によって与えられる。

$\boldsymbol{X} \sim f(x|\theta)$ の同時確率（密度）関数 $f_n(\boldsymbol{x}|\theta) = \prod_{i=1}^n f(x_i|\theta)$ について、**スコア関数**を

$$
S_n(\theta,\boldsymbol{X}) = \frac{d}{d\theta} \log f_n(\boldsymbol{x}|\theta)
$$

と定義すると、その2乗の期待値が**フィッシャー情報量**となる。

$$
I_n(\theta) = \mathbb{E}[S_n^2] = \mathbb{E}\left[\left\{\frac{d}{d\theta} \log f_n(\boldsymbol{x}|\theta)\right\}^2\right]
$$


フィッシャー情報量の定義と性質を導くために、次の条件を仮定する。

- (C1) $f(x|\theta)$ の台 $\{ x|f(x|\theta) \gt 0\}$ は $\theta$ に依存しない。
- (C2) $f(x|\theta)$ は $\theta$ に関して2階微分可能で、$\int f(x|\theta) dx$ の2回までの微分は積分と可換である。
- (C3) フィッシャー情報量 $I_1(\theta)$ に対して $0\lt I_1(\theta) \lt \infty$ とする。

> フィッシャー情報量の定義のみであれば (C1) と $f(x|\theta)$ の微分可能性、$I_1(\theta) \lt \infty$ の条件だけで十分。

<div style="border: 1px solid #000000; padding: 0.5em">

#### 命題 6.17
(C1), (C2), (C3) を仮定したとき、次の性質が成り立つ。

1. $\mathbb{E}[S_1(\theta,X_i)]=0$
2. $I_n(\theta)=nI_1(\theta)$
3. $I_1(\theta) = -\mathbb{E}\left[\frac{d^2}{d\theta^2} \log f(X_i|\theta)\right]$

<details>
    <summary>証明</summary>

(1) $S_1(\theta,x)=\frac{f'(x|\theta)}{f(x|\theta)}$ と書けるので、

$$
\begin{aligned}
    \mathbb{E}[S_1(\theta,X_i)] &= \int \frac{f'(x|\theta)}{f(x|\theta)} f(x|\theta) dx \\
    &= \int\frac{d}{d\theta}f(x|\theta) dx = \frac{d}{d\theta} \int f(x|\theta) dx = \frac{d}{d\theta} (1) = 0
\end{aligned}
$$

(2) 同時分布から

$$
S_n(\theta|\boldsymbol{X}) = \frac{d}{d\theta} \log \left[ \prod_{i=1}^n f(x_i|\theta)\right] = \sum_{i=1}^n \frac{d}{d\theta} \log f(x_i|\theta) = \sum_{i=1}^n S_1(\theta|X_i)
$$

であるから、(1) の結果を利用して

$$
\begin{aligned}
    I_n(\theta) &= \mathbb{E}[S_n^2] 
    = \mathbb{E}\left[ \left\{ \sum_{i=1}^nS_1(\theta|X_i) \right\}^2 \right] \\
    &= \sum_{i=1}^n \mathbb{E}\left[ \left\{ S_1(\theta|X_i) \right\}^2 \right] + \sum_{i=1}^n \sum_{j=1,j\ne i}^n \mathbb{E}\left[ S_1(\theta|X_i) \right] \mathbb{E}\left[ S_1(\theta|X_j) \right] \\
    &= nI_1(\theta)
\end{aligned}
$$

(3) $\frac{d^2}{d\theta^2} \log f(X_i|\theta)$ について

$$
\begin{aligned}
    \frac{d^2}{d\theta^2} \log f(X_i|\theta) &= \frac{d}{d\theta} \frac{f'(X_i|\theta)}{f(X_i|\theta)} \\
    &= \frac{f''(X_i|\theta)}{f(X_i|\theta)} - \left\{\frac{f'(X_i|\theta)}{f(X_i|\theta)} \right\}^2 \\
    &= \frac{f''(X_i|\theta)}{f(X_i|\theta)} - \left\{S_1(\theta,x) \right\}^2
\end{aligned}
$$

ここで、(1) より1階微分の期待値が 0 だったので、2階微分の期待値も 0 になる。したがって、期待値をとると右辺の第1項は 0 となる。よって、

$$
I_1(\theta) = \mathbb{E}[\left\{S_1(\theta,x) \right\}^2] = -\mathbb{E}\left[\frac{d^2}{d\theta^2} \log f(X_i|\theta)\right]
$$
</details><br>
</div><br>

<div style="border: 1px solid #000000; padding: 0.5em">

#### 定理 6.18 クラメール・ラオの不等式
(C1), (C2), (C3) を仮定し、不変推定量 $\hat{\theta}$ の分散が存在して、微積分の可換性 $\frac{d}{d\theta} \int \hat{\theta}f_n(\boldsymbol{x}|\theta)d\boldsymbol{x} =  \int \hat{\theta} \frac{d}{d\theta} f_n(\boldsymbol{x}|\theta)d\boldsymbol{x}$ を仮定する。このとき任意の $\theta$ に対して

$$
\text{Var}_{\theta}[\hat{\theta}] \ge \frac{1}{I_n(\theta)}
$$

という不等式が成立する。これを**クラメール・ラオの不等式**といい、右辺をクラメール・ラオの下限という。

<details>
    <summary>証明</summary>

コーシー・シュワルツの不等式 $(\mathbb{E}[fg])^2 \le \mathbb{E}[f^2]\mathbb{E}[g^2]$ を用いると、

$$
\{\mathbb{E}[(\hat{\theta}-\theta)S_n] \}^2 \le \mathbb{E}[(\hat{\theta}-\theta)^2] \times \mathbb{E}[S_n^2] = \text{Var}_{\theta}[\hat{\theta}] \times I_n(\theta) 
$$

となるから、変形して

$$
\text{Var}_{\theta}[\hat{\theta}] \ge \frac{\{\mathbb{E}[(\hat{\theta}-\theta)S_n] \}^2}{I_n(\theta)}
$$

となる。ここで、$\mathbb{E}[(\hat{\theta}-\theta)S_n]$ について、$\mathbb{E}[S_n] = 0$ に気を付けると、

$$
\begin{aligned}
    \mathbb{E}[(\hat{\theta}-\theta)S_n] &= \mathbb{E}[\hat{\theta}S_n] - \theta\mathbb{E}[S_n] = \int \hat{\theta} \frac{d}{d\theta} f_n(\boldsymbol{x}|\theta) d\boldsymbol{x}\\
    &= \frac{d}{d\theta} \int \hat{\theta} f_n(\boldsymbol{x}|\theta) d\boldsymbol{x} = \frac{d}{d\theta} \mathbb{E}[\hat{\theta}] = \frac{d}{d\theta} \theta = 1
\end{aligned}
$$

となるため、上記の不等式が成立する。
</details><br>
</div><br>

<details>
    <summary>例 6.19</summary>

$X_i \sim \mathcal{N}(\mu,1)$ についてフィッシャー情報量は

$$
\begin{aligned}
    I_1(\mu) &= -\mathbb{E}\left[\frac{d^2}{d\mu^2} \log f(X_1|\mu)\right] \\
    &= -\mathbb{E}\left[\frac{d^2}{d\mu^2} \left(-\frac{1}{2} \log(2\pi)-\frac{1}{2}(X_i-\mu)^2\right)\right] = -\mathbb{E}[-1] = 1
\end{aligned}
$$

であるから、クラメール・ラオの下限は $\frac{1}{n}$ となり、標本平均の分散 $\text{Var}[\bar{X}]=\frac{1}{n}$ に一致する。すなわち、$\bar{X}$ は不偏推定量の中で、分散を最小にする推定量である。
</details><br>




### 最尤推定量の漸近的な性質
標本のサイズ $n$ が大きいときには推定量の漸近的な性質が考えられる。


<div style="border: 1px solid #000000; padding: 0.5em">

#### 定義 6.20 
$\theta$ の推定量 $\hat{\theta}_n$ が**一致性**をもつとは、$\hat{\theta}_n$ が $\theta$ に確率収束すること、すなわち任意の $\varepsilon \gt 0$ に対して以下が成り立つことである。

$$
\lim_{n\to\infty} P_{\theta}(\|\hat{\theta}_n-\theta\| \ge \varepsilon) = 0
$$
</div><br>

チェビシェフの不等式 $P(\|X-\mu\|\ge k) \le \frac{\sigma^2}{k^2}$ を用いると

$$
P(\|\hat{\theta}_n-\theta\|\ge \varepsilon) \le \frac{\mathbb{E}[\{ \hat{\theta}_n - \theta\}^2]}{\varepsilon^2} = \frac{\text{Var}[\hat{\theta}_n] + ( \text{Bias}[\hat{\theta}_n] )^2}{\varepsilon^2}
$$

と書けるから、分散とバイアスが共にゼロに収束するならば、推定量は一致性をもつ。

<div style="border: 1px solid #000000; padding: 0.5em">

#### 命題 6.21
$a_n$ を $a_n \uparrow \infty$ という数列とし、$a_n(\hat{\theta}_n-\theta) \to_d U$ という確率変数 $U$ が存在するとき、$\hat{\theta}_n$ は一致性をもつ。

<details>
    <summary>証明</summary>

スラツキーの定理より、$(\hat{\theta}_n-\theta) = \frac{1}{a_n} a_n(\hat{\theta}_n-\theta) \to_d 0 \cdot U=0$ となる。したがって、定数に分布収束するならば確率収束するので、$\hat{\theta}_n\to_p\theta$ となって一致性をもつ。
</details><br>
</div><br>

一般に、$\sqrt{n}(\hat{\theta}_n-\theta) \to_d \mathcal{N}(0,\sigma^2)$ となるとき、$\sigma^2$ を**漸近分散**という。これを評価した性質がある。

<div style="border: 1px solid #000000; padding: 0.5em">

#### 定義 6.22 
$\theta$ の推定量 $\hat{\theta}_n$ の漸近分散が $\frac{1}{I_1(\theta)}$ のとき、すなわち

$$
\sqrt{n}(\hat{\theta}_n-\theta) \to_d \mathcal{N}\left(0,\ \frac{1}{I_1(\theta)}\right)
$$

となるとき、$\hat{\theta}_n$ は**漸近有効**であるという。
</div><br>

漸近有効性は、漸近分散が下限に達していることを意味する。実際、不変推定量に対して、クラメール・ラオの不等式は

$$
\text{Var}[\hat{\theta}_n] \ge \frac{1}{nI_1(\theta)}
$$

となっていて、極限分散 $\lim_{n\to\infty} n\text{Var}[\hat{\theta}_n]$ の下限が $\frac{1}{I_1(\theta)}$ となっている。


次に、最尤推定量（MLE）の一致性（クラメールの一致性）と漸近正規性および漸近有効性を示す。そのために、次のような正則条件を仮定する。

$X_1,\dots,X_n \sim f(x|\theta)$ という無作為標本について、(C1), (C2) に加えて以下の条件を仮定する。

- (C4) パラメーター $\theta$ は**識別可能**である。すなわち、$\theta\ne\theta'$ ならば $f(x|\theta)\ne f(x|\theta')$ である。
- (C5) パラメーターの真の値 $\theta_0$ がパラメーター空間の内点にある。すなわち、パラメーター空間に含まれる $\theta_0$ の開近傍がとれる。

<div style="border: 1px solid #000000; padding: 0.5em">

#### 定理 6.23 一致性 
正則条件 (C1), (C2), (C4), (C5) を仮定する。このとき、尤度方程式 $\frac{d}{d\theta}L(\theta|\boldsymbol{x})=0$ はパラメーターの真の値 $\theta_0$ に確率収束する解を含む。

<details>
    <summary>簡単な証明</summary>

対数尤度関数は

$$
\log L(\theta) = \sum_{i=1}^n \log f(X_i,\theta)
$$

となるが、大数の法則 $\frac{1}{n} \sum_{i=1}^nY_i \to_p \mathbb{E}[Y_1]$ を利用すると

$$
\begin{aligned}
    \frac{1}{n} \log L(\theta) &= \frac{1}{n} \sum_{i=1}^n \log f(X_i,\theta) \\
    &\to_p \mathbb{E}_{\theta_0}[\log f(X_1,\theta)] = \int \{\log f(x,\theta)\} f(x,\theta_0) dx 
\end{aligned}
$$

となる。ここで、$X_i$ は真の値 $\theta_0$ による分布 $f(x,\theta_0)$ に従っていることに注意する。さらに KL情報量 $KL(f(x,\theta_0),f(x,\theta))$ を考えると

$$
KL(f(x,\theta_0),f(x,\theta)) = \int f(x,\theta_0) \log \frac{f(x,\theta_0)}{f(x,\theta)} dx \ge 0
$$

が成り立つ（イェンセンの不等式によって証明できる）。したがって、

$$
\int \{\log f(x,\theta_0)\} f(x,\theta_0)  dx - \int \{\log f(x,\theta)\} f(x,\theta_0) \ge 0 \\
\iff \int \{\log f(x,\theta)\} f(x,\theta_0)  dx \le \int \{\log f(x,\theta_0)\} f(x,\theta_0)
$$

といえる。つまり、$\int \{\log f_\theta \}f_\theta dx$ を最大にする $\theta$ は識別性の下で $\theta_0$ となる。したがって、最尤推定量 $\hat{\theta}$ は $\frac{1}{n} \log L(\theta)$ を最大にし、それは $\int \{\log f_\theta \}f_\theta dx$ を最大にする $\theta$ すなわち $\theta_0$ へ確率収束するため、$\hat{\theta} \to_p \theta_0$ が示される（内点の仮定によって極大点の一致が保証される）。
</details><br>
</div><br>

最尤推定量の漸近正規性を示すために次の正則条件を仮定する。

- (C6) $f(x|\theta)$ が $\theta$ に関して3回連続微分可能とする。また、正の実数 $c$ と関数 $M(x)$ が存在して、$\theta_0-c\lt\theta\lt\theta_0+c$ なるすべての $\theta$ に対して $\|\frac{d^3}{d\theta^3} \log f(x|\theta) \| \le M(x)$ であり、$\mathbb{E}_{\theta}[M(X_1)]\lt\infty$ を満たすものとし、さらに $M(x)$ は $\theta_0,c$ に依存してもよいが、$\theta$ に依存しないものとする。

<div style="border: 1px solid #000000; padding: 0.5em">

#### 定理 6.24 漸近正規性
正則条件 (C1) から (C6) を仮定する。このとき、MLE $\hat{\theta}$ について

$$
\sqrt{n}(\hat{\theta}_n-\theta) \to_d \mathcal{N}\left(0,\ \frac{1}{I_1(\theta)}\right)
$$

が成り立つことを**漸近正規性**という。漸近分散がクラメール・ラオの下限に達しているため、この推定量は漸近有効である。

<details>
    <summary>証明</summary>

対数尤度関数 $l(\theta) = \log L({\theta})$ について $l'(\hat{\theta})$ を $\hat{\theta}=\theta_0$ の周りでテーラー展開すると、

$$
l'(\hat{\theta}) = l'(\theta_0) + (\hat{\theta}-\theta_0) l''(\theta_0) + \frac{1}{2} (\hat{\theta}-\theta_0)^2 l'''(\theta^*)
$$

と近似できる。ここで、$\theta^*$ は $\hat{\theta}$ と $\theta_0$ の間の点である。$l'(\hat{\theta})=0$ より

$$
\sqrt{n}(\hat{\theta}-\theta_0) = \frac{(1/\sqrt{n})l'(\theta_0)}{-(1/n)l''(\theta_0)-(1/2n)(\hat{\theta}-\theta_0)l'''(\theta^*)}
$$

と変形できる。スコア関数 $S_1(\theta,X_i)$ を利用すると $l'(\theta_0)=\sum_{i=1}^nS_1(\theta,X_i)$ であり、$S_1(\theta,X_1), \dots, S_1(\theta,X_n) \ i.i.d. \ \sim (0,I_1(\theta_0))$ であるから、中心極限定理を利用して

$$
(1/\sqrt{n})l'(\theta_0) \to_d \mathcal{N}(0,I_1(\theta_0))
$$

となる。また、大数の法則 $\frac{1}{n} \sum_{i=1}^nY_i \to_p \mathbb{E}[Y_1]$ より

$$
-(1/n)l''(\theta_0) \to_p -\mathbb{E}_{\theta_0}[(d^2/d\theta^2)\log f(X_i|\theta)] = I_1(\theta_0)
$$

となる。最後に、仮定 (C6) と大数の法則より

$$
|(1/n)l'''(\theta^*)| \lt (1/n) \{ M(X_1) + \cdots + M(X_n)\} \to_p \mathbb{E}[M(X_1)]
$$

となって $(1/n)l'''(\theta^*)$ は確率有界である。したがって、一致性から $\hat{\theta}-\theta_0 \to_p 0$ なので $(1/n)(\hat{\theta}-\theta_0)l'''(\theta^*) \to_p 0$ である。以上より、スラツキーの定理を用いると

$$
\sqrt{n}(\hat{\theta}-\theta_0) \to_d \frac{1}{I_1(\theta)} \mathcal{N}(0,I_1(\theta)) = \mathcal{N}\left(0,\ \frac{1}{I_1(\theta)}\right)
$$
</details><br>
</div><br>

> 正則条件下であれば、最尤推定量は漸近正規性と一致性を持つ。しかし、非正則条件下ではこれらが成り立たないことに注意する。

定理 6.24 の拡張との1つとして、関数 $h(\theta)$ の最尤推定量 $h(\hat{\theta})$ の漸近分布が挙げられる。この場合、デルタ法を用いれば、$h'(\theta)\ne0$ である $\theta$ について次が成立する。

$$
\sqrt{n}(h(\hat{\theta})-h(\theta)) \to_d \mathcal{N}\left(0,\ \frac{\{h'(\theta)\}^2}{I_1(\theta)}\right)
$$

<details>
    <summary>デルタ法（定理5.20）のおさらい</summary>

連続微分可能な関数 $g(\cdot)$ について、点 $\theta$ で $g'(\theta)$ が存在し $g'(\theta) \ne 0$ とする。$\sqrt{n}(U_n - \mu) \to_d \mathcal{N}(0,\sigma^2)$ が成り立つとき、

$$
\sqrt{n}(g(U_n) - g(\mu)) \to_d \mathcal{N}(0,\{g'(\mu)\}^2\sigma^2) \quad (5.14)
$$
</details><br>

## 発展的事項
### 経験ベイズと階層ベイズ
ベイズ推定のハイパーパラメーター $\boldsymbol{\xi}$ の設定方法について補足する。

(1) **主観ベイズ法**  
ハイパーパラメーター $\boldsymbol{\xi}$ を**既知の値**として扱うことで、ベイズ推定量はハイパーパラメーターの影響を受ける。パラメーターを事後分布で平均することによって、推定量を得る。

$$
\begin{aligned}
    \text{Marginal: } & f_\pi(\boldsymbol{x}|\boldsymbol{\xi}) = \int f(\boldsymbol{x}|\boldsymbol{\theta})\pi(\boldsymbol{\theta}|\boldsymbol{\xi})d\boldsymbol{\theta} \\
    \text{Posterior: } & \pi(\boldsymbol{\theta}|\boldsymbol{x},\boldsymbol{\xi}) = \frac{f(\boldsymbol{x}|\boldsymbol{\theta})\pi(\boldsymbol{\theta}|\boldsymbol{\xi})}{f_\pi(\boldsymbol{x}|\boldsymbol{\xi})} \\
    \text{Estimate: } & \boldsymbol{\theta}^B(\boldsymbol{\xi}) = \mathbb{E}[\boldsymbol{\theta}|\boldsymbol{X},\boldsymbol{\xi}] = \int \boldsymbol{\theta}\pi(\boldsymbol{\theta}|\boldsymbol{X},\boldsymbol{\xi}) d\boldsymbol{\theta}    
\end{aligned}
$$

(2) **経験ベイズ法**  
ハイパーパラメーター $\boldsymbol{\xi}$ を**未知のパラメーター**として扱い、それを $\boldsymbol{X}$ の周辺分布から推定し、得られた推定量 $\hat{\boldsymbol{\xi}}$ をベイズ推定量 $\boldsymbol{\theta}^B(\boldsymbol{\xi})$ に代入することで推定量を得る。

> 「第2種の最尤推定」や「エビデンス近似」とも呼ばれている。

(3) **階層ベイズ法**  
ハイパーパラメーター $\boldsymbol{\xi}$ を**確率変数**として扱う。その分布を $\pi_2(\boldsymbol{\xi})$ とすれば、$\boldsymbol{\theta}$ の事前分布は次のような階層性をもつ。

$$
\begin{aligned}
    \boldsymbol{\theta}|\boldsymbol{\xi} &\sim \pi_1(\boldsymbol{\theta}|\boldsymbol{\xi}) \\
    \boldsymbol{\xi} &\sim \pi_2(\boldsymbol{\xi})
\end{aligned}
$$

$\boldsymbol{\theta}$ の周辺分布 $\pi(\boldsymbol{\theta}) = \int \pi_1(\boldsymbol{\theta}|\boldsymbol{\xi}) \pi_2(\boldsymbol{\xi}) d\boldsymbol{\xi}$ を周辺化 $\int \pi(\boldsymbol{\theta}) d\boldsymbol{\theta}$ した際に、これが有限となるものを**正則**（proper）であるといい、発散してしまうものを**非正則**（improper）であるという。非正則な場合でもベイズ推定量を求めたものは、**一般化ベイズ推定量**と呼ばれている。

〇 $\boldsymbol{\xi}$ に関して客観性を持たせるために、2段階目はより曖昧な分布を与える必要がある？？？？？

$\to$ 全て恣意的に統計モデルを立てているので、事前分布に客観性の程度などという観点は必要ない。考えうる統計モデルと事前分布の中で、最もデータを説明できると評価したものを選べばよい。

> 『主観的にモデリングされた統計モデルと事前分布を客観的に評価する方法を作る，という問題が現代の 統計学の重要な課題です．』 - [事前分布について](http://watanabe-www.math.dis.titech.ac.jp/users/swatanab/prior.html)

### 多次元への拡張
パラメーターが多次元の場合を考える。$\boldsymbol{\theta} = (\theta_1, \dots, \theta_k)^\top$ とし、$X_1,\dots,X_n$ を $f(x|\boldsymbol{\theta})$ からの無作為標本とすると、**フィッシャー情報量行列**が定義される。

$$
\boldsymbol{I} \ni I_{ij} = \mathbb{E}\left[\left(\frac{\partial}{\partial \theta_i} \log f(X_1|\boldsymbol{\theta})\right)\left(\frac{\partial}{\partial \theta_j} \log f(X_1|\boldsymbol{\theta})\right)\right]
$$

クラメール・ラオの不等式は、不変推定量 $\hat{\boldsymbol{\theta}}$ の共分散行列 $\text{Cov}(\boldsymbol{\theta})=\mathbb{E}[(\hat{\boldsymbol{\theta}}-\boldsymbol{\theta})(\hat{\boldsymbol{\theta}}-\boldsymbol{\theta})^\top]$ を用いて表現できる。

$$
\text{Cov}(\boldsymbol{\theta}) \ge \{n \boldsymbol{I} \}^{-1}
$$

<details>
    <summary>証明</summary>

$f_n(\boldsymbol{x}|\boldsymbol{\theta})=\prod_i f(x_i|\boldsymbol{\theta})$ について

$$
S_n(\boldsymbol{\theta},\boldsymbol{x}) = \left(\frac{\partial}{\partial \theta_1} \log f_n(\boldsymbol{x}|\boldsymbol{\theta}), \dots, \frac{\partial}{\partial \theta_n} \log f_n(\boldsymbol{x}|\boldsymbol{\theta}) \right)^\top
$$

とおくと、$\mathbb{E}[S_nS_n^\top]=n\boldsymbol{I} ,\mathbb{E}[(\hat{\boldsymbol{\theta}}-\boldsymbol{\theta})S_n^\top]=I_k$ を満たす（$I_k$は単位行列）。任意の $\boldsymbol{a},\boldsymbol{b} \in \R^k$ について、コーシー・シュワルツの不等式より

$$
\{\mathbb{E}[\boldsymbol{a}^\top(\hat{\boldsymbol{\theta}}-\boldsymbol{\theta})\boldsymbol{b}^\top S_n]\}^2 \le \mathbb{E}[\{\boldsymbol{a}^\top(\hat{\boldsymbol{\theta}}-\boldsymbol{\theta})\}^2] \ \mathbb{E}[\{\boldsymbol{b}^\top S_n^\top\}^2]
$$

となる。それぞれ

$$
\begin{aligned}
    \mathbb{E}[\boldsymbol{a}^\top(\hat{\boldsymbol{\theta}}-\boldsymbol{\theta})\boldsymbol{b}^\top S_n] &= \boldsymbol{a}^\top \mathbb{E}[(\hat{\boldsymbol{\theta}}-\boldsymbol{\theta})S_n^\top]\boldsymbol{b} = \boldsymbol{a}^\top \boldsymbol{b} \\
    \mathbb{E}[\{\boldsymbol{a}^\top(\hat{\boldsymbol{\theta}}-\boldsymbol{\theta})\}^2] &= \boldsymbol{a}^\top \mathbb{E}[(\hat{\boldsymbol{\theta}}-\boldsymbol{\theta})(\hat{\boldsymbol{\theta}}-\boldsymbol{\theta})^\top] \boldsymbol{a} = \boldsymbol{a}^\top \text{Cov}(\boldsymbol{\theta})\boldsymbol{a} \\
    \mathbb{E}[\{\boldsymbol{b}^\top S_n^\top\}^2] &= \boldsymbol{b}^\top \mathbb{E}[S_nS_n^\top] \boldsymbol{b} = \boldsymbol{b}^\top n\boldsymbol{I}  \boldsymbol{b}
\end{aligned}
$$

となるので、次の不等式が成り立つ。

$$
\boldsymbol{a}^\top \text{Cov}(\boldsymbol{\theta})\boldsymbol{a} \ge \sup_{\boldsymbol{b}\in\R^k} \frac{\boldsymbol{b}^\top \boldsymbol{a} \boldsymbol{a}^\top \boldsymbol{b}}{\boldsymbol{b}^\top n\boldsymbol{I}  \boldsymbol{b}} = \boldsymbol{a}^\top \{n\boldsymbol{I}\}^{-1} \boldsymbol{a}
$$

これが任意の $\boldsymbol{a}$ について成り立つので、$\text{Cov}(\boldsymbol{\theta})-\{n\boldsymbol{I}\}^{-1}$ は非負正定値（$\ge0$）となる。
</details><br>

また $\boldsymbol{\theta}$ の最尤推定量を $\hat{\boldsymbol{\theta}}_n^M$ とすると、定理 6.24 と同様にして漸近正規性を示すことができる。

$$
\sqrt{n} (\hat{\boldsymbol{\theta}}_n^M-\boldsymbol{\theta}) \to_d \mathcal{N}_k(\boldsymbol{0},\boldsymbol{I}^{-1})
$$

> フィッシャー情報行列は、モデル $M=\{p_{\xi}\in\mathcal{S}_{n-1}|\xi\in\Xi\}$ の局所座標系 $\xi=(\xi^i)$ に関するフィッシャー計量 
>$$
>g_p^{[n]} (X,Y) = \sum_{\omega=1}^n p_{\xi}(\omega) (X \log p_{\xi}(\omega))(Y \log p_{\xi}(\omega))
>$$
>の成分 $g_p^{[n]}(\frac{\partial}{\partial \xi^i},\frac{\partial}{\partial \xi^j})$ として定まる行列である。つまり、局所座標系 $\xi$ のまわりでの推定量の散らばり具合として解釈される。さらに、フィッシャー計量を用いて確率分布の距離を測ると便利な性質がある。

## Reference
1. [久保川達也著 『現代数理統計学の基礎』 共立出版 2017年04月](https://www.kyoritsu-pub.co.jp/bookdetail/9784320111660)
2. [AVILEN AI TREND](https://ai-trend.jp/author/masa/)
3. [正則条件下における 最尤推定量の漸近正規性](http://watanabe-www.math.dis.titech.ac.jp/users/swatanab/ML_cons2.pdf)
4. [数学特別講義（現代保険リスク理論）](https://www.math.kyoto-u.ac.jp/probability/shimizu3.pdf)
5. [でたらめの法則 — 極限定理 —](https://www.ms.u-tokyo.ac.jp/~nakahiro/surijoho16/surijoho2.pdf)
6. [[教材] 今更だが, ベイズ統計とは何なのか.](https://ill-identified.hatenablog.com/entry/2017/03/17/025625)
7. [事前分布について](http://watanabe-www.math.dis.titech.ac.jp/users/swatanab/prior.html)
8. [情報幾何入門 - 産業技術総合研究所](https://staff.aist.go.jp/s.akaho/papers/infogeo.ppt)
9. [幾何を使った統計のはなし](https://www.slideshare.net/motivic/ss-14465676)
