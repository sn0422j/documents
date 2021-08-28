---
title: パターン認識と機械学習 第11章 サンプリング法
---

パターン認識と機械学習の第11章「サンプリング法」の解説資料です。各ノーテーションは本紙に従っています。

## 11.0 はじめに
10章に引き続き、近似推論法を考える。  
決定的な近似法では、積分計算したい分布 $p(\bold{z})$ から**サンプル $\{\bold{z}^{(\ell)}\}_{\ell=1}^{L}$** を取得し、積分を有限和で近似する。

$$
\int f(\bold{z}) p(\bold{z})d\bold{z} \simeq \mathbb{E}\left[\frac{1}{L} \sum_{\ell=1}^{L} f(\bold{z}^{(\ell)})\right]
$$

## 11.1 基本的なサンプリングアルゴリズム
以下では、区間 $(0,1)$ で一様に分布する**疑似乱数**を発生させるアルゴリズムが与えられていると仮定する。

> メルセンヌ・ツイスタ法などがある。[Random sampling (numpy.random)](https://numpy.org/doc/stable/reference/random/index.html)

### 11.1.1 標準的な分布
一様分布に従う乱数を変換して、他の分布に従う乱数を生成できる。
- **逆変換法**  
累積分関数 $F(x)=p(X\le x)$ となる $X$ を生成したい。任意の $u \in \mathcal{U}(0,1)$ に対して、$F(x)=u$ となる $x$ は一対一対応するため、逆関数で $x=F^{-1}(u)$ と求められる。これを利用して、一様乱数 $U$ から $X=F^{-1}(U)$ と変換して $X$ を得る。
    - 例1: 指数分布 $X \sim \epsilon(\lambda)$ は $X = -\ln(1-U)/\lambda$ で生成できる。
    - 例2: コーシー分布 $X \sim \mathcal{C}(\mu,\sigma)$ は $X = \tan(\pi U - \pi/2)$ で生成できる。
- **ボックス・ミュラー法**
標準正規分布に従う乱数を生成する方法。$U_1,U_2 \ i.i.d \ \sim \mathcal{U}(0,1)$ から $Z_1 = \sqrt{-2\ln U_1} \cos 2\pi U_2, Z_2 = \sqrt{-2\ln U_1} \sin 2\pi U_2$ と変換すれば、$Z_1, Z_2$ は独立な標準正規分布 $\mathcal{N}(0,1)$ に従う。

これらは変数変換の公式 $p(\bold{y}) = p(\bold{z}) |\partial \bold{z}/\partial\bold{y}|$ で捉え直せる。しかし、**単純な分布でしか変換法は使えないため別のアプローチを考える必要がある**。

### 11.1.2 棄却サンプリング
サンプリングしたい分布 $p(z)$ から直接サンプリングするのは困難だが、任意の $z$ について以下の $\tilde{p}(z)$ を求められる場合を考える。

$$
p(z) = \frac{1}{Z_p} \tilde{p}(z)
$$

棄却サンプリングには、より簡単なサンプリング分布として**提案分布 $q(z)$** を必要とする。  要請として、次を満たす定数 $k\gt 0$ が存在すると仮定する。 **$kq(z)$ で $\tilde{p}(z)$ を覆っている**といえる。

$$
\tilde{p}(z) \le k q(z)
$$

以下の手順を繰り返してサンプル $u_0$ を得る。
1. $z_0 \sim q(z), u_0 \sim \mathcal{U}(0,kq(z_0))$ を順に生成する。
2. もし $u_0 \gt \tilde{p}(z_0)$ ならば $u_0$ は棄却される。そうでなければ $u_0$ が保持される。

採択確率は次のように求められる。したがって、**$k$ は小さい方が採択確率を高くすることができる**ため、$\tilde{p}(z) \le kq(z)$ の中で出来るだけ小さいものを選ぶと良い。
$$
(\text{採択確率}) = \int \frac{\tilde{p}(z_0)}{kq(z)} q(z) dz = \frac{1}{k} \int \tilde{p}(z) dz
$$

<div style="text-align: center">
    <video controls src="../videos/rejection_sampling.mp4" type="video/mp4"></video>
</div>

### 11.1.3 適応的棄却サンプリング
適切な提案分布 $q(z)$ を決めるのは難しいので、**ある包絡関数 $q(z)$ で覆うこと**を考える。初期のグリッド点集合によって区分的な包絡関数 $q(z)$ を構築し、それを用いてサンプリングする。

$$
q(z) = k_i \lambda_i \exp[-\lambda_i(z-z_i)] \quad (\hat{z}_{i-1,i} \lt z \le \hat{z}_{i,i+1})
$$

棄却されたサンプルは、包絡関数のグリッド点集合に取り入れることで包絡関数が改良されていく。

> 棄却サンプリングは1,2次元の乱数生成には有用であるが、高次元では採択確率が小さくなりすぎて適用が難しい。採択確率は次元数に応じて指数的に減少してしまう。

### 11.1.4 重点サンプリング
重点サンプリングは期待値を直接近似する方法であって、分布 $p(z)$ 自体のサンプルは生成しない。

$p(z)$ が容易に計算出来る場合を想定し、同じ状態空間の**提案分布 $q(z)$ から抽出したサンプル $\{z^{(\ell)}\}$ 上の有限和**で積分計算を近似する。
$$
\begin{aligned}
    \mathbb{E}_{p(z)}[f] &= \int f(z) p(z) dz \\
    &= \int f(z) \frac{p(z)}{q(z)} q(z) dz \\
    &\simeq \frac{1}{L} \sum_{\ell=1}^L f(z^{(\ell)}) \frac{p(z^{(\ell)})}{q(z^{(\ell)})} = \frac{1}{L} \sum_{\ell=1}^L f(z^{(\ell)}) \ r_{\ell}
\end{aligned}
$$

分布が正規化定数を除いてしか評価できないときは、以下のように計算する。

$$
\begin{aligned}
    \mathbb{E}_{p(z)}[f] &= \int f(z) p(z) dz \\
    &= \int f(z) \frac{\tilde{p}(z)/Z_p}{\tilde{q}(z)/Z_q} q(z) dz \\
    &\simeq \frac{Z_q}{Z_p} \frac{1}{L} \sum_{\ell=1}^L f(z^{(\ell)}) \underbrace{\frac{\tilde{p}(z^{(\ell)})}{\tilde{q}(z^{(\ell)})}}_{= \ \tilde{r}_{\ell}} = \frac{Z_q}{Z_p} \frac{1}{L} \sum_{\ell=1}^L f(z^{(\ell)}) \ \tilde{r}_{\ell} \\
    &\simeq \frac{1}{\sum_{m}\tilde{r}_m} \sum_{\ell=1}^L f(z^{(\ell)}) \ \tilde{r}_{\ell} = \sum_{\ell=1}^L f(z^{(\ell)}) \ \frac{\tilde{r}_{\ell}}{\sum_{m}\tilde{r}_m}
\end{aligned}
$$

> $Z_p/Z_q = 1/Z_q \int \tilde{p} dz = \int \tilde{p}/(\tilde{q}/q) dz \simeq (1/L) \sum_\ell r_\ell$

### 11.1.5 SIR
**SIR (sampling-importance-resampling)** は棄却サンプリングと同様に提案分布 $q(z)$ を利用するが、定数 $k$ を決定する必要はない。

1. $L$ 個のサンプル $\{z^{(\ell)}\}$ を $q(z)$ から抽出する。
2. $\{\tilde{r}_{\ell}\}$ を計算し、次の $L$ 個のサンプリングに利用する。
3. 繰り返す。

$L \to \infty$ の極限で、サンプルは $p(z)$ に正確に従う。これは、再サンプリングされた累積分布が $p(z)$ の累積分布に収束することからわかる。尚、$I(\cdot)$ は指示関数である。

$$
\begin{aligned}
    p(z\le a) &= \sum_{ \{ \ell \ : \ z^{(\ell)} \le a \}} \frac{\tilde{r}_{\ell}}{\sum_{m}\tilde{r}_m} = \frac{\sum_{\ell} I(z^{(\ell)} \le a) \tilde{p}(z^{(\ell)})/q(z^{(\ell)})}{\sum_{\ell} \tilde{p}(z^{(\ell)})/q(z^{(\ell)})} \\
    &\underset{L\to\infty}{\to} \frac{\int I(z \le a) \{ \tilde{p}(z)/q(z) \} q(z)dz}{\int \{ \tilde{p}(z)/q(z) \} q(z)dz} = \int I(z \le a) p(z) dz
\end{aligned}
$$

### 11.1.6 サンプリングとEMアルゴリズム
EMアルゴリズムで**Eステップ（事後分布による期待値計算）を解析的に実行できないモデル**に対して、サンプリングを用いて近似することができる。推定事後分布 $p(\bold{Z}|\bold{X}, \boldsymbol{\theta}^{\text{old}})$ からサンプル $\{Z^{(\ell)}\}$ を抽出して有限和を考えればよい。 

$$
Q(\boldsymbol{\theta}, \boldsymbol{\theta}^{\text{old}}) = \int p(\bold{Z}|\bold{X}, \boldsymbol{\theta}^{\text{old}}) \ln p(\bold{Z},\bold{X}| \boldsymbol{\theta}) d\bold{Z} \\
\simeq \frac{1}{L} \sum_{\ell=1}^L \ln p(\bold{Z}^{(\ell)},\bold{X}| \boldsymbol{\theta})
$$

> この手続きは**モンテカルロEMアルゴリズム**と呼ばれる。特別な例として、**確率的EM**では抽出するサンプルを1つとする。

パラメーターの事後分布 $p(\boldsymbol{\theta}| \bold{Z}, \bold{X})$ からサンプルを抽出するものは**IPアルゴリズム**と呼ばれる。

1. $I$ ステップ：$p(\boldsymbol{\theta}| \bold{X})$ からサンプル $\{\boldsymbol{\theta}^{(\ell)}\}$ を抽出し、これを用いた $p(\bold{Z}|\bold{X}, \boldsymbol{\theta}^{(\ell)})$ からサンプル $\{\bold{Z}^{(\ell)}\}$ を抽出する。
2. $P$ ステップ：$I$ ステップで得られたサンプルを用いて $p(\boldsymbol{\theta}| \bold{X})$ を更新する。

$$
p(\boldsymbol{\theta}| \bold{X}) \simeq \frac{1}{L} \sum_{\ell=1}^L p(\boldsymbol{\theta}| \bold{Z}^{(\ell)}, \bold{X})
$$

## 11.2 マルコフ連鎖モンテカルロ
マルコフ連鎖モンテカルロ法 (MCMC) では、**現在の状態 $\bold{z}^{(\tau)}$ に依存した提案分布 $q(\bold{z}|\bold{z}^{(\tau)})$ からサンプリングを行う**。そのため、**サンプルの系列はマルコフ連鎖を成す**。 棄却サンプリングと同様に、サンプル $\bold{z}^*$ は適切な基準に従って棄却・採択する。

基本的な Metropolis アルゴリズムでは、提案分布は対称 $q(\bold{z}_A|\bold{z}_B) = q(\bold{z}_B|\bold{z}_A)$ である。  
サンプル候補は次の確率で採択される。採択されたサンプルは保持し、次の状態に設定する。
$$
P( A(\bold{z}^*,\bold{z}^{(\tau)}) \gt u), \quad u \sim \mathcal{U}(0,1) \\[0.5em]
A(\bold{z}^*,\bold{z}^{(\tau)}) = \min \left( 1, \frac{\tilde{p}(\bold{z}^*)}{\tilde{p}(\bold{z}^{(\tau)})} \right)
$$

こうして得られたサンプル系列の分布は、$\tau\to\infty$ で $p(\bold{z})$ に近づく。ただし、連続したサンプル間は高い相関を持ち、独立なサンプルではない。独立性を保ちたい場合は、十分大きな間隔で間引くとよい。

ランダムウォークで状態空間を探索するのは非効率であるため、ランダムウォーク的な振る舞いを避けるMCMCを設計する必要がある。

### 11.2.1 マルコフ連鎖
マルコフ連鎖 $\{\bold{z}^{(m)}\}$ は、**初期分布 $p(\bold{z}^{(0)})$ と遷移確率 $T_m(\bold{z}^{(m)}, \bold{z}^{(m+1)}) = p(\bold{z}^{(m+1)} | \bold{z}^{(m)})$ で指定できる**。

特定の変数の周辺分布率は、以下のように計算できる。
$$
p(\bold{z}^{(m+1)}) = \sum_{\bold{z}^{(m)}} p(\bold{z}^{(m+1)}|\bold{z}^{(m)}) p(\bold{z}^{(m)})
$$

周辺分布がマルコフ連鎖の各ステップで変わらないとき、その分布は**連鎖に関して不変である ( 定常である )** という。  
不変性は遷移確率を用いて $p^*(\bold{z}) = \sum_{\bold{z}'} T(\bold{z}',\bold{z}) p^*(\bold{z}')$ とも書ける。

求めたい分布が**不変分布である十分条件**は、次の**詳細釣り合い条件**を満たす遷移確率を選ぶことである。
$$
p^*(\bold{z})T(\bold{z},\bold{z}') = p^*(\bold{z}')T(\bold{z}',\bold{z}) \tag{11.40}
$$

詳細釣り合い条件を満たすマルコフ連鎖は**可逆である**という。

不変性以外にも、**エルゴード性**を持つ不変分布 = **平衡分布**である必要がある。エルゴード性とは、初期分布 $p(\bold{z}^{(0)})$ に依らず $m\to\infty$ で $p(\bold{z}^{(m)})$ が求めたい分布 $p^*(\bold{z})$ に収束することを指す。

多くの場合、遷移確率を基本遷移の組 $\{B_k\}$ から構築する。

次の場合は
$$
T(\bold{z}',\bold{z}) = \sum_{k=1}^K \alpha_k B_k(\bold{z}',\bold{z})
$$
- 不変分布になる。
- 各基本遷移が詳細釣り合い条件を満たすならば、$T$ も詳細釣り合い条件を満たす。

次の場合は
$$
T(\bold{z}',\bold{z}) = \sum_{\bold{z}_1} \cdots \sum_{\bold{z}_{K-1}} B_1(\bold{z}',\bold{z}_1) \cdots B_{K-2}(\bold{z}_{K-2},\bold{z}_{K-1}) B_{K-1}(\bold{z}_{K-1},\bold{z})
$$
- 不変分布になる。
- 詳細釣り合い条件を満たす基本遷移の順序を $B_1, \dots, B_K, B_K, \dots, B_1$ と並べれば、$T$ も詳細釣り合い条件を満たす。

### 11.2.2 Metropolis-Hastings アルゴリズム
Metropolis-Hastings アルゴリズムでは、**提案分布 $q_k(\bold{z}|\bold{z}^{(\tau)})$ は対称でなくてもよい**。  
採択率は以下のようになる。
$$
A_k(\bold{z}^*, \bold{z}^{(\tau)}) = \min \left(1, \frac{\tilde{p}(\bold{z}^*)q_k(\bold{z}^{(\tau)}|\bold{z}^*)}{\tilde{p}(\bold{z}^{(\tau)})q_k(\bold{z}^*|\bold{z}^{(\tau)})} \right)
$$

このマルコフ連鎖が詳細釣り合い条件 $(11.40)$ を満たすことを以下に示す。
$$
\begin{aligned}
    p(\bold{z})q_k(\bold{z}'|\bold{z})A_k(\bold{z}', \bold{z}) &= \min ( p(\bold{z})q_k(\bold{z}'|\bold{z}), p(\bold{z}')q_k(\bold{z}|\bold{z}') )\\
    &= \min ( p(\bold{z}')q_k(\bold{z}|\bold{z}'), p(\bold{z})q_k(\bold{z}'|\bold{z}) ) \\
    &= p(\bold{z}')q_k(\bold{z}|\bold{z}') A_k(\bold{z}, \bold{z}')
\end{aligned}
$$

> 分布の変動が方向によって大きく異なる場合、MHアルゴリズムの収束は非常に遅くなり得る問題がある。

<div style="text-align: center">
    <video controls src="../videos/metropolis_hastings.mp4" type="video/mp4"></video>
</div>

## 11.3 ギブズサンプリング
ギブズサンプリングは、MHアルゴリズムの特別な場合であるが、**単純で適用範囲の広いMCMC法**である。

以下の手順でサンプル $\{\bold{z}^{(\tau)}\}$ を得る。
1. $\{z_i\}_{i=1}^M$ を初期化する。
2. $\tau=1,\dots,T$ に対して以下を行う。
    - $z_1^{(\tau+1)} \sim p(z_1|\bold{z}_{\backslash 1}^{(\tau)})$ をサンプリングする。  
    $\vdots$
    - $z_M^{(\tau+1)} \sim p(z_M|\bold{z}_{\backslash M}^{(\tau)})$ をサンプリングする。

各種性質については次の通りである。
- 不変性：$\bold{z}_{\backslash i}$ は固定するので、$p(\bold{z}_{\backslash i})$ は不変。したがって $p(\bold{z}^{(\tau+1)})=\sum_{i} p(z_i|\bold{z}_{\backslash i})p(\bold{z}_{\backslash i})$ となる。
- エルゴード性：十分条件は条件付き分布の確率が0となる場合がないこと。

> $q_k(\bold{z}^*|\bold{z})=p(z_k^*|\bold{z}_{\backslash k})$ かつ $A(\bold{z}^*, \bold{z}) = 1$ のMHアルゴリズムになる。

ギブズサンプリングにおけるランダムウォーク的な振る舞いを低減するためのアプローチには、**過剰緩和**がある。条件付き分布がガウス分布となるとき、各ステップにおいて $z_i \sim \mathcal{N}(\mu_i, \sigma_i^2)$ を次のように変換する。
$$
z_i' = \mu_i + \alpha(z_i - \mu_i) + \sigma_i \sqrt{1-\alpha^2} \nu, \quad \nu \sim \mathcal{N}(0,1), \ -1 \lt \alpha \lt 1
$$

> 8.3.3節のICMアルゴリズムは、ギブズサンプリングの貪欲法的な近似法とみなせる。

<div style="text-align: center">
    <video controls src="../videos/gibbs_sampling.mp4" type="video/mp4"></video>
</div>

## 11.4 スライスサンプリング
スライスサンプリングは、**分布の特徴に合わせてステップサイズを自動的に調整する**ことができる。

$z$ を付加的な変数 $u$ で拡張し、結合された $(z,u)$ 空間からサンプルを抽出する。以下の $\hat{p}(z,u)$ からサンプリングし、$u$ の値を無視することで、$p(z)$ からのサンプリングを行うことができる。

$$
\hat{p}(z,u) = \begin{cases}
    1/Z_p & 0 \le u \le \tilde{p}(z) \\
    0 & \text{otherwise}
\end{cases} \\[1em]
\int \hat{p}(z,u) du = \int_0^{\tilde{p}(z)} \frac{1}{Z_p} du = \frac{1}{Z_p} \tilde{p}(z) = p(z)
$$

これは $z$ と $u$ を交互にサンプルすれば達成できる。$u$ を固定したときの空間 $\{z \ : \ \tilde{p}(z) \gt u \}$ が分布の「スライス」となる。 

実際は、分布のスライスから直接サンプリングするのは困難なことがあり、代わりに $\hat{p}(z,u)$ に従う一様分布を不変にするサンプリングを行う。

## 11.5 ハイブリッドモンテカルロアルゴリズム
勾配を用いて棄却率を低く保ったまま効率よく探索する方法を考えていく。

### 11.5.1 力学系
位置変数 $\bold{z}$ と運動量変数 $\bold{r}$ を考える。ハミルトン関数は以下で与えられる。  
力学系の時間発展においてハミルトン関数の値は一定である。
$$
H(\bold{z}, \bold{r}) = \underbrace{E(\bold{z})}_{\text{potential}} + \underbrace{K(\bold{r})}_{\text{kinetic}} = \text{const}
$$

> 解析力学では $(q,p)$ と表示されることが多いと思う。

力学系は以下のハミルトン方程式で表される。
$$
\begin{aligned}
    \frac{dz_i}{d\tau} &= \frac{\partial H}{\partial r_i} \\
    \frac{dr_i}{d\tau} &= -\frac{\partial H}{\partial z_i} \\
\end{aligned}
$$

ハミルトン力学系では、位相空間（位置と運動量の結合空間）の体積が保存される（リウヴィルの定理）。  
つまり、位相空間の流れ場 $\bold{V}$ が一定になる。
$$
\bold{V} = \left( \frac{d\bold{z}}{d\tau}, \frac{d\bold{r}}{d\tau} \right) = \text{const}
$$

これらの性質から、位相空間上での同時分布 $p(\bold{z},\bold{r})$ が不変になる。
$$
p(\bold{z},\bold{r}) = \frac{1}{Z_H} \exp[-H(\bold{z},\bold{r})] \text{ is invariant.}
$$

この力学系で時間発展を追ってサンプリング（時間区間積分）すると系統的な変化を起こせる。エルゴード的にサンプリングするためには、同時分布 $p(\bold{z},\bold{r})$ が不変に保ちつつ $H$ の値を変える必要がある。例えば、$\bold{r}$ の値を $p(\bold{r}|\bold{z})$ で抽出した値で置き換えることができる。

実際にハミルトン方程式の数値積分をするには、**リープフロッグ離散化**を用いて数値誤差を最小にする必要がある。  
位置変数と運動変数の離散時間近似を交互に更新する。複数のリープフロッグが連続すると、ステップサイズ $\epsilon$ で全ステップを更新できる。 
$$
\begin{aligned}
    \colorbox{lightcyan}{$ \hat{r}_i (\tau + \epsilon/2) $} &= \hat{r}_i (\tau) - \frac{\epsilon}{2} \frac{\partial E(\hat{\bold{z}}(\tau))}{\partial r_i} \\
    \colorbox{beige}{$ \hat{z}_i (\tau + \epsilon) $}  &= \hat{z}_i (\tau) + \epsilon \colorbox{lightcyan}{$ \hat{r}_i (\tau + \epsilon/2) $} \\
    \hat{r}_i (\tau + \epsilon) &=  \colorbox{lightcyan}{$ \hat{r}_i (\tau + \epsilon/2) $}  - \frac{\epsilon}{2} \frac{\partial E(\colorbox{beige}{$ \hat{\bold{z}}(\tau+\epsilon)$})}{\partial r_i} 
\end{aligned}
$$

時間区間 $\Tau$ だけ進めるには $\Tau/\epsilon$ ステップで実現でき、$E(\bold{z})$ が滑らかならば $\epsilon\to 0$ の極限で数値誤差はゼロになる。

### 11.5.2 ハイブリッドモンテカルロアルゴリズム
ハミルトン力学と Metropolis アルゴリズムを組み合わせる。
もし $(\bold{z},\bold{r})$ が初期状態で $(\bold{z}^*,\bold{r}^*)$ がリープフロッグによる積分後の状態とすると、以下の確率でこれを採択する。

$$
\min(1, \exp[H(\bold{z},\bold{r})-H(\bold{z}^*,\bold{r}^*)])
$$

数値誤差のバイアスを取り除くため、リープフロッグの更新式が詳細釣り合い条件 $(11.40)$ を満たすように修正する。時間が進む方向 $+\epsilon$ と戻る方向 $-\epsilon$ をランダムに選ぶと、積分の効果を打ち消せる。また、$z_i$ または $r_i$ のどちらかを他の変数のみの関数となる量だけ更新する。


> 【詳細釣り合い条件を満たすことの証明】  
> 位相空間の小さな領域 $\mathcal{R}$ がリープフロッグ $L$ 回で $\mathcal{R}'$ に移されるとし、各領域の体積を $\delta V$ とする。初期点を分布 $p(\bold{z},\bold{r})$ から選んだときの遷移確率は
> $$
>\frac{1}{Z_H} \exp[-H(\mathcal{R})]\delta V \frac{1}{2} \min(1, \exp[H(\mathcal{R})-H(\mathcal{R}')]) \tag{11.68}
>$$
> となる。逆方向の遷移確率は
> $$
>\frac{1}{Z_H} \exp[-H(\mathcal{R}')]\delta V \frac{1}{2} \min(1, \exp[H(\mathcal{R}')-H(\mathcal{R})]) \tag{11.69}
>$$
> となる。これら2つの遷移確率は、$H(\mathcal{R})=H(\mathcal{R}')$ の時は自明に等しく、$H(\mathcal{R})\gt H(\mathcal{R}')$ のときは
> $$
>\begin{aligned}
>    &(11.68) = \frac{1}{Z_H} \exp[-H(\mathcal{R})]\delta V \frac{1}{2} \\
>    &(11.69) = \frac{1}{Z_H} \exp[-H(\mathcal{R}')]\delta V \frac{1}{2} \exp[H(\mathcal{R}')-H(\mathcal{R})] = (11.68)
>\end{aligned}
>$$ 
> となって等しく、$H(\mathcal{R})\lt H(\mathcal{R}')$ のときも同様である。従って、詳細釣り合い条件を満たす。

<div style="text-align: center">
    <video controls src="../videos/hybridMCMC.mp4" type="video/mp4"></video>
</div>

## 11.6 分配関数の推定
分配関数はモデルエビデンスになるため関心がある。  
モデル比較のためには、**分配関数の比率**を知られれば良い。
$$
\begin{aligned}
    \frac{Z_E}{Z_G} &= \frac{\sum_{\bold{z}}\exp[-E(\bold{z})]}{\sum_{\bold{z}}\exp[-G(\bold{z})]} \\[1em]
    &= \frac{\sum_{\bold{z}}\exp[-E(\bold{z})+G(\bold{z})]\exp[-G(\bold{z})]}{\sum_{\bold{z}}\exp[-G(\bold{z})]} \\[1em]
    &= \mathbb{E}_{G(z)}[\exp[-E(\bold{z})+G(\bold{z})]]
\end{aligned}
$$

重点サンプリングを用いる場合は、以下のようにサンプル $\{\bold{z}^{(\ell)}\sim p_G(\bold{z}) \}$ を利用して近似できる。しかし、このアプローチは $p_E/p_G$ が大きな変動をしない場合のみにしか精度良く適用できない。
$$
\mathbb{E}_{G(z)}[\exp[-E(\bold{z})+G(\bold{z})]] \simeq \frac{1}{L} \sum_{\ell=1}^{L} \exp[-E(\bold{z}^{(\ell)})+G(\bold{z}^{(\ell)})]
$$

遷移確率 $T(\bold{z},\bold{z}')$ のマルコフ連鎖のサンプル $\{\bold{z}^{(\ell)}\}$ を利用する場合もある。
$$
\frac{1}{Z_G} \exp[-G(\bold{z})] = \frac{1}{L} \sum_{\ell=1}^L T(\bold{z}^{(\ell)},\bold{z})
$$

比率の推定には、2つの分布が近くなければならない。複雑な分布の場合は、以下の連鎖を用いて対応することが可能である。
$$
\frac{Z_M}{Z_1} = \frac{Z_2}{Z_1} \frac{Z_3}{Z_2} \cdots \frac{Z_M}{Z_{M-1}}
$$

## Reference
- Bishop, Christopher M., 元田 浩, 栗田 多喜夫, 樋口 知之, 松本 裕治, 村田 昇, (2012), パターン認識と機械学習 : ベイズ理論による統計的予測, 丸善出版
- [第11章　サンプリング法 博士課程１年 原　祐輔.](https://slidesplayer.net/slide/11520091/)
- [PRML輪読#11](https://www.slideshare.net/matsuolab/prml11-78266228)
- [Bishop prml 11.5-11.6_wk77_100606-1152(発表に使った資料)](https://www.slideshare.net/wk77/bishop-prml-115116wk771006061152)