## Covariance of a probe with $F$ for generalized cones

We compute $\text{Cov}(\text{val}(u), F \mid \text{known})$ for an unqueried vertex $u$ at layer $L$.

### Setup

$F = \frac{1}{W_0}\sum_{v_0} \text{val}(v_0)$. Known $v_0$ contribute constants (zero covariance), so:

$$\text{Cov}(\text{val}(u), F) = \frac{1}{W_0} \sum_{v_0 \text{ unknown}} \text{Cov}(\text{val}(u), \text{val}(v_0))$$

### Key independence

An unknown $v_0$'s path to layer $L$ depends on oracle queries at layers $0, \ldots, L{-}1$.  
The value $\text{val}(u)$ depends on queries at layers $L, \ldots, D{-}1$.  
These are disjoint qids $\Rightarrow$ **independent**. So we can condition on $v_0$'s position $w$ at layer $L$:

$$\text{Cov}(\text{val}(u), \text{val}(v_0)) = \sum_w P(v_0 \text{ at } w) \cdot \text{Cov}(\text{val}(u), \text{val}(w))$$

### Three cases for $w$ at layer $L$

- **$w = u$**: $\text{Cov} = \text{Var}(\text{val}(u))$
- **$w \neq u$, known**: $\text{val}(w)$ is a constant $\Rightarrow$ $\text{Cov} = 0$
- **$w \neq u$, unknown**: $\text{Cov} = \text{merge\_var}[L]$ (defined below)

### Reaching layer $L$

From an unknown $v_0$, the path follows unknown transitions (uniform random) until it either reaches layer $L$ or hits a known vertex (after which it stays known, by the invariant that known transitions point only to known vertices). The probability of being at a specific unknown vertex at layer $L$:

$$p_u = \frac{1}{n_{\text{unk}}[L]} \prod_{\ell=1}^{L} \frac{n_{\text{unk}}[\ell]}{W_\ell}, \qquad p_{\text{unk},\neq u} = p_u \cdot (n_{\text{unk}}[L] - 1)$$

### Assembling the covariance

$$\boxed{\text{Cov}(\text{val}(u), F) = \frac{n_{\text{unk}}[0]}{W_0}\, p_u \left[\text{Var\_unk}[L] + (n_{\text{unk}}[L]-1)\,\text{merge\_var}[L]\right]}$$

### Computing the components (all backward DPs in $D$ scalars)

**$\text{Var\_unk}[L]$** — conditional variance of an unknown vertex's true value:

$$\text{sq}[D{-}1] = 1, \quad \text{sq}[L] = \frac{1}{W_{L+1}}\!\left(\sum_{\text{known } v} \text{val}(v)^2 + n_{\text{unk}}[L{+}1]\,\text{sq}[L{+}1]\right)$$
$$\text{Var\_unk}[L] = \text{sq}[L] - \overline{\text{layer\_vals}[L{+}1]}^{\,2}$$

**$\text{merge\_var}[L]$** — covariance of two distinct unknown vertices at layer $L$, from their forward paths merging at a deeper layer:

$$\text{merge\_var}[D{-}1] = 0$$
$$\text{merge\_var}[L] = \frac{n_{\text{unk}}[L{+}1]\left(\text{Var\_unk}[L{+}1] + (n_{\text{unk}}[L{+}1]-1)\,\text{merge\_var}[L{+}1]\right)}{W_{L+1}^2}$$

Derivation: two independent transitions from layer $L$ each land uniformly on $W_{L+1}$ vertices. They collide (prob $1/W_{L+1}$) sharing variance, or land on two distinct vertices whose covariance is $\text{merge\_var}[L{+}1]$ if both unknown, or $0$ if either is known.
