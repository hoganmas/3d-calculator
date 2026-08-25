# NOTES: Approximating Polynomial Transmittance

> **Viewer context:** The shipped poly-cloud **golden path** is dens atlas bake + Beer raymarch ([`clip-space-babbage.md`](./clip-space-babbage.md)), not the analytic forms below. Chebyshev-\(\hat\tau\) Path C in the explorer is documented separately in [`path-c.md`](./path-c.md) (legacy). This note is the older research writeup on approximating \(T\) with simpler summands.

# Introduction

For Volume Rendering, we need to efficiently determine the value of the integral:

$\int_{z_0}^{z_f} dz\ g(z)\exp(- \int _z^{z_f}dz'  f(z'))$ 

However, a closed from solution for this expression exists for a very limited subset of $f(z)$ and $g(z)$. For instance, if $f(z)$ is a polynomial, then we find that a closed form solution only exists if $f(z)$ is linear. 

Here, we’ll attempt to find a general way to approximate $T(z)=\exp(- \int _z^{z_f}dz'  f(z'))$ as a sum of simpler functions that allow for the full integral to be analytically evaluated.

# Analysis

## Assumptions

We will assume that $\forall z$, $f(z) \ge 0$. This assumption holds valid assuming this terms corresponds to light attenuation coefficient in the original radiative transfer equation.

Thus, $\int _z^{z_f}dz'  f(z'))$ is a monotonically increasing function where $\lim_{z \to z_f}\int _z^{z_f}dz'  f(z') = 0$. Therefore, $T(z)$ is a monotonically increasing function where:

- $\lim_{z \to z_f}T(z)= \lim_{z \to z_f}\exp(- \int _z^{z_f}dz'  f(z')) = 1$
- $T(z)=\exp(- \int _z^{z_f}dz'  f(z')) > 0$

Thus, we can get an accurate approximation of $T(z)$ by approximating as a sum of sigmoids on within $z \in [z_i,z_f]$ 

## Approximating Transmittance as Sum of Gaussian CDFs

Let $f(z)=\sum_i^N w_i (z-z_0)^i$. That is, the input attenuation coefficient is a polynomial function w.r.t depth.

We can express $T(z)$ as the following:

$T(z)= \exp(- \int _z^{z_f}dz' \sum_i w_i (z' - z_0)^i)$

$=\exp(- \sum_i w_i \int _z^{z_f}dz' (z' - z_0)^i)$

$=\exp(- \sum_i w_i |\frac{(z' - z_0)^{i+1}}{i+1}|_z^{z_f})$

$=\exp(- \sum_i \frac{w_i}{i+1} ((z_f-z_0)^{i+1} - (z-z_0)^{i+1}))$

$=\exp(\sum_i \frac{w_i}{i+1} (z - z_0)^{i+1}) \exp(-\sum_i \frac{w_i}{i+1} (z_f - z_0)^{i+1})$

$=A \exp(\sum_i \frac{w_i}{i+1} (z - z_0)^{i+1})$

$=A \prod_i \exp(\frac{w_i}{i+1} (z - z_0)^{i+1})$

where $A=\exp(-\sum_i \frac{w_i}{i+1} (z_f - z_0)^{i+1})$.

<aside>
💡

Additionally, we can substitute the following approximation (See Appendix A):

$\exp(A(z-z_0)^n) \approx B+C\Phi(\frac{z-\mu}{\sigma})=B+C \Phi(h(z))$

where $B= 1$ for $n > 1$.

</aside>

$=A \prod_i (B_i+C_i \Phi(h_i(z)))$

where $h_i(z)=\frac{z - \mu_i}{\sigma_i}$

We can expand this product out:

<aside>
💡

For two terms:

$\prod_i^2 (B_i+C_i \Phi(h_i(z)))$

$=(B_1+C_1 \Phi(h_1(z)))(B_2+C_2 \Phi(h_2(z)))$

$=B_1B_2 + B_2C_1 \Phi(h_1(z)) + B_1C_2 \Phi(h_2(z)) + C_1 C_2  \Phi(h_1(z)) \Phi(h_2(z))$ 

We can substitute the approximation:

 $\Phi(\frac{z-\mu_1}{\sigma_1}) \Phi(\frac{z-\mu_2}{\sigma_2}) \approx \begin{cases}
\Phi(\frac{z-\mu_1}{\sigma_1}) & \text{ if } \mu_1 > \mu_2 \\

\Phi(\frac{z-\mu_2}{\sigma_2}) & \text{ if } \mu_1 < \mu_2 \\
\end{cases}$

If we assume that terms are ordered by $\mu$:

$\approx B_1B_2+ B_2C_1\Phi(h_1(z))+B_1C_2 \Phi(h_2(z)) +C_1C_2   \Phi(h_2(z))$ 

$=  B_1B_2+ B_2C_1 \Phi(h_1(z))+ (C_1+B_1)C_2\ \Phi(h_2(z))$ 

</aside>

We can employ this expansion recursively, and find that the final result reduces down to an approximate sum of $N$ weighted Gaussian CDFs and a constant term $F=\prod_i B_i$.

For a set of $N$ depth-ordered Gaussian CDFs:

$\approx A (F+ \sum_i E_i\Phi(h_i(z))$

where $E_i= C_i (\prod_{j<i} (C_j+B_j) )(\prod_{k>i} B_k)$

Thus, we have the final form of our approximated transmittance:

$T(z) \approx A (F+ \sum_i E_i\Phi(h_i(z))$,

where:

- $A=\exp(-\sum_i \frac{w_i}{i+1} (z_f - z_0)^{i+1})$
- $E_i= C_i (\prod_{j<i} (C_j+B_j) )(\prod_{k>i} B_k)$
- $F=\prod_{i}^{N}B_i$
- $h_i(z)=\frac{z - \mu_i}{\sigma_i}$

## Calculating Radiative Transfer

Now, we can return to our original expression:

$\int_{z_0}^{z_f} dz\ g(z)\exp(- \int _z^{z_f}dz'  f(z'))$

$\approx \int_{z_0}^{z_f} dz\ g(z) *A (F+ \sum_i E_i\Phi(h_i(z))$

$= A(F\int_{z_0}^{z_f} dz\ g(z))+ A\sum_i E_i\int_{z_0}^{z_f} dz\ g(z)\ \Phi(h_i(z))$

Assuming our emission function is also a polynomial: 

- $g(z)= \sum_j c_jz^j$

Additionally, we can perform a change of basis on the polynomial based on a given $h_i(z)$

- $g(z)= \sum_j c_jz^j = \sum_j c_{ij}(\frac{z - \mu_i}{\sigma_i})^j$

$= A(F\int_{z_0}^{z_f} dz\  \sum_j c_jz^j )+ A\sum_i E_i\int_{z_0}^{z_f} dz\  \sum_j c_{ij}(\frac{z - \mu_i}{\sigma_i})^j\ \Phi(\frac{z - \mu_i}{\sigma_i})$

$=A( F\sum_j \int_{z_0}^{z_f} dz\ c_jz^j )+ A\sum_i \sum_j E_i c_{ij}\int_{z_0}^{z_f} dz\  (\frac{z - \mu_i}{\sigma_i})^j\ \Phi(\frac{z - \mu_i}{\sigma_i})$

$=A(F \sum_j \frac{c_j}{j+1}(z_f^{j+1} - z_0^{j+1}))+ A\sum_i \sum_j E_i c_{ij}\int_{z_0}^{z_f} dz\  (\frac{z - \mu_i}{\sigma_i})^j\ \Phi(\frac{z - \mu_i}{\sigma_i})$

<aside>
💡

$\int_{z_0}^{z_f} dz\  (\frac{z - \mu_i}{\sigma_i})^j\ \Phi(\frac{z - \mu_i}{\sigma_i})$

$= \frac{\sigma_i}{j+1}[u_f^{j+1} \Phi(u_f)-u_{0}^{j+1} \Phi(u_0) - M_{j+1}(u_0,u_f)]$

where:

- $u_f =  \frac{z_f-\mu_i}{\sigma_i}$
- $u_0 =  \frac{z_0-\mu_i}{\sigma_i}$

and $M_j(u_0,u_f)$ is defined by the following recurrence relation:

- $M_0(u_0,u_f)= \Phi(u_f)-\Phi(u_0)$
- $M_1(u_0,u_f)= \phi(u_0)-\phi(u_f)$
- $M_k(u_0,u_f)= (k - 1)M_{k-2}(u_0,u_f) + u_0^{k-1} \phi(u_0)-u_f^{k-1} \phi(u_f)$
</aside>

$= A(F \sum_j \frac{c_j}{j+1}(z_f^{j+1} - z_0^{j+1}))+ A\sum_i \sum_j E_i c_{ij} \frac{\sigma_i}{j+1}[u_f^{j+1} \Phi(u_f)-u_{0}^{j+1} \Phi(u_0) - M_{j+1}(u_0,u_f)]$

---

# Appendix

## Appendix A: Approximating Exponentiated Monomial

Let $f(x)=\exp(A(x-x_i)^n)$. 

### Case 1: $n > 1$

Hence, we can approximate the function as a scaled and shifted Gaussian CDF on the interval $[x_i,x_f]$. 

For $n > 1$:

- $f'(x_i)=0$
- $f(x_i)=\exp(0)=1$

Thus the form of the approximation should look like:

$f(x)=\exp(A(x-x_i)^n) \approx 1+C\Phi(\frac{x-\mu}{\sigma})$

Additionally, we will find that $f(x)$ features a potential inflection point (center) at:

 $\frac{d}{dx}f(x) = An(x-x_i)^{n-1} \exp(A(x-x_i)^n)$ 

 $\frac{d^2}{dx^2}f(x) = An(n-1)(x-x_i)^{n-2} \exp(A(x-x_i)^n) + A^2n^2 (x-x_i)^{2n-2} \exp(A(x-x_i)^n) = 0$

 $[(n-1) + An (x-x_i)^{n}][An(x-x_i)^{n-2} \exp(A(x-x_i)^n)] = 0$

 $(n-1) + An (x-x_i)^{n} = 0$

 $(x-x_i)^{n} = \frac{1-n}{An}$

 $x-x_i = (|\frac{1-n}{An}|)^{\frac{1}{n}}$

 $x = x_i + (|\frac{1-n}{An}|)^{\frac{1}{n}}$

 $x= x_i + (|\frac{1-n}{An}|)^{\frac{1}{n}}$

However, the reality is that this point is only an inflection point when $A < 0$. Thus in the case that either $A > 0$ or $x_i + (|\frac{1-n}{An}|)^{\frac{1}{n}} > x_f$, we can simply use $x_f$ as the mean:

$\mu=\begin{cases}
x_i + (|\frac{1-n}{An}|)^{\frac{1}{n}} & \text{ if } A < 0 \text{ and } x_i + (|\frac{1-n}{An}|)^{\frac{1}{n}} < x_f \\

x_f & \text{ else }
\end{cases}$

To determine the value of $C$, we assume reflectional symmetry across $\mu$:

$C=2(f(\mu)-1)$

We can then approximate the variance by solving for the the slope at the mean:

$\frac{d}{dx}[1+C\Phi(\frac{x-\mu}{\sigma})]=\frac{1}{\sigma}C\phi(\frac{x-\mu}{\sigma})$

At $x = \mu$:

$\frac{d}{dx}[1+C\Phi(\frac{x-\mu}{\sigma})]=\frac{1}{\sigma}C\phi(0)

=\frac{1}{\sigma}C\frac{1}{\sqrt{2\pi}}

=\frac{C}{\sigma \sqrt{2\pi}}$

$\frac{d}{dx}f(\mu) = An(\mu-x_i)^{n-1} \exp(A(\mu-x_i)^n)$

Letting the slope be equal for the approximation:

$\frac{C}{\sigma \sqrt{2\pi}} =  An(\mu-x_i)^{n-1} \exp(A(\mu-x_i)^n)$

$\sigma=\frac{C}{An\sqrt{2\pi}(\mu-x_i)^{n-1} \exp(A(\mu-x_i)^n)}$

$= \frac{C}{An\sqrt{2\pi}}  (\mu-x_i)^{1-n} \exp(-A(\mu-x_i)^n)$

When $\mu=x_f$:

$\sigma =  \frac{C}{An\sqrt{2\pi}}  (x_f-x_i)^{1-n} \exp(-A(x_f-x_i)^n)$

When $\mu = x_i + (|\frac{1-n}{An}|)^{\frac{1}{n}}$:

$\sigma =  \frac{C}{An\sqrt{2\pi}}  (|\frac{1-n}{An}|)^{\frac{1}{n}-1} \exp(-A\frac{1-n}{An})$

 $=  \frac{C}{An\sqrt{2\pi}}  (|\frac{1-n}{An}|)^{\frac{1}{n}-1} \exp(\frac{n-1}{n})$

We thus get 

$\sigma =\begin{cases}

\frac{C}{An\sqrt{2\pi}}  (|\frac{1-n}{An}|)^{\frac{1}{n}-1} \exp(\frac{n-1}{n})  & \text{ if } A < 0 \text{ and } x_i + (|\frac{1-n}{An}|)^{\frac{1}{n}} < x_f \\

\frac{C}{An\sqrt{2\pi}}  (x_f-x_i)^{1-n} \exp(-A(x_f-x_i)^n)  & \text{ else } \\

\end{cases}$

Our ultimate form looks like:

$f(x)\approx 1+C\Phi(\frac{x-\mu}{\sigma})$

where

$\mu=\begin{cases}
x_i + (|\frac{1-n}{An}|)^{\frac{1}{n}} & \text{ if } A < 0 \text{ and } x_i + (|\frac{1-n}{An}|)^{\frac{1}{n}} < x_f \\

x_f & \text{ else }
\end{cases}$

$C=2(f(\mu)-1)$

$\sigma = \frac{C}{An\sqrt{2\pi}}  (\mu-x_i)^{1-n} \exp(-A(\mu-x_i)^n)$

### Case 2: $n = 1$

We cannot take advantage of the fact that $f'(x_i)=0$. This means that the approximation cannot follow the same form as the others. Instead, we add an additional free variable representing the constant term in the approximation.

$f(x)=\exp(A(x-x_i)) \approx B+C\Phi(\frac{x-\mu}{\sigma})$

When $A < 0$, we can use $x=x_i$ as the inflection point. On the other hand, when $A > 0$, we use $x = x_f$ as the inflection point:

$\mu=\begin{cases}
x_i & \text{ if } A < 0 \\

x_f & \text{ if } A > 0
\end{cases}$

We can then approximate the variance by solving for the the slope at the mean:

$\frac{d}{dx}[B+C\Phi(\frac{x-\mu}{\sigma})]=\frac{1}{\sigma}C\phi(0)

=\frac{1}{\sigma}C\frac{1}{\sqrt{2\pi}}

=\frac{C}{\sigma \sqrt{2\pi}}$

$\frac{d}{dx}f(\mu) = A\exp(A(\mu-x_i)^n)$

Letting the slope be equal for the approximation:

$\frac{C}{\sigma \sqrt{2\pi}} =  A \exp(A(\mu-x_i))$

$\sigma=\frac{C}{A\sqrt{2\pi}} \exp(-A(\mu-x_i))$

Next, we take a both endpoints of the interval, and use the values at each endpoint to determine $B$ and $C$.

#### Case 2a: $\mu = x_i$

$f(x_i) = \exp(0) = 1 \approx B+C\Phi(0) = B + \frac{C}{2}$

$f(x_f) = \exp(A(x_f-x_i)) \approx B+C\Phi(\frac{x_f-x_i}{\sigma})$

This becomes a system of equations we can solve:

$\begin{cases}
B+C\Phi(\frac{x_f-x_i}{\sigma}) =  \exp(A(x_f-x_i)) \\

B + \frac{C}{2} = 1
\end{cases}$ 

Subtracting the two equations, we get:

$(\Phi(\frac{x_f-x_i}{\sigma}) - \frac{1}{2})C= \exp(A(x_f-x_i)) - 1$

From the previously derived expression for $\sigma$:

$\Phi(\frac{x_f-x_i}{\sigma})= \Phi((x_f-x_i) \frac{A\sqrt{2\pi}}{C} \exp(A(\mu-x_i)))$

$= \Phi((x_f-x_i) \frac{A\sqrt{2\pi}}{C})$

<aside>
💡

We can approximate $\Phi(t)$ as a clamped cubic sigmoid:

$\Phi(t) = \begin{cases}
0 & \text{ if } x < -M \\

3 (\frac{x+M}{2M})^2 - 2 (\frac{x+M}{2M})^3  & \text{ if } -M \le x \le M \\

1  & \text{ if } x > M \\
\end{cases}$

The value of $M$ controls the curve. To minmize $L_2$ norm, $M \approx 2.050$.

</aside>

Thus, we can use the expansion:

 $\Phi((x_f-x_i) \frac{A\sqrt{2\pi}}{C}) \approx 3((x_f-x_i) \frac{A\sqrt{2\pi}}{2MC}+\frac{1}{2})^2-2((x_f-x_i) \frac{A\sqrt{2\pi}}{2MC}+\frac{1}{2})^3$

Letting $K= (x_f-x_i) \frac{A\sqrt{2\pi}}{2M}$

 $\Phi((x_f-x_i) \frac{A\sqrt{2\pi}}{C}) \approx 3(\frac{K}{C}+\frac{1}{2})^2-2(\frac{K}{C}+\frac{1}{2})^3$

 $= 3\frac{K^2}{C^2} + 3\frac{K}{C}  +\frac{3}{4}-2 \frac{K^3}{C^3} -3\frac{K^2}{C^2} -\frac{3K}{2C} -\frac{1}{4}$ 

 $=  -2 \frac{K^3}{C^3} +\frac{3K}{2C} +\frac{1}{2}$ 

We can thus solve for C:

$(-2 \frac{K^3}{C^3} +\frac{3K}{2C} +\frac{1}{2}  - \frac{1}{2})C= \exp(A(x_f-x_i)) - 1$

$(-2 \frac{K^3}{C^3} +\frac{3K}{2C} )C= \exp(A(x_f-x_i)) - 1$

$-2 \frac{K^3}{C^2} +\frac{3K}{2} = \exp(A(x_f-x_i)) - 1$

$-2K^3 = (\exp(A(x_f-x_i)) - 1 - \frac{3K}{2})C^2$

$C^2 = \frac{2K^3}{\frac{3K}{2} + 1 - \exp(A(x_f-x_i))}$

We should expect a negative value for $C$, in order to fit a downward sloping curve. thus:

$C =-\sqrt{ \frac{2K^3}{\frac{3K}{2} + 1 - \exp(A(x_f-x_i))} } = -C_0$

where

$C_0 =\sqrt{ \frac{2K^3}{\frac{3K}{2} + 1 - \exp(A(x_f-x_i))} }$

For the case where $\Phi(\frac{2KM}{C}) \approx 0$:

$\frac{2KM}{C} < -M$

$\frac{2K}{C} < -1$

$2K > -C$

$K > -\frac{1}{2}C$

We should never encounter this case since $C < 0$ and $K < 0$.

For the case where $\Phi(\frac{2KM}{C}) \approx 1$:

$\frac{2KM}{C} > M$

$\frac{2K}{C} > 1$

$2K < C$

$K < -\frac{1}{2}C_0$

$(1-\frac{1}{2})C= \exp(A(x_f-x_i)) - 1$

$\frac{1}{2}C= \exp(A(x_f-x_i)) - 1$

$C= 2\exp(A(x_f-x_i)) - 2$

We get the final expression for $C$:

$C = \begin{cases}
2\exp(A(x_f-x_i)) - 2  & \text{ if } K < -\frac{1}{2}C_0 \\

-C_0 & \text{ else } \\

\end{cases}$

where:

- $K= (x_f-x_i) \frac{A\sqrt{2\pi}}{2M}$
- $C_0 =\sqrt{ \frac{2K^3}{\frac{3K}{2} + 1 - \exp(A(x_f-x_i))} }$

It follows that:

$B=1-\frac{C}{2}$

#### Case 2b: $\mu = x_f$

$f(x_i) = \exp(0) = 1 \approx B+C\Phi(\frac{x_i-x_f}{\sigma}) = B + C(1-\Phi(\frac{x_f-x_i}{\sigma}))$

$f(x_f) = \exp(A(x_f-x_i)) \approx B+\frac{C}{2}$

This becomes a system of equations we can solve:

$\begin{cases}
 B + C(1-\Phi(\frac{x_f-x_i}{\sigma})) = 1 \\

B + \frac{C}{2} = \exp(A(x_f-x_i))
\end{cases}$ 

Subtracting the two equations, we get:

$(1 - \Phi(\frac{x_f-x_i}{\sigma}) - \frac{1}{2})C= 1 - \exp(A(x_f-x_i))$ 

$(\frac{1}{2} - \Phi(\frac{x_f-x_i}{\sigma}))C= 1 - \exp(A(x_f-x_i))$ 

$(\Phi(\frac{x_f-x_i}{\sigma}) - \frac{1}{2} )C=  \exp(A(x_f-x_i)) - 1$

This expression is identical to that in Case 2a. Thus, we can reuse the same formulation but with an altered $\sigma$:

$\sigma=\frac{C}{A\sqrt{2\pi}} \exp(-A(\mu-x_i)) = \frac{C}{A\sqrt{2\pi}} \exp(-A(x_f-x_i))$

$K= (x_f-x_i) \frac{A\sqrt{2\pi}}{2M}\exp(A(x_f-x_i))$

$C_0=\sqrt{ \frac{2K^3}{\frac{3K}{2} + 1 - \exp(A(x_f-x_i))} }$

However, the cases change, as we find that $\Phi(\frac{x_f-x_i}{\sigma})  \approx 1$. This corresponds to when:

$\frac{2KM}{C} > M$

$\frac{2K}{C} > 1$

$2K > C$

$K > \frac{1}{2}C$

We get the final expression for $C$:

$C = \begin{cases}
2\exp(A(x_f-x_i)) - 2  & \text{ if } K > \frac{1}{2}C_0 \\

C_0 & \text{ else } \\

\end{cases}$

It follows that:

$B = \exp(A(x_f-x_i))- \frac{1}{2}C$ 

#### Final Expression

$f(x)=\exp(A(x-x_i)) \approx B+C\Phi(\frac{x-\mu}{\sigma})$

where:

$\mu=\begin{cases}
x_i & \text{ if } A < 0 \\

x_f & \text{ if } A > 0
\end{cases}$

$C = \begin{cases}
2\exp(A(x_f-x_i)) - 2  & \text{ if } |K| > \frac{1}{2}C_0 \\

-C_0  & \text{ if } |K| \le \frac{1}{2}C_0 \text{ and } A \le 0 \\

C_0  & \text{ if } |K| \le \frac{1}{2}C_0 \text{ and } A > 0 \\

\end{cases}$

$\sigma=\frac{C}{A\sqrt{2\pi}} \exp(-A(\mu-x_i))$

$B = \begin{cases}
1-\frac{C}{2} & \text{ if } A < 0 \\

\exp(A(x_f-x_i))- \frac{1}{2}C  & \text{ if } A > 0
\end{cases}$

$K = \begin{cases}
(x_f-x_i) \frac{A\sqrt{2\pi}}{2M} & \text{ if } A < 0 \\

(x_f-x_i) \frac{A\sqrt{2\pi}}{2M}\exp(A(x_f-x_i)) & \text{ if } A > 0
\end{cases}$

$C_0=\sqrt{ \frac{2K^3}{\frac{3K}{2} + 1 - \exp(A(x_f-x_i))} }$

### Case 3: $n = 0$

This case is simple, since it corresponds simply to

$f(x)=\exp(A)$