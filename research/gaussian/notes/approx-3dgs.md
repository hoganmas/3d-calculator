# NOTES: Approximating Gaussian CDF Transmittance

# Introduction

For Volume Rendering, we need to efficiently determine the value of the integral:

$\int_{z_i}^{z_f} dz\ g(z)\exp(- \int _z^{z_f}dz'  f(z'))$ 

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

Let $f(z)=\sum_i^N w_i \phi_i(z)$. That is, the input attenuation coefficient corresponds to a weighted sum of $N$ gaussians.

We can express $T(z)$ as the following:

$T(z)= \exp(- \int _z^{z_f}dz' \sum_i w_i \phi_i(z'))$

$=\exp(- \sum_i w_i \int _z^{z_f}dz' \phi_i(z'))$

$=\exp(- \sum_i w_i |\Phi_i(z')|_z^{z_f})$

$=\exp(- \sum_i w_i (\Phi_i(z_f)-\Phi_i(z)))$

$=\exp(\sum_i w_i (\Phi_i(z) - \Phi_i(z_f)))$

$=\exp(\sum_i w_i \Phi_i(z)) \exp(-\sum_i w_i \Phi_i(z_f)))$

$=A\exp(\sum_i w_i \Phi_i(z))$

$=A \prod_i \exp(w_i \Phi_i(z))$

where $A=\exp(-\sum_i w_i \Phi_i(z_f)))$.

<aside>
💡

Additionally, we can substitute the following approximation (See Appendix A):

$\exp(w_i\Phi(\frac{z-\mu_i}{\sigma_i})) \approx 1+(e^{w_i}-1) \Phi(\frac{z-\mu_i-\frac{\sigma}{A}(-\sqrt{2\pi} + \sqrt{2\pi+2A^2})}{\frac{\sigma |e^A-1| \exp(-A\Phi(\frac{\mu}{\sigma}))}{A  \phi(\frac{\mu}{\sigma}) \sqrt{2\pi}}})$

$= 1+(e^{w_i}-1) \Phi(h_i(z))$

where:

- $h_i(z)=\frac{z-\mu_i-\frac{\sigma}{A}(-\sqrt{2\pi} + \sqrt{2\pi+2A^2})}{\frac{\sigma |e^A-1| \exp(-A\Phi(\frac{\mu}{\sigma}))}{A  \phi(\frac{\mu}{\sigma}) \sqrt{2\pi}}}$

</aside>

$=A \prod_i (1+(e^{w_i}-1) \Phi(h_i(z)))$

We can expand this product out:

<aside>
💡

For two gaussians:

$\prod_i^2 (1+(e^{w_i}-1) \Phi(h_i(z)))$

$=(1+(e^{w_1}-1) \Phi(h_1(z)))(1+(e^{w_2}-1) \Phi(h_2(z)))$

$=1+(e^{w_1}-1) \Phi(h_1(z))+(e^{w_2}-1) \Phi(h_2(z)) + (e^{w_1}-1)(e^{w_2}-1)  \Phi(h_1(z)) \Phi(h_2(z))$ 

We can substitute the approximation:

 $\Phi(\frac{z-\mu_1}{\sigma_1}) \Phi(\frac{z-\mu_2}{\sigma_2}) \approx \begin{cases}
\Phi(\frac{z-\mu_1}{\sigma_1}) & \text{ if } \mu_1 > \mu_2 \\

\Phi(\frac{z-\mu_2}{\sigma_2}) & \text{ if } \mu_1 < \mu_2 \\
\end{cases}$

If we assume that gaussians are ordered by depth:

$\approx 1+(e^{w_1}-1) \Phi(h_1(z))+(e^{w_2}-1) \Phi(h_2(z)) + (e^{w_1}-1)(e^{w_2}-1)   \Phi(h_2(z))$ 

$= 1+(e^{w_1}-1) \Phi(h_1(z))+ (e^{w_2}-1)(1+e^{w_1}-1)   \Phi(h_2(z))$ 

$= 1+(e^{w_1}-1) \Phi(h_1(z))+ (e^{w_2}-1)e^{w_1}\ \Phi(h_2(z))$ 

We can repeat this for three depth-ordered gaussians:

$\prod_i^3 (1+(e^{w_i}-1) \Phi(h_i(z)))$

$=[1+(e^{w_1}-1) \Phi(h_1(z))+ (e^{w_2}-1)e^{w_1}\ \Phi(h_2(z))][1+(e^{w_3}-1) \Phi(h_3(z))]$

$= 1+(e^{w_1}-1) \Phi(h_1(z))+ (e^{w_2}-1)e^{w_1}\ \Phi(h_2(z)) + \\

[1+(e^{w_1}-1) \Phi(h_1(z))+ (e^{w_2}-1)e^{w_1}\ \Phi(h_2(z))](e^{w_3}-1) \Phi(h_3(z))$

$= 1+(e^{w_1}-1) \Phi(h_1(z))+ (e^{w_2}-1)e^{w_1}\ \Phi(h_2(z)) + \\

(e^{w_3}-1) \Phi(h_3(z)) +(e^{w_1}-1) \Phi(h_1(z))(e^{w_3}-1) \Phi(h_3(z)) + (e^{w_2}-1)e^{w_1}\ \Phi(h_2(z))(e^{w_3}-1) \Phi(h_3(z))$

$= 1+(e^{w_1}-1) \Phi(h_1(z))+ (e^{w_2}-1)e^{w_1}\ \Phi(h_2(z)) + \\

(e^{w_3}-1) \Phi(h_3(z)) +(e^{w_1}-1) (e^{w_3}-1) \Phi(h_3(z)) + (e^{w_2}-1)e^{w_1}(e^{w_3}-1) \Phi(h_3(z))$

$= 1+(e^{w_1}-1) \Phi(h_1(z))+ (e^{w_2}-1)e^{w_1}\ \Phi(h_2(z)) + \\

[(e^{w_3}-1) +(e^{w_1}-1) (e^{w_3}-1) + (e^{w_2}-1)e^{w_1}(e^{w_3}-1)] \Phi(h_3(z))$

$= 1+(e^{w_1}-1) \Phi(h_1(z))+ (e^{w_2}-1)e^{w_1}\ \Phi(h_2(z)) + \\

(e^{w_3}-1)[1 +(e^{w_1}-1)  + (e^{w_2}-1)e^{w_1}] \Phi(h_3(z))$

$= 1+(e^{w_1}-1) \Phi(h_1(z))+ (e^{w_2}-1)e^{w_1}\ \Phi(h_2(z)) + \\

(e^{w_3}-1)[e^{w_1} + (e^{w_2}-1)e^{w_1}] \Phi(h_3(z))$

$= 1+(e^{w_1}-1) \Phi(h_1(z))+ (e^{w_2}-1)e^{w_1}\ \Phi(h_2(z)) + 

(e^{w_3}-1)e^{w_2}e^{w_1}\ \Phi(h_3(z))$

</aside>

We can employ this expansion recursively, and find that the final result reduces down to an approximate sum of $N$ gaussians and a constant term of $1$.

For a set of $N$ depth-ordered gaussians:

$\approx A (1+ \sum_i B_i\Phi(h_i(z))$

where $B_i= (e^{w_i}-1) \prod_{j}^{i-1} e^{w_j}$

Thus, we have the final form of our approximated transmittance:

$T(z) \approx A (1+ \sum_i B_i\Phi(h_i(z))$,

where:

- $A=\exp(-\sum_{i=0}^N w_i \Phi_i(z_f)))$
- $B_i= (e^{w_i}-1) \prod_{j=0}^{i-1} e^{w_j}$
- $h_i(z)=\frac{z-\mu_i-\frac{\sigma}{A}(-\sqrt{2\pi} + \sqrt{2\pi+2A^2})}{\frac{\sigma |e^A-1| \exp(-A\Phi(\frac{\mu}{\sigma}))}{A  \phi(\frac{\mu}{\sigma}) \sqrt{2\pi}}}$

## Calculating Radiative Transfer

Now, we can return to our original expression:

$\int_{z_i}^{z_f} dz\ g(z)\exp(- \int _z^{z_f}dz'  f(z'))$

$\approx \int_{z_i}^{z_f} dz\ g(z) *A (1+ \sum_i B_i\Phi(h_i(z))$

$= A(\int_{z_i}^{z_f} dz\ g(z))+ A\sum_i B_i\int_{z_i}^{z_f} dz\ g(z)\ \Phi(h_i(z)$

Assuming our emission function is also a sum of weighted gaussians: $g(z)= \sum_j c_j\phi(\frac{z-\mu_j}{\sigma_j})$

$= A(\int_{z_i}^{z_f} dz\ \sum_j c_j\phi(\frac{z-\mu_j}{\sigma_j}))+ A\sum_i B_i\int_{z_i}^{z_f} dz\  \sum_j c_j\phi(\frac{z-\mu_j}{\sigma_j})\ \Phi(h_i(z)$

$= A(\sum_j  c_j\int_{z_i}^{z_f} dz\ \phi(\frac{z-\mu_j}{\sigma_j}))+ A\sum_i \sum_j B_i c_j\int_{z_i}^{z_f} dz\  \phi(\frac{z-\mu_j}{\sigma_j})\ \Phi(h_i(z)$

<aside>
💡

$\sum_j  c_j\int_{z_i}^{z_f} dz\ \phi(\frac{z-\mu_j}{\sigma_j})$

$=\sum_j  \sigma_j \int_{\frac{z_i-\mu_j}{\sigma_j}}^{\frac{z_f-\mu_j}{\sigma_j}} du\ \phi(u)$

$=\sum_j  \sigma_j | \Phi(u) |_{\frac{z_i-\mu_j}{\sigma_j}}^{\frac{z_f-\mu_j}{\sigma_j}}$ 

$=\sum_j  \sigma_j (\Phi(\frac{z_f-\mu_j}{\sigma_j}) - \Phi(\frac{z_0-\mu_j}{\sigma_j}))$

$\int_{z_i}^{z_f} dz\  \phi(\frac{z-\mu_j}{\sigma_j})\ \Phi(\frac{z-\mu_i}{\sigma_i})$

$= \int_{z_i}^{z_f} dz\  \phi(\frac{z-\mu_j}{\sigma_j})\ \Phi(\frac{z-\mu_i}{\sigma_i})$

$= \sigma_j \int_{\frac{z_i-\mu_j}{\sigma_j}}^{\frac{z_f-\mu_j}{\sigma_j}} du\  \phi(u)\ \Phi(\frac{\sigma_j u + \mu_j-\mu_i}{\sigma_i})$

$= \sigma_j[ \Phi_2(\frac{\mu_j-\mu_i}{\sqrt{\sigma_i^2 + \sigma_j^2}},\frac{z_f-\mu_j}{\sigma_j};\rho=-\frac{\sigma_j}{\sqrt{\sigma_i^2 + \sigma_j^2}})

-  \Phi_2(\frac{\mu_j-\mu_i}{\sqrt{\sigma_i^2 + \sigma_j^2}},\frac{z_i-\mu_j}{\sigma_j};\rho=-\frac{\sigma_j}{\sqrt{\sigma_i^2 + \sigma_j^2}})]$

</aside>

$= A \sum_j (  \sigma_j (\Phi(\frac{z_f-\mu_j}{\sigma_j}) - \Phi(\frac{z_0-\mu_j}{\sigma_j})) +  \sum_i B_i c_j \sigma_j[ \Phi_2(\frac{\mu_j-\mu_i}{\sqrt{\sigma_i^2 + \sigma_j^2}},\frac{z_f-\mu_j}{\sigma_j};\rho=-\frac{\sigma_j}{\sqrt{\sigma_i^2 + \sigma_j^2}})

-  \Phi_2(\frac{\mu_j-\mu_i}{\sqrt{\sigma_i^2 + \sigma_j^2}},\frac{z_i-\mu_j}{\sigma_j};\rho=-\frac{\sigma_j}{\sqrt{\sigma_i^2 + \sigma_j^2}})])$

---

# Appendix

## Appendix A: Approximating exponentiated Gaussian CDF

Let $f(x)=\exp(A\Phi(\frac{1}{\sigma}x))$. This looks roughly like a scaled and translated Gaussian CDF.

We can find the limits as such:

$\lim_{x \to -\infty} f(x)=  \exp(0)=1$ 

$\lim_{x \to \infty} f(x)= \exp(A)=e^A$

Thus the form of the approximation should look like:

$f(x)=\exp(A \Phi(\frac{1}{\sigma}x)) \approx 1+(e^A-1)\Phi(\frac{x-\mu}{\nu})$

We can find the inflection point (center) as such:

 $\frac{d}{dx}f(x)=\frac{A}{\sigma} \exp(A\Phi(\frac{1}{\sigma}x))\phi(\frac{1}{\sigma}x)$ 

 $\frac{d^2}{dx^2}f(x)=\frac{A^2}{\sigma^2} \exp(A\Phi(\frac{1}{\sigma}x)) \phi(\frac{1}{\sigma}x)^2 

+ \frac{A}{\sigma} \exp(A\Phi(\frac{1}{\sigma}x)) \frac{d}{dx} \phi(\frac{1}{\sigma}x)$

 $=\frac{A^2}{\sigma^2} \exp(A\Phi(\frac{1}{\sigma}x)) \phi(\frac{1}{\sigma}x)^2 

+ \frac{A}{\sigma} \exp(A\Phi(\frac{1}{\sigma}x)) (-\frac{x}{\sigma^2} \phi(\frac{1}{\sigma}x)) = 0$

 

 $A \phi(\frac{1}{\sigma}x)

- \frac{x}{\sigma} =0$ 

Approximating $\phi(x) \approx \frac{1}{\sqrt{2\pi}}(1-\frac{x^2}{2})$

 $A \frac{1}{\sqrt{2\pi}}(1-\frac{x^2}{2\sigma^2})

- \frac{x}{\sigma} =0$ 

 $A \frac{1}{\sqrt{2\pi}}(2\sigma^2-x^2)

- 2\sigma x =0$ 

 $2\sigma^2-x^2
- 2 \sigma \frac{\sqrt{2 \pi}}{A}x =0$ 

 $x^2
+ 2 \sigma \frac{\sqrt{2\pi}}{A} x - 2\sigma^2 =0$ 

 $x=-\sigma \frac{\sqrt{2\pi}}{A} \pm \sqrt{\frac{2\pi\sigma^2}{A^2}+2\sigma^2}$ 

 $=\frac{\sigma}{A}(- \sqrt{2\pi} \pm \sqrt{2\pi+2A^2})$ 

Since we want a positive mean, we will opt for

$\mu=\frac{\sigma}{A}(- \sqrt{2\pi} + \sqrt{2\pi+2A^2})$

Finally, we can approximate the variance by solving for the the slope at the mean:

$\frac{d}{dx}[1+(e^A-1)\Phi(\frac{x-\mu}{\nu})]=\frac{1}{\nu}(e^A-1)\phi(\frac{x-\mu}{\nu})$

At $x = \mu$:

$\frac{d}{dx}[1+(e^A-1)\Phi(\frac{x-\mu}{\nu})]=\frac{1}{\nu}(e^A-1)\phi(0)

=\frac{1}{\nu}(e^A-1)\frac{1}{\sqrt{2\pi}}

=\frac{e^A-1}{\nu \sqrt{2\pi}}$

$\frac{d}{dx}f(\mu)=\frac{A}{\sigma} \exp(A\Phi(\frac{\mu}{\sigma}))\phi(\frac{\mu}{\sigma})$

Letting the slope be equal for the approximation:

$\frac{e^A-1}{\nu \sqrt{2\pi}}=\frac{A}{\sigma} \exp(A\Phi(\frac{\mu}{\sigma}))\phi(\frac{\mu}{\sigma})$

$\nu=\frac{\sigma (e^A-1)}{A \sqrt{2\pi} \exp(A\Phi(\frac{\mu}{\sigma}))\phi(\frac{\mu}{\sigma})}$

$=\frac{\sigma (e^A-1) \exp(-A\Phi(\frac{\mu}{\sigma}))}{A  \phi(\frac{\mu}{\sigma}) \sqrt{2\pi}}$

Our ultimate form looks like:

$f(x)=\exp(A \Phi(\frac{1}{\sigma}x)) \approx 1+(e^A-1)\Phi(\frac{x+\frac{\sigma}{A}(\sqrt{2\pi} - \sqrt{2\pi+2A^2})}{\frac{\sigma |e^A-1| \exp(-A\Phi(\frac{\mu}{\sigma}))}{A  \phi(\frac{\mu}{\sigma}) \sqrt{2\pi}}})$