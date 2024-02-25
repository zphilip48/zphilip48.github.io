---
layout: post
title:  "Gradient boosting"
date:   2023-01-26
categories: LEARNING
tags: AI
---

## Gradient boosting <font color=red>performs gradient descent</font>

[Entropy of the Gaussian (gregorygundersen.com)](https://gregorygundersen.com/blog/2020/09/01/gaussian-entropy/)

**<font color=red>总结：可以说 loss function (下面列出) 是 $\hat{y}$ 的函数， 那么它的gradient boosting 其实是gradient descent，只不过由于其gradient 导数 : $-2(\mathbf{y}-\hat{\mathbf{y}})$ of MSE and $-\operatorname{sign}(\mathbf{y}-\hat{\mathbf{y}})$ of MAE 恰好是相反反向的 $\mathbf{y}-\hat{\mathbf{y}}$  which is residual vector $\Delta_m(X)$, 所以导致boosting公式的形态.</font>**
$$
\begin{array}{cc}
\text { Gradient descent } & \text { Gradient boosting } \\
\mathbf{x}_t=\mathbf{x}_{t-1}-\eta \nabla f\left(\mathbf{x}_{t-1}\right) & \hat{\mathbf{y}}_m=\hat{\mathbf{y}}_{m-1}+\eta\left(-\nabla L\left(\mathbf{y}, \hat{\mathbf{y}}_{m-1}\right)\right)
\end{array}
$$
When *L* is the MSE loss function, *L*'s gradient is the residual vector and a gradient descent optimizer should chase that residual, which is exactly what the gradient boosting machine does as well. When *L* is the MAE loss function, *L*'s gradient is the sign vector, leading gradient descent and gradient boosting to step using the sign vector.

The implications of all of this fancy footwork is that **we can use a GBM to optimize any differentiable loss function by training our weak models on the negative of the loss function gradient** (with respect to the previous approximation). Understanding this derivation from the GBM recurrence relation to gradient descent update equation is much harder to see without the $\hat{\mathbf{y}}_m=F_m(X)$ substitution.

### **Boosting as gradient descent in prediction space**

Our goal is to show that training a GBM（Gradient Boosting Machine） is performing gradient-descent minimization on some loss function between our true target, $\mathrm{y}$, and our approximation, $\hat{\mathbf{y}}_m=F_m(X)$. That means showing that adding weak models, $\Delta_m$, to our GBM additive model:
$$
F_m(X)=F_{m-1}(X)+\eta \Delta_m(X)
$$
**is performing gradient descent in some way**.   

It makes sense that nudging our approximation, $\hat{\mathbf{y}}$, closer and closer to the true target y would be performing gradient descent. For example, at each step, the residual $\mathbf{y}-\hat{\mathbf{y}}$ gets smaller. We must be minimizing some function related to the distance between the true target and our approximation. Let's revisit our golfer analogy and visualize the squared error between the approximation and the true value, $\left(y-F_m\right)^2$ :

<img src="/assets/Gradient%20boosting.assets/golf-MSE.png" alt="img" style="zoom: 25%;" />

**The key insight**
The key to unlocking the relationship for more than one observation is to see that the residual, **$y-\hat{y}$, is a direction vector**. It's not just the magnitude of the difference. Moreover, the vector points in the direction of a better approximation and, hence, a smaller loss between the true $y$ and $\hat{y}$ vectors. That <font color=red>suggests that the direction vector is also (the negative of) a loss function gradient</font>. **Chasing the direction vector in a GBM is chasing the (negative) gradient of a loss function via gradient descent.**

### **The MSE function gradient**

To uncover the loss function optimized by a GBM **whose $\Delta_m$ weak models are trained on the residual vector**, we just have to integrate the residual $\mathbf{y}-\hat{\mathbf{y}}$. It's actually easier, though, to go the other direction and compute the gradient of the MSE loss function to show that it is the residual vector. The MSE loss function computed from $N$ observations in matrix $X=\left[\mathbf{x}_1 \ldots \mathbf{x}_N\right]$ is:
$$
L\left(\mathbf{y}, F_M(X)\right)=\frac{1}{N} \sum_{i=1}^N\left(y_i-F_M\left(\mathrm{x}_i\right)\right)^2
$$
but let's substitute $\hat{y}$ for the model output, $F_M(X)$, to make the equation more clear:
$$
L(\mathbf{y}, \hat{\mathbf{y}})=\frac{1}{N} \sum_{i=1}^N\left(y_i-\hat{y}_i\right)^2
$$
Also, since $N$ is a constant once we start boosting, and $f(x)$ and $c f(x)$ have the same $x$ minimum point, let's drop the $\frac{1}{N}$ constant:
$$
L(\mathbf{y}, \hat{\mathbf{y}})=\sum_{i=1}^N\left(y_i-\hat{y}_i\right)^2
$$
Now, let's take the partial derivative of the loss function with respect to a specific $\hat{y}_j$ approximation:
$$
\begin{aligned}
\frac{\partial}{\partial \hat{y}_j} L(\mathbf{y}, \hat{\mathbf{y}}) & =\frac{\partial}{\partial \hat{y}_j} \sum_{i=1}^N\left(y_i-\hat{y}_i\right)^2 \\
& =\frac{\partial}{\partial \hat{y}_j}\left(y_j-\hat{y}_j\right)^2 \\
& =2\left(y_j-\hat{y}_j\right) \frac{\partial}{\partial \hat{y}_j}\left(y_j-\hat{y}_j\right) \\
& =-2\left(y_j-\hat{y}_j\right)
\end{aligned}
$$
(We can remove the summation because the partial derivative of $L$ for $i \neq j$ is 0 .)
That means the gradient with respect to $\hat{y}$ is:
$$
\nabla_{\hat{\mathbf{y}}} L(\mathbf{y}, \hat{\mathbf{y}})=-2(\mathbf{y}-\hat{\mathbf{y}})
$$
Dropping the constant in front again leaves us with the <font color=red>**gradient being the same as the residual vector: $\mathbf{y}-\hat{\mathbf{y}}$**</font>. **So, chasing the residual vector in a GBM is chasing the gradient vector of the MSE $L_2$ loss function while performing gradient descent**.

### **The MAE(Mean Absolute Error) function gradient**

Let's see what happens with the MAE loss function:
$$
L(\mathbf{y}, \hat{\mathbf{y}})=\sum_{i=1}^N\left|y_i-\hat{y}_i\right|
$$
The partial derivative with respect to a specific approximation $\hat{y}_j$ is:
$$
\begin{aligned}
\frac{\partial}{\partial \hat{y}_j} L(\mathbf{y}, \hat{\mathbf{y}}) & =\frac{\partial}{\partial \hat{y}_j} \sum_{i=1}^N\left|y_i-\hat{y}_i\right| \\
& =\frac{\partial}{\partial \hat{y}_j}\left|y_j-\hat{y}_j\right| \\
& =\operatorname{sign}\left(y_j-\hat{y}_j\right) \frac{\partial}{\partial \hat{y}_j}\left(y_j-\hat{y}_j\right) \\
& =-\operatorname{sign}\left(y_j-\hat{y}_j\right)
\end{aligned}
$$
giving gradient:
$$
\nabla_{\hat{\mathbf{y}}} L(\mathbf{y}, \hat{\mathbf{y}})=-\operatorname{sign}(\mathbf{y}-\hat{\mathbf{y}})
$$
This shows that **chasing the sign vector in a GBM is chasing the gradient vector of the MAE $L_1$ loss function while performing gradient descent.**

### Function space is prediction space

把函数空间gradient descent 解释为预测值的gradient descent ...其实差不多

Most GBM articles follow Friedman's notation (on page 4, equation for $g_m\left(\mathbf{x}_i\right)$ ) and describe the gradient as this scary-looking expression for the partial derivative with respect to our approximation of $y_i$ for observation $\mathbf{x}_i$ :
$$
\left[\frac{\partial L\left(y_i, F\left(\mathbf{x}_i\right)\right)}{\partial F\left(\mathbf{x}_i\right)}\right]_{F(\mathbf{x})=F_{m-1}(\mathbf{x})}
$$
Hmm... let's see if we can tease this apart. First, evaluate the expression according to the subscript, $F(\mathrm{x})=F_{m-1}(\mathrm{x})$
$$
\frac{\partial L\left(y_i, F_{m-1}\left(\mathrm{x}_i\right)\right)}{\partial F_{m-1}\left(\mathrm{x}_i\right)}
$$
Next, let's remove the $i$ index variable to look at the entire gradient, instead of a specific observation's partial derivative:
$$
\nabla_{F_{m-1}(X)} L\left(\mathbf{y}, F_{m-1}(X)\right)
$$
But, what does it mean to take the partial derivative with respect to a function, $F_{m-1}(X)$ ? Here is where we find it much easier to understand the gradient expression using $\hat{\mathbf{y}}_m$, rather than $F_m(X)$. Both are just vectors, but it's easier to see that when we use a simple variable name, $\hat{\mathbf{y}}_m$. Substituting, we get a gradient expression that references two vector variables $\mathbf{y}$ and $\hat{\mathbf{y}}_{m-1}$ :
$$
\nabla_{\hat{\mathbf{y}}_{m-1}} L\left(\mathbf{y}, \hat{\mathbf{y}}_{m-1}\right)
$$
Variable $\hat{\mathbf{y}}_{m-1}$ is a position in "function space," which just means a vector result of evaluating function $F_{m-1}(X)$. This is why GBMs perform "gradient descent in function space," but it's easier to think of it as "gradient descent in prediction space" where $\hat{\mathbf{y}}_{m-1}$ is our prediction.

有些解释为泰勒展开式：

> 在数学中，泰勒公式 (英语: Taylor's Formula) 是一个用函数在某点的信息描述其附近取值的公式。这个公式来自 于微积分的泰勒定理 (Taylor's theorem)，泰勒定理描述了一个可微函数，如果函数足㿟光滑的话，在已知函数在 某一点的各阶导数值的情况之下，泰勒公式可以用这些导数值做系数构建一个多项式来近似函数在这一点的邻域中 的值，这个称为泰勒多项式 (Taylor polynomial)。
> 相当于告诉我们可由利用泰勒冬项式的某些次项做原函数的近似。
> 泰勒定理:
> 设 $n$ 是一个正整数。如果定义在一个包含 $a$ 的区间上的函数 $f$ 在 $a$ 点处 $n+1$ 次可导，那么对于这个区间上的任意 x, 都有:

$$
f(x)=f(a)+\frac{f^{\prime}(a)}{1 !}(x-a)+\frac{f^{(2)}(a)}{2 !}(x-a)^2+\cdots+\frac{f^{(n)}(a)}{n !}(x-a)^n+R_n(x)
$$

> 其中的多项式称为函数在 $\mathrm{a}$ 处的泰勒展开式，剩余的 $R_n(x)$ 是泰勒公式的余项，是 $(x-a)^n$ 的高阶无穷小。
>
> <img src="/assets/Gradient%20boosting.assets/image-20230603235440862.png" alt="image-20230603235440862" style="zoom: 33%;" />
>
> 在梯度下降法中，我们可以看出，对于最终的最优解 $\theta^*$ ，是由初始值 $\theta_0$ 经过T次迭代之后得到 的，这里设 $\theta_0=-\frac{\delta L(\theta)}{\delta \theta_0}$ ，则 $\theta^*$ 为:

$$
\theta^*=\sum_{t=0}^T \alpha_t *\left[-\frac{\delta L(\theta)}{\delta \theta}\right]_{\theta=\theta_{t-1}}
$$

> 其中， $\left[-\frac{\delta L(\theta)}{\delta \theta}\right]_{\theta=\theta_{t-1}}$ 表示 $\theta$ 在 $\theta_{t-1}$ 处泰勒展开式的一阶导数。
> 在函数空间中，我们也可以借鉴梯度下降的思想，进行最优函数的搜索。对于模型的损失函数 $L(y, F(x))$ ，为了能够求解出最优的函数 $F^*(x)$ ，首先设置初始值为: $F_0(x)=f_0(x)$
> 以函数 $F(x)$ 作为一个整体，与梯度下降法的更新过程一致，假设经过 T次迭代得到最优的函数 $F^*(x)$ 为:

$$
F^*(x)=\sum_{t=0}^T f_t(x)
$$

> 其中， $f_t(x)$ 为:

$$
f_t(x)=-\alpha_t g_t(x)=-\alpha_t *\left[\frac{\delta L(y, F(x))}{\delta F(x)}\right]_{F(x)=F_{t-1}(x)}
$$

### How gradient boosting differs from gradient descent

Before finishing up, it's worth examining the differences between gradient descent and gradient boosting. To make things more concrete, let's consider applying gradient descent to train a neural network (NN). Training seeks to find the weights and biases, model parameters, that optimize the loss between the desired NN output, y , and the current output, $\hat{y}$. If we assume a squared error loss function, NN gradient descent training computes the next set of parameters by adding the residual vector, ,$y-\hat{y}$ to the current ![img](/assets/Gradient%20boosting.assets/eqn-8BB50605FF63759107F02187B2EE1A8D-depth000.00.svg+xml) (subtracting the squared error gradient).

**NN gradient descent 计算结果会作用在全部参数上， 而GBM只会作用在新的weak model上, 老的模型上的参数是固定不变的。而且直接作用在输出预测值上**

In contrast, GBMs are meta-models consisting of multiple weak models whose output is added together to get an overall prediction. The optimization we're concerned with here occurs, **not on the parameters of the weak models** themselves but, instead, on the composite model prediction, $\hat{y}_m = F_m(x)$. GBM training occurs on two levels then, one to train the weak models and one on the overall composite model. It is the overall training of the composite model that performs gradient descent by adding the residual vector (assuming a squared error loss function) to get the improved model prediction. **Training a NN using gradient descent tweaks model parameters whereas training a GBM tweaks (boosts) the model output**.

Also, training a NN with gradient descent directly adds a direction vector to the current ![img](/assets/Gradient%20boosting.assets/eqn-8BB50605FF63759107F02187B2EE1A8D-depth000.00.svg+xml), whereas training a GBM adds a weak model's approximation of the direction vector to the current output, $\hat{y}$. Consequently, it's likely that a GBM's MSE and MAE will decrease monotonically during training but, **given the weak approximations of our $\Delta_m$, monotonicity is not guaranteed**. The GBM loss function could bounce around a bit on its way down.

One final note on training regression trees used for weak models. The interesting thing is that, regardless of the direction vector (negative gradient), regression trees can always get away with using the squared error to compute node split points; i.e., even when the overall GBM is minimizing the absolute error. The difference between optimizing MSE and MAE error for the GBM is that the weak models train on different direction vectors. How the regression trees compute splits is not a big issue since the stumps are really weak and give really noisy approximations anyway.

### General algorithm with regression tree weak models

This general algorithm assumes the use of regression trees and is more complex than the specific algorithms for $L_2$ and $L_1$ loss. We need to compute the gradient of the loss function, **instead of just using the residual or sign of the residual**, and we need to compute weights for regression tree leaves. Each leaf, $l$, has weight value, $w$, that minimizes the $\sum_{i \in l} L\left(y_i, F_{m-1}\left(\mathbf{x}_i\right)+w\right)$ for all $\mathbf{x}_i$ observations within that leaf.

![image-20230603211406691](/assets/Gradient%20boosting.assets/image-20230603211406691.png)

==这里貌似没有讲如何生成regression tree to have the minimized MSE..  I think it could be the 最小二乘回归树生成算法 if the loss is MSE==

To see how this algorithm reduces to that for the $L_2$ loss function we have two steps to do. First, let $L\left(\mathbf{y}, \hat{\mathbf{y}}_{m-1}\right)=\left(\mathbf{y}-\hat{\mathbf{y}}_{m-1}\right)^2$, whose gradient gives the residual vector $\mathbf{r}_{m-1}=2\left(\mathbf{y}-\hat{\mathbf{y}}_{m-1}\right)$. **Second, show that leaf weight, $w$, is the mean of the residual of the observations in each leaf because the mean minimizes $\sum_{i \in l} L\left(y_i, F_{m-1}\left(\mathbf{x}_i\right)+w\right)$. That means minimizing**:
$$
\sum_{i \in l}\left(y_i-\left(F_{m-1}\left(\mathbf{x}_i\right)+w\right)\right)^2
$$
To find the minimal value of the function with respect to $w$, we take the partial derivative of that function with respect to $w$ and set to zero; then we solve for $w$. Here is the partial derivative:
$$
\frac{\partial}{\partial w} \sum_{i \in l}\left(y_i-\left(F_{m-1}\left(\mathbf{x}_i\right)+w\right)\right)^2=2 \sum_{i \in l}\left(y_i-\left(F_{m-1}\left(\mathbf{x}_i\right)+w\right)\right) \times \frac{\partial}{\partial w}\left(y_i-\left(F_{m-1}\left(\mathbf{x}_i\right)+w\right)\right)
$$
And since $\frac{\partial}{\partial w}\left(y_i-\left(F_{m-1}\left(\mathbf{x}_i\right)+w\right)\right)=1$, the last term drops off:
$$
2 \sum_{i \in l}\left(y_i-F_{m-1}\left(\mathbf{x}_i\right)-w\right)
$$
Now, set to 0 and solve for $w$ :
$$
\sum_{i \in l} 2 F_{m-1}\left(x_i\right)+2 w-2 y_i=0
$$
We can drop the constant by dividing both sides by 2 :
$$
\sum_{i \in l} F_{m-1}\left(\mathbf{x}_i\right)+w-y_i=0
$$
Then, pull out the $w$ term:
$$
\sum_{i \in l} F_{m-1}\left(\mathbf{x}_i\right)-y_i+\sum_{i \in l} w=0
$$
and move to the other side of the equation:
$$
\sum_{i \in l}\left(F_{m-1}\left(\mathbf{x}_i\right)-y_i\right)=-\sum_{i \in l} w
$$
We can simplify the $w$ summation to a multiplication:
$$
\sum_{i \in l}\left(F_{m-1}\left(\mathbf{x}_i\right)-y_i\right)=-n_l w \text { where } n_l \text { is number of obs. in } l
$$
Let's also flip the order of the elements within the summation to get the target variable first:
$$
\sum_{i \in l}\left(y_i-F_{m-1}\left(\mathbf{x}_i\right)\right)=n_l w
$$
Divide both sides of the equation by the number of observations in the leaf:
$$
w=\frac{1}{n_l} \sum_{i \in l}\left(y_i-F_{m-1}\left(\mathbf{x}_i\right)\right)
$$
**Finally, we see that leaf weights, $w$, should be the mean when the loss function is the mean squared error:**
$$
w=\operatorname{mean}\left(y_i-F_{m-1}\left(\mathbf{x}_i\right)\right)
$$
The mean is exactly what the leaves of the regression tree, trained on residuals, will predict.



### **Another version to have better understanding the relationship between $w$ (below is $\gamma$) and final model value $F_m(x)$**

refer to [All You Need to Know about Gradient Boosting Algorithm − Part 1. Regression | by Tomonori Masui | Towards Data Science](https://towardsdatascience.com/all-you-need-to-know-about-gradient-boosting-algorithm-part-1-regression-2520a34a502)

Gradient Boosting Algorithm

1. Initialize model with a constant value:

$$
F_0(x)=\underset{\gamma}{\operatorname{argmin}} \sum_{i=1}^n L\left(y_i, \gamma\right)
$$

> if it is squared loss in our regression case:
> $$
> \begin{aligned}
> \frac{\partial}{\partial \gamma} \sum_{i=1}^n L & =\frac{\partial}{\partial \gamma} \sum_{i=1}^n\left(y_i-\gamma\right)^2 \\
> & =-2 \sum_{i=1}^n\left(y_i-\gamma\right) \\
> & =-2 \sum_{i=1}^n y_i+2 n \gamma
> \end{aligned}
> $$
>
> $$
> \begin{aligned}
> & -2 \sum_{i=1}^n y_i+2 n \gamma=0 \\
> & n \gamma=\sum_{i=1}^n y_i \\
> & \gamma=\frac{1}{n} \sum_{i=1}^n y_i=\bar{y}
> \end{aligned}
> $$
>
>  so we have the $F_0(x)=\stackrel{*}{\gamma}=\bar{y}$

2. for $m=1$ to $M$ :
   2-1. Compute residuals $r_{i m}=-\left[\frac{\partial L\left(y_i, F\left(x_i\right)\right)}{\partial F\left(x_i\right)}\right]_{F(x)=F_{m-1}(x)}$ for $i=1, \ldots, n$ , 

   > previous we already know $r_{i m}=-\left[\frac{\partial L\left(y_i, F\left(x_i\right)\right)}{\partial F\left(x_i\right)}\right]_{F(x)=F_{m-1}(x)}$  is equal to $\nabla_{\hat{\mathbf{y}}_{m-1}} L\left(\mathbf{y}, \hat{\mathbf{y}}_{m-1}\right)$

   2-2. Train regression tree with features $x$ against $r$ and create terminal node reasions $R_{j m}$ for $j=1, \ldots, J_m$
   2-3. Compute $\gamma_{j m}=\underset{\gamma}{\operatorname{argmin}} \sum_{x_i \in R_{\text {im }}} L\left(y_i, F_{m-1}\left(x_i\right)+\gamma\right)$ for $j=1, \ldots, J_m$

   > here we already know the $\gamma_{jm}$ will be the mean of the loss function is the mean squared error,  $w=\operatorname{mean}\left(y_i-F_{m-1}\left(\mathbf{x}_i\right)\right)$.  or another word $\gamma_{jm}$ is regular prediction values of regression trees that are the average of the target values (in our case, residuals) in each terminal node.

   2-4. Update the model:
   $$
   F_m(x)=F_{m-1}(x)+v \sum_{j=1}^{J_m} \gamma_{j m} 1\left(x \in R_{j m}\right)
   $$

   > In the final step, we are updating the prediction of the combined model $\mathrm{Fm}$. $\gamma_{j m} 1\left(x \in R_j m\right)$ means that **we pick the value $\gamma_{j m}$ if a given $x$ falls in a terminal node $R_{\mathrm{j}} m$.** As all the terminal nodes are exclusive, any given single $x$ falls into only a single terminal node and corresponding $\gamma_j m$ is added to the previous prediction $\mathrm F_{m-1}$ and it makes updated prediction $F m$.
   >
   > As mentioned in the previous section, $ν$` is learning rate ranging between 0 and 1 which controls the degree of contribution of the additional tree prediction `*γ*` to the combined prediction `*F𝑚*. A smaller learning rate reduces the effect of the additional tree prediction, **but it basically also reduces the chance of the model overfitting to the training data.**

### Regression Tree 回归树

事实上，分类与回归是两个很接近的问题，分类的目标是根据已知样本的某些特征，判断一个新的样本属于哪种已知的样本类，它的结果是离散值。而回归的结果是连续的值。当然，本质是一样的，都是特征（feature）到结果/标签（label）之间的映射。  

对于回归树，你没法再用分类树那套信息增益、信息增益率、基尼系数来判定树的节点分裂了，你需要采取新的方式评估效果，包括预测误差（常用的有均方误差、对数误差等）。而且节点不再是类别，是数值（预测值），那么怎么确定呢？有的是节点内样本均值，有的是最优化算出来的比如Xgboost。

CART回归树是假设树为二叉树，通过不断将特征进行分裂。比如当前树结点是基于第j个特征值进行分裂的，设该特征值小于s的样本划分为左子树，大于s的样本划分为右子树。 

![img](data:image/png;base64,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)

 而CART回归树实质上就是在该特征维度对样本空间进行划分，而这种空间划分的优化是一种NP难问题，因此，在决策树模型中是使用启发式方法解决。典型CART回归树产生的目标函数为： 

![img](data:image/png;base64,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)

 因此，当我们为了求解最优的切分特征j和最优的切分点s，就转化为求解这么一个目标函数： 

![img](data:image/png;base64,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)

 所以我们只要遍历所有特征的的所有切分点，就能找到最优的切分特征和切分点。最终得到一棵回归树。

一个回归树对应着输入空间 (即特征空间) 的一个划分以及在划分的单元上的输出值。假设已将输 入空间划分为 $\mathrm{M}$ 个单元 $R_1, R_2, \ldots, R_M$ ，并且在每个单元 $R_m$ 上有一个固定的输出值 $c_m$ ， 于是回归树模型可以表示为:
$$
f(x)=\sum_{m=1}^M c_m I\left(x \in R_m\right)
$$
当输入空间的划分确定时，可以用平方误差 $\sum_{x_i \in R_m}\left(y_i-f\left(x_i\right)\right)^2$ 来表示回归树对于训练数据 的预测误差，用平方误差最小的准则求解每个单元上的最优输出值。易知，单元 $R_m$ 上的 $c_m$ 的 最优值 $\hat{c}_m$ 是 $R_m$ 上的所有输入实例 $x_i$ 对应的输出 $y_i$ 的均值，即:
$$
\hat{c}_m=\operatorname{ave}\left(y_i \mid x_i \in R_m\right)
$$
<img src="/assets/Gradient%20boosting.assets/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0VtbWFfTG92ZQ==,size_16,color_FFFFFF,t_70.png" alt="img" style="zoom:50%;" />

#### 问题 1: 怎样对输入空间进行划分? 即如何选择划分点?

CART回归树采用启发式的方法对输入空间进行划分，选择第 $\mathrm{j}$ 个变量 $x^{(j)}$-- feature  和它取的值S，作为切分 变量 (splitting variable) 和切分点 (splitting point)，并定义两个区域:
$$
R_1(j, s)=\left\{x \mid x^{(j)} \leq s\right\} \text { 和 } R_2(j, s)=\left\{x \mid x^{(j)}>s\right\}
$$
然后寻找最优切分变量 $\mathrm{j}$ 和最优切分点 $\mathrm{s}$ 。具体地，求解:
$$
\min _{j, s}\left[\min _{c_1} \sum_{x_i \in R_1(j, s)}\left(y_i-c_1\right)^2+\min _{c_2} \sum_{x_i \in R_2(j, s)}\left(y_i-c_2\right)^2\right]
$$
对固定输入变量可以找到最优切分点 $\mathrm{s}$ 。

#### 问题2: 如何决定树中叶节点的输出值?

用选定的最优切分变量和最优切分点划分区域并决定相应的输出值:
$$
\hat{c}_1=\operatorname{ave}\left(y_i \mid x_i \in R_1(j, s)\right) \text { 和 } \hat{c}_2=\operatorname{ave}\left(y_i \mid x_i \in R_2(j, s)\right)
$$
**遍历所有输入变量，找到最优的切分变量j, 构成一个对 $(j, s)$** 。依此将输入空间划分为两个区 域。接着，对每个区域重复上述划分过程，直到满足停止条件为止。这样就生成一颗回归树。**这样 的回归树通常称为最小二乘回归树 (least squares regression tree)**。
如果已将输入空间划分为 $\mathrm{M}$ 个区域 $R_1, R_2, \ldots, R_M$ ，并且在每个区域 $R_m$ 上有一个固定的输 出值 $\hat{c}_m$ ，于是回归树模型可以表示为:
$$
f(x)=\sum_{m=1}^M \hat{c}_m I\left(x \in R_m\right)
$$

#### 算法流程- (最小二乘回归树生成算法)

输入: 训练数据集 $D$;
输出: 回归树 $f(x)$.
在训练数据集所在的输入空间中, 递归地将每个区域划分为两个子区域并决 定每个子区域上的输出值, 构建二叉决策树:
（1）选择最优切分变量 $j$ 与切分点 $s$, 求解
$$
\min _{j, s}\left[\min _{a_1} \sum_{x_i \in R_1(j, s)}\left(y_i-c_1\right)^2+\min _{c_2} \sum_{x_i \in R_2(j, s)}\left(y_i-c_2\right)^2\right]
$$
**遍历变量 $j$**, 对固定的切分变量 $j$ 扫描切分点 $s$, 选择使上式达到最小 值的对 $(j, s)$.
(2) 用选定的对 $(j, s)$ 划分区域并决定相应的输出值:
$$
\begin{gathered}
R_1(j, s)=\left\{x \mid x^{(j)} \leqslant s\right\}, \quad R_2(j, s)=\left\{x \mid x^{(j)}>s\right\} \\
\hat{c}_m=\frac{1}{N_m} \sum_{x_i \in R_m(j, s)} y_i, \quad x \in R_m, m=1,2
\end{gathered}
$$
（3）继续对两个子区域调用步㗶 (1), (2), 直至满足停止条件.
(4) 将输入空间划分为 $M$ 个区域 $R_1, R_2, \cdots, R_M$, 生成决策树:
$$
f(x)=\sum_{m=1}^M \hat{c}_m I\left(x \in R_m\right)
$$

## Xgboost (extreme Boost)

XGBoost: A Scalable Tree Boosting System ： https://arxiv.org/pdf/1603.02754v1.pdf

xgboost与gbdt比较大的不同就是目标函数的定义。xgboost的目标函数如下图所示

![img](Machine_LearningNote.assets/format,png.png)

其中

- 红色箭头所指向的 $\mathrm{L}$ 即为损失函数（比如平方损失函数: $l\left(y_i, y^i\right)=\left(y_i-y^i\right)^2$ ，或logistic损失函 数: $l\left(y_i, \hat{y}_i\right)=y_i \ln \left(1+e^{-\hat{y}_i}\right)+\left(1-y_i\right) \ln \left(1+e^{\hat{y}_i}\right)$,

- 红色方框所框起来的是正则项 （regularization)（包括L1正则、L2正则)

  $l_1$ 正则化 ( $l_1$ Regularization)
  根据权重的绝对值的总和来惩罚权重。
  $$
  l_1: \Omega(w)=\|w\|_1=\sum_i\left|w_i\right|
  $$
  $l_2$ 正则化 ( $l_2$ Regularization)
  根据权重的平方和来惩罚权重。
  $$
  l_2: \Omega(w)=\|w\|_2^2=\sum_i\left|w_i{ }^2\right|
  $$

- 红色圆圈所圈起来的为常数项

- 对于 $f(x)$ ， xgboos利用泰勒展开三项，做一个近似
  我们可以很清晰地看到，**最终的目标函数只依赖于每个数据点在误差函数上的一阶导数和二阶导数**。

xgboost的核心算法思想继承自GBM，基本就是

1. 不断地添加树，不断地进行特征分裂来生长一棵树，每次添加一个树，其实是学习一个新函数， 去拟合上次预测的残差。

$$
\begin{gathered}
\hat{y}=\phi\left(x_i\right)=\sum_{k=1}^K f_k\left(x_i\right) \\
\text { where } F=\left\{f(x)=w_{q(x)}\right\}\left(q: R^m \rightarrow T, w \in R^T\right)
\end{gathered}
$$

其中一棵回归树。

2. 当我们训练完成得到k棵树，我们要预测一个样本的分数，其实就是根据这个样本的特征，在每 棵树中会落到对应的一个叶子节点，每个叶子节点就对应一个分数
3. 最后只需要将每棵树对应的分数加起来就是该样本的预测值。

显然，我们的目标是要使得树群的预测值 $\hat{y}_i$ 尽量接近真实值 $y_i$ ，而且有尽量大的泛化能力。所以，从数学角度看这是一个泛函最优化问题， 已知训练数据集 $T=\left\{\left(x_1, y_1\right),\left(x_2, y_2\right), \cdots,\left(x_n, y_n\right)\right\}$ ，损失函数 $l\left(y_i, \widehat{y}_i\right)$ ，正则化项 $\Omega\left(f_k\right)$ ，则整体目标函数可记为 :
$$
L(\phi)=\sum_i l\left(y_i, \hat{y}_i\right)+\sum_k \Omega\left(f_k\right)
$$
$\sum_k \Omega\left(f_k\right)$ 表示 $k$ 棵树的复杂度
其中 :
$>\mathcal{L}(\phi)$ 是线性空间上的表达。
$>i$ 是第 $i$ 个样本， $k$ 是第 $k$ 棵树。
$>\hat{y}_i$ 是第 $i$ 个样本 $x_i$ 的预测值
$$
\widehat{y}_i=\sum_{k=1}^K f_k\left(x_i\right)
$$
如你所见，这个目标函数分为两部分：损失函数和正则化项。且损失函数揭示训练误差（即预测分数 和真实分数的差距)，正则化定义复杂度。 杂度越低，泛化能力越强，其表达式为
$$
\Omega(f)=\gamma T+\frac{1}{2} \lambda\|w\|^2
$$
T表示叶子节点的个数，w表示叶子节点的分数。直观上看，目标要求预测误差尽量小，且叶子节点 $T$ 尽量少（Y控制叶子结点的个数），节点数值w尽量不极端（ $\lambda$ 控制叶子节点的分数不会过大），防止 过拟合。

#### **模型学习与训练误差**

具体来说，目标函数第一部分中的 $i$ 表示第 $i$ 个样本， $l\left(\hat{y}_i-y_i\right)$ 表示第 $i$ 个样本的预测误差，我们的 目标当然是误差越小越好。
类似之前GBDT的套路， xgboost也是需要将多棵树的得分侽加得到最终的预测得分（每一次迭代，都 在现有树的基础上，增加一棵树去拟合前面树的预测结果与真实值之间的残差)。

- Start from constant prediction, add a new function each time

$$
\begin{aligned}
\hat{y}_i^{(0)} & =0 \\
\hat{y}_i^{(1)} & =f_1\left(x_i\right)=\hat{y}_i^{(0)}+f_1\left(x_i\right) \\
\hat{y}_i^{(2)} & =f_1\left(x_i\right)+f_2\left(x_i\right)=\hat{y}_i^{(1)}+f_2\left(x_i\right) \\
& \cdots \\
\hat{y}_i^{(t)} & =\sum_{k=1}^t f_k\left(x_i\right)=\hat{y}_i^{(t-1)} +f_t\left(x_i\right) \\
&\hat{y}_i^{(t)} \text {; Model at training round } \mathbf{t} \\
&\hat{y}_i^{(t-1)} \text{; Keep functions added in previous round } \\
&f_t\left(x_i\right) \text{;New function}
\end{aligned}
$$

但，我们如何选择每一轮加入什么 $f$ 呢? 答案是非常直接的，选取一个 $f$ 来使得我们的目标函数尽量 最大地降低。
$$
\begin{aligned}
& O b j^{(t)}=\sum_{i=1}^n l\left(y_i, \hat{y}_i^{(t)}\right)+\sum_{i=1}^t \Omega\left(f_i\right) \\
&=\sum_{i=1}^n l\left(y_i, \hat{y}_i^{(t-1)}+f_t\left(x_i\right)\right)+\Omega\left(f_t\right)+\text { constant } \\
& \text { Goal: find } f_t \text { to minimize this }
\end{aligned}
$$
再强调一下，考虑到第轮的模型预测值 $\hat{y}_i^{(t)}=$ 前 $\mathrm{t}-1$ 轮的模型预测 $\hat{y}_i^{(t-1)}+f_t\left(x_i\right)$ ，因此误差函数记 为: $l\left(y_i, \hat{y}_i^{(t-1)}+f_t\left(x_i\right)\right)$ ，后面一项为正则化项。**除了regularization 其他和GBM是一致的**

#### 第一步：二阶泰勒展开，去除常数项

<img src="/assets/Gradient%20boosting.assets/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3ZfSlVMWV92,size_16,color_FFFFFF,t_70-1685933461156-70.png" alt="img" style="zoom:50%;" />

$f(x)$ 在 $x$ 处进行二阶泰勒展开得到 :
$$
f(x+\Delta x) \simeq f(x)+f^{\prime}(x) \Delta x+\frac{1}{2} f^{\prime \prime}(x) \Delta x^2
$$
**对 $l(y_i, \hat{y}^{t})$在 $\hat{y}_i^{(t-1)}$ 处进行二阶泰勒展开得到** :
$>\widehat{y}_i^{(t-1)}$ 是已知的, $\Delta = f_t\left(x_i\right)$
$$
l\left(y_i, \widehat{y}_i^{(t-1)}+f_t(x_i)\right) \approx l\left(y_i, \widehat{y}_i^{(t-1)}\right)+l^{\prime}\left(y_i, \widehat{y}_i^{(t-1)}\right)\left(\widehat{y}_i^{(t-1)}+f_t(x_i)-\widehat{y}_i^{(t-1)}\right)+\frac{l^{\prime \prime}\left(y_i, \widehat{y}_i^{(t-1)}\right)}{2}\left(\widehat{y}_i^{(t-1)}+f_t(x_i)-\widehat{y}_i^{(t-1)}\right)^2 \\

l\left(y_i, \widehat{y}_i^{(t-1)}+f_t(x_i)\right) \approx l\left(y_i, \widehat{y}_i^{(t-1)}\right)+l^{\prime}\left(y_i, \widehat{y}_i^{(t-1)}\right)f_t(x_i)+\frac{l^{\prime \prime}\left(y_i, \widehat{y}_i^{(t-1)}\right)}{2}(f_t(x_i))^2
$$
记一阶导为 $g_i=l^{\prime}\left(y_i, \widehat{y}_i^{(t-1)}\right)$ ，二阶导为 $h_i=l^{\prime \prime}\left(y_i, \widehat{y}_i^{(t-1)}\right)$.  

**Recall GBM if the lost function is MSE we have  $\nabla_{\hat{\mathbf{y}}} L(\mathbf{y}, \hat{\mathbf{y}})=-2(\mathbf{y}-\hat{\mathbf{y}})$ and $g_i=l^{\prime}\left(y_i, \widehat{y}_i^{(t-1)}\right)= - 2(y_i-\hat{y}^{(t-1)}_i)$ and $h_i=l^{\prime \prime}\left(y_i, \widehat{y}_i^{(t-1)}\right) = 2 $.**

得到 $l\left(y_i, \widehat{y}_i^{(t-1)}+f_t\left(x_i\right)\right)$ 的二阶泰勒展开 :
$$
l\left(y_i, \widehat{y}_i^{(t-1)}+f_t\left(x_i\right)\right) \approx l\left(y_i, \widehat{y}_i^{(t-1)}\right)+g_i f_t\left(x_i\right)+\frac{h_i}{2} f_t^2\left(x_i\right)
$$
带入目标函数可得 :
$$
\mathcal{L}^{(t)}=\sum_{i=1}^n\left[l\left(y_i, \widehat{y}_i^{(t-1)}\right)+g_i f_t\left(x_i\right)+\frac{1}{2} h_i f_t^2\left(x_i\right)\right]+\sum_k \Omega\left(f_k\right)
$$

#### 第二步：正则化项展开，去除常数项

接下来，考虑到我们的第 $\mathrm{t}$ 颗回归树是根据前面的t-1颗回归树的残差得来的，相当于t-1 颗树的值 $\hat{y}_i^{(t-1)}$ 是已知的。换句话说， $l\left(y_i, \hat{y}_i^{(t-1)}\right)$ 对目标函数的优化不影响，可以直接去掉，且常数项 也可以移除，从而得到如下一个比较统一的目标函数。

将正则项进行拆分得
$$
\sum_k \Omega\left(f_k\right)=\sum_{k=1}^t \Omega\left(f_k\right)=\Omega\left(f_t\right)+\sum_{k=1}^{t-1} \Omega\left(f_k\right)=\Omega\left(f_t\right)+\text { 常数 }
$$
因为 $t-1$ 棵数的结构已经确定，所以$\sum_{k=1}^{t-1} \Omega\left(f_k\right)=\text { 常数 }$ 即目标函数可记为

- Objective, with constants removed

$$
\sum_{i=1}^n\left[g_i f_t\left(x_i\right)+\frac{1}{2} h_i f_t^2\left(x_i\right)\right]+\Omega\left(f_t\right)
$$

- where $g_i=\partial_{\hat{y}^{(t-1)}} l\left(y_i, \hat{y}^{(t-1)}\right), \quad h_i=\partial_{\hat{y}^{(t-1)}}^2 l\left(y_i, \hat{y}^{(t-1)}\right)$

这时，目标函数只依赖于每个数据点在误差函数上的一阶导数 $g$ 和二阶导数 $h$

because of $\Omega(f)=\gamma T+\frac{1}{2} \lambda\|w\|^2$
$$
\begin{aligned}
O b j^{(t)} & \simeq \sum_{i=1}^n\left[g_i f_t\left(x_i\right)+\frac{1}{2} h_i f_t^2\left(x_i\right)\right]+\Omega\left(f_t\right) \\
& =\sum_{i=1}^n\left[g_i w_{q\left(x_i\right)}+\frac{1}{2} h_i w_{q\left(x_i\right)}^2\right]+\gamma T+\lambda \frac{1}{2} \sum_{j=1}^T w_j^2 \\
& =\sum_{j=1}^T\left[\left(\sum_{i \in I_j} g_i\right) w_j+\frac{1}{2}\left(\sum_{i \in I_j} h_i+\lambda\right) w_j^2\right]+\gamma T
\end{aligned}
$$
将属于第 $j$ 个叶子结点的所有样本 $x_i$ ，划入到一个叶子结点样本集合中，数学描述如下: $I_j=\left\{i \mid q\left(x_i\right)=j\right\}$  --- the instance set in leaf $j$ as $I_j$

#### 第三步：合并一次项系数、二次项系数

- 用叶子节点集合以及叶子节点得分表示
- 每个样本都落在一个叶子节点上
- q(x)表示样本x在某个叶子节点上，wq(x)是该节点的打分，即该样本的模型预测值

<img src="/assets/Gradient%20boosting.assets/image-20230604204030621.png" alt="image-20230604204030621" style="zoom:50%;" />

<img src="/assets/Gradient%20boosting.assets/image-20230604203954383.png" alt="image-20230604203954383" style="zoom:50%;" />

<img src="/assets/Gradient%20boosting.assets/image-20230604204045839.png" alt="image-20230604204045839" style="zoom:50%;" />

<img src="/assets/Gradient%20boosting.assets/image-20230604204125881.png" alt="image-20230604204125881" style="zoom:50%;" />

#### **XGBoost目标函数解**

已知XGBoost的目标函数 :
$$
\mathcal{L}^{(t)}=\sum_{j=1}^T\left[G_j w_j+\frac{1}{2}\left(H_j+\lambda\right) w_j^2\right]+\gamma T
$$
则每个叶子结点 $j$ 的目标函数是 :
$$
f\left(w_j\right)=G_j w_j+\frac{1}{2}\left(H_j+\lambda\right) w_j{ }^2
$$
其是一个 $w_j$ 的一元二次函数。 $\left(H_j+\lambda\right)>0$ ，则 $f\left(w_j\right)$ 在 $w_j=-\frac{G_j}{H_j+\lambda}$ 处取得最小值，最小值为 $=-\frac{1}{2} \frac{G_j{ }^2}{H_j+\lambda}$

XGBoost目标函数的各个叶子结点的目标式子是相互独立的。即每个叶子结点的式子都达到最值点，整个目标函数也达到最值点。
则每个叶子结点的权重 $w_j^*$ 及此时达到最优的 $O b j$ 目标值 :
$$
w_j^*=-\frac{G_j}{H_j+\lambda} \quad O b j=-\frac{1}{2} \sum_{j=1}^T \frac{G_j{ }^2}{H_j+\lambda}+\gamma T
$$
$>$ 目标值 $O b j$ 最小，则树结构最好，此时即是目标函数的最优解。

#### 打分函数（结构分数structure score -- Obj function value)

<img src="/assets/Gradient%20boosting.assets/20160421110535150.png" alt="img" style="zoom:80%;" />

#### 分裂节点

##### (1) 枚举所有不同树结构的贪心法 Greedy Learning   

==[zphilip48: 最小二乘回归树 算法类似，区别在于选择(增益)条件变化]==

现在的情况是只要知道树的结构，就能得到一个该结构下的最好分数，那如何确定树的结构呢?
一个想当然的方法是: 不断地枚举不同树的结构，然后利用打分函数来寻找出一个最优结构的树，接 看加入到模型中，不断重复这样的操作。而再一想，你会意识到要枚举的状态太多了，基本属于无穷 种，那咋办呢?
**贪心法**:  从树深度 0 开始，每一节点都遍历所有的特征，比如年龄、性别等等，然后对于某 个特征，**先按照该特征里的值进行排序**，然**后线性扫描该特征进而确定最好的分割点**，最后对所有特 征进行分割后，

比如总共五个人，按年龄排好序后，一开始我们总共有如下4种划分方法：

1. 把第一个人和后面四个人划分开
2. 把前两个人和后面三个人划分开
3. 把前三个人和后面两个人划分开
4. 把前面四个人和后面一个人划分开

我们选择所谓的增益Gain最高的那个特征，而Gain如何计算呢? we have following object function: 
$$
O b j=-\frac{1}{2} \sum_{j=1}^T \frac{G_j^2}{H_j+\lambda}+\gamma T
$$
换句话说，目标函数中的 $\mathrm{G} /(\mathrm{H}+\lambda)$ 部分，表示着每一个叶子节点对当前模型损失的贡献程度，融合一 下，得到Gain的计算表达式，如下所示:

![img](/assets/Gradient%20boosting.assets/20160421110908655.png)

![img](/assets/Gradient%20boosting.assets/20160421111024891.png)

因为每次分割都有$\hat{c}_1=\operatorname{ave}\left(y_i \mid x_i \in R_1(j, s)\right) \text { 和 } \hat{c}_2=\operatorname{ave}\left(y_i \mid x_i \in R_2(j, s)\right)$ 得到其中的$\hat{y}_i$那么就可以计算它的$g_i=l^{\prime}\left(y_i, \widehat{y}_i^{(t-1)}\right)= - 2(y_i-\hat{y}^{(t-1)}_i)$ 同样处理$h_i$

**对于所有的特征x**，我们只要做一遍从左到右的扫描就可以枚举出所有分割的梯度和GL和GR。然后用计算Gain的公式计算每个分割方案的分数就可以了。然后后续则依然按照这种划分方法继续第二层、第三层、第四层、第N层的分裂。

第二个值得注意的事情就是引入分割不一定会使得情况变好，所以我们有一个引入新叶子的惩罚项。优化这个目标对应了树的剪枝， 当引入的分割带来的增益小于一个阀值γ 的时候，则忽略这个分割

![img](Machine_LearningNote.assets/20170228144201588.png)

##### (2) Approximate Algorithm 近似算法

对于连续型特征值，当样本数量非常大，该特征取值过多时，遍历所有取值会花费很多时间，且容易过拟合。**因此XGBoost思想是对特征进行分桶，即找到$l$个划分点，将位于相邻分位点之间的样本分在一个桶中**。在遍历该特征的时候，只需要遍历各个分位点，从而计算最优划分。

从算法伪代码中该流程还可以分为两种：

全局的近似: 是在新生成一棵树之前就对各个特征计算分位点并划分样本，之后在每次分裂过程中都采用近似划分
局部近似: 就是在具体的某一次分裂节点的过程中采用近似算法。



![img](/assets/Gradient%20boosting.assets/20170228144525979.png)

- **第一个 for 循环：**对特征 k 根据该特征分布的分位数找到切割点的候选集合 $S_k=\left\{s_{k 1}, s_{k 2}, \ldots, s_{k l}\right\}$。XGBoost 支持 Global 策略和 Local 策略。
- **第二个 for 循环：**针对每个特征的候选集合，将样本映射到由该特征对应的候选点集构成的分桶区间中，即 $s_{k, v} \geq x_{j k}>s_{k, v-1}$  ，对每个桶统计 $G$, $H$值，最后在这些统计量上寻找最佳分裂点。

下图给出近似算法的具体例子，以三分位为例：

![img](/assets/Gradient%20boosting.assets/v2-5d1dd1673419599094bf44dd4b533ba9_720w.webp)

根据样本特征进行排序，然后基于分位数进行划分，并统计三个桶内的 $G$, $H$值，最终求解节点划分的增益。

事实上， XGBoost 不是简单地按照样本个数进行分位，而是以二阶导数值 $ℎ_i$ 作为样本的权重进行划分，如下：

![img](/assets/Gradient%20boosting.assets/v2-5f16246289eaa2a3ae72f971db198457_720w.webp)
$$
\begin{aligned}
O b j^{(t)} & \approx \sum_{i=1}^n\left[g_i f_t\left(x_i\right)+\frac{1}{2} h_i f_t^2\left(x_i\right)\right]+\sum_{i=1}^t \Omega\left(f_i\right) \\
& =\sum_{i=1}^n\left[g_i f_t\left(x_i\right)+\frac{1}{2} h_i f_t^2\left(x_i\right)+\frac{1}{2} \frac{g_i^2}{h_i}\right]+\Omega\left(f_t\right)+C \\
& =\sum_{i=1}^n \frac{1}{2} h_i\left[f_t\left(x_i\right)-\left(-\frac{g_i}{h_i}\right)\right]^2+\Omega\left(f_t\right)+C
\end{aligned}
$$
其中 $\frac{1}{2} \frac{g_i^2}{h_i}$ 与 $C$ 皆为常数。我们可以看到 $h_i$ 就是平方损失函数中样本的权重。

对于样本权值相同的数据集来说，找到候选分位点已经有了解决方案（GK 算法），但是当样本权值不一样时，该如何找到候选分位点呢？作者给出了一个 Weighted Quantile Sketch 算法

#### **针对稀疏数据的算法——缺失值处理**

当样本的第i个特征值缺失时，无法利用该特征进行划分时，XGBoost的想法是将该样本分别划分到左结点和右结点，然后计算其增益，哪个大就划分到哪边。具体见算法3

<img src="/assets/Gradient%20boosting.assets/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0VtbWFfTG92ZQ==,size_16,color_FFFFFF,t_70-1685936352265-73.png" alt="img" style="zoom: 50%;" />

#### Boosted Tree Algorithm 

![img](/assets/Gradient%20boosting.assets/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3ZfSlVMWV92,size_16,color_FFFFFF,t_70-1685885343558-67.png)

####  XGBoost vs. GBDT

1. 传统的GBDT以CART树作为基学习器，XGBoost还支持线性分类器，这个时候XGBoost相当于L1和L2正则化的逻辑斯蒂回归（分类）或者线性回归（回归）；
2. 传统的GBDT在优化的时候只用到一阶导数信息，XGBoost则对代价函数进行了二阶泰勒展开，得到一阶和二阶导数
3. XGBoost在代价函数中加入了正则项，用于控制模型的复杂度。从权衡方差偏差来看，它降低了模型的方差，使学习出来的模型更加简单，放置过拟合，这也是XGBoost优于传统GBDT的一个特性；
4. shrinkage（缩减），相当于学习速率（XGBoost中的eta）。XGBoost在进行完一次迭代时，会将叶子节点的权值乘上该系数，主要是为了削弱每棵树的影响，让后面有更大的学习空间。（GBDT也有学习速率）；
5. Column Subsampling列抽样。XGBoost借鉴了随机森林的做法，支持列抽样，不仅防止过 拟合，还能减少计算；
6. 对缺失值的处理。对于特征的值有缺失的样本，XGBoost还可以自动 学习出它的分裂方向；
7. XGBoost工具支持并行。Boosting不是一种串行的结构吗?怎么并行 的？注意XGBoost的并行不是tree粒度的并行，XGBoost也是一次迭代完才能进行下一次迭代的（第t次迭代的代价函数里包含了前面t-1次迭代的预测值）。**XGBoost的并行是在特征粒度上的**。我们知道，决策树的学习最耗时的一个步骤就是对特征的值进行排序（因为要确定最佳分割点），**XGBoost在训练之前，预先对数据进行了排序，然后保存为block结构**，后面的迭代 中重复地使用这个结构，大大减小计算量。这个block结构也使得并行成为了可能，在进行节点的分裂时，需要计算每个特征的增益，最终选增益最大的那个特征去做分裂，那么各个特征的增益计算就可以开多线程进行



## LightGBM

LightGBM（Light Gradient Boosting Machine）是一个实现GBDT算法的框架，支持高效率的并行训练，并且具有更快的训练速度、更低的内存消耗、更好的准确率、支持分布式可以快速处理海量数据等优点

- **资料：**

[LightGBM: A Highly Efficient Gradient Boosting Decision Tree (neurips.cc)](https://proceedings.neurips.cc/paper_files/paper/2017/file/6449f44a102fde848669bdd9eb6b76fa-Paper.pdf)

A communication-efficient parallel algorithm for decision tree：[1611.01276.pdf (arxiv.org)](https://arxiv.org/pdf/1611.01276.pdf)

[深入理解LightGBM - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/99069186)

[【机器学习】决策树（下）——XGBoost、LightGBM（非常详细） - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/87885678)

- **LightGBM提出的动机**

常用的机器学习算法，例如神经网络等算法，都可以以mini-batch的方式训练，训练数据的大小不会受到内存限制。而GBDT在每一次迭代的时候，都需要遍历整个训练数据多次。如果把整个训练数据装进内存则会限制训练数据的大小；如果不装进内存，反复地读写训练数据又会消耗非常大的时间。尤其面对工业级海量的数据，普通的GBDT算法是不能满足其需求的。

LightGBM提出的主要原因就是为了解决GBDT在海量数据遇到的问题，让GBDT可以更好更快地用于工业实践。

- **XGBoost的缺点及LightGBM的优化**

在LightGBM提出之前，最有名的GBDT工具就是XGBoost了，它是基于预排序方法的决策树算法。这种构建决策树的算法基本思想是：**首先，对所有特征都按照特征的数值进行预排序**。其次，在遍历分割点的时候用O(#data)的代价找到一个特征上的最好分割点。最后，在找到一个特征的最好分割点后，将数据分裂成左右子节点。

这样的预排序算法的优点是能精确地找到分割点。但是缺点也很明显：首先，空间消耗大。这样的算法需要保存数据的特征值，还保存了特征排序的结果（例如，为了后续快速的计算分割点，保存了排序后的索引），这就需要消耗训练数据两倍的内存。其次，时间上也有较大的开销，在遍历每一个分割点的时候，都需要进行分裂增益的计算，消耗的代价大。最后，对cache优化不友好。在预排序后，特征对梯度的访问是一种随机访问，并且不同的特征访问的顺序不一样，无法对cache进行优化。同时，在每一层长树的时候，需要随机访问一个行索引到叶子索引的数组，并且不同特征访问的顺序也不一样，也会造成较大的cache miss。

###  **LightGBM的基本原理**

[【白话机器学习】算法理论+实战之LightGBM算法-腾讯云开发者社区-腾讯云 (tencent.com)](https://cloud.tencent.com/developer/article/1651704)

Histogram algorithm应该翻译为直方图算法，直方图算法的基本思想是：先把连续的浮点特征值离散化成 K 个整数，比如[0, 0.1) ->0, [0.1, 0.3)->1， 同时构造一个宽度为 k 的直方图用于统计信息（含有 k 个 bin）。在遍历数据的时候，根据离散化后的值作为索引在直方图中累积统计量，当遍历一次数据后，直方图累积了需要的统计量，然后根据直方图的离散值（ k 个 bin ），遍历寻找最优的分割点。

- 内存占用更小: XGBoost 需要用 32 位的浮点数去存储特征值，并用 32 位的整形去存储索引， 而 LightGBM 只需要用 8 位去存储直方图，相当于减少了 $1 / 8$;

- 计算代价更小: 计算特征分裂增益时，XGBoost 需要遍历一次数据找到最佳分裂点，而 Light $G B M$ 只需要遍历一次 $\mathrm{k}$ 次，直接将时间复杂度从 $O(\#$ data * \#feature $)$ 降低到 $O(k * \#$ feature $)$ ，而我们知道 $\#$ data $>>k$ 。

  

  <img src="/assets/Gradient%20boosting.assets/1200.png" alt="img" style="zoom:50%;" />



![img](/assets/Gradient%20boosting.assets/1200-1685938257159-78.png)

### **直方图作差加速**

当节点分裂成两个时，右边的子节点的直方图其实等于其父节点的直方图减去左边子节点的直方图：

![img](/assets/Gradient%20boosting.assets/1200-1685938350319-81.png)

![img](/assets/Gradient%20boosting.assets/1200-1685938371956-84.png)

- 离散化的分裂点对最终的精度影响并不大，甚至会好一些。原因在于decision tree本身就是一个弱学习器，分割点是不是精确并不是太重要，采用Histogram算法会起到正则化的效果，有效地防止模型的过拟合（bin数量决定了正则化的程度，bin越少惩罚越严重，欠拟合风险越高）。

- 直方图算法可以起到的作用就是可以减小分割点的数量， 加快计算。

### **带深度限制的 Leaf-wise 算法**

LightGBM进行进一步的优化。首先它抛弃了大多数GBDT工具使用的按层生长 (level-wise) 的决策树生长策略，而使用了带有深度限制的按叶子生长 (leaf-wise) 算法

XGBoost 采用 Level-wise 的增长策略，该策略遍历一次数据可以同时分裂同一层的叶子，容易进行多线程优化，也好控制模型复杂度，不容易过拟合。但实际上Level-wise是一种低效的算法，因为它不加区分的对待同一层的叶子，实际上很多叶子的分裂增益较低，没必要进行搜索和分裂，因此带来了很多没必要的计算开销

![img](/assets/Gradient%20boosting.assets/v2-79a074ec2964a82301209fb66df37113_720w.webp)

LightGBM采用Leaf-wise的增长策略，该策略每次从当前所有叶子中，找到分裂增益最大的一个叶子，然后分裂，如此循环。因此同Level-wise相比，**Leaf-wise的优点是：在分裂次数相同的情况下，Leaf-wise可以降低更多的误差，得到更好的精度；Leaf-wise的缺点是：可能会长出比较深的决策树，产生过拟合**。因此LightGBM会在Leaf-wise之上增加了一个最大深度的限制，在保证高效率的同时防止过拟合。

![img](/assets/Gradient%20boosting.assets/v2-e762a7e4c0366003d7f82e1817da9f89_r.jpg)

### **单边梯度抽样算法(GOSS)**

Gradient-based One-Side Sampling, 

在AdaBoost中，会给每个样本一个权重，然后每一轮之后调大错误样本的权重，让后面的模型更加关注前面错误区分的样本，这时候样本权重是数据重要性的标志，到了GBDT中， 确实没有一个像Adaboost里面这样的样本权重，理论上说是不能应用权重进行采样的， **但是GBDT中每个数据都会有不同的梯度值**， 这个对采样是十分有用的， 即梯度小的样本，训练误差也比较小，说明数据已经被模型学习的很好了，因为GBDT聚焦残差 which is negative gradient in function space! **在训练新模型的过程中，梯度比较小的样本对于降低残差的作用效果不是太大(residual (gradient) is small , no much room to reduce)，所以我们可以关注梯度/residual 高的样本**，which is $\nabla_{\hat{\mathbf{y}}} L(\mathbf{y}, \hat{\mathbf{y}})=-2(\mathbf{y}-\hat{\mathbf{y}})$ if using MSE....

GOSS 算法保留了梯度大的样本，并对梯度小的样本进行随机抽样，为了不改变样本的数据分布，**在计算增益时为梯度小的样本引入一个常数进行平衡**。首先将要进行分裂的特征的所有取值按照绝对值大小降序排序 (xgboost也进行了排序，但是LightGBM不用保存排序后的结果），选取绝对值最大的 a 个数据。然后在剩下的较小梯度数据中随机选择 b个数据。接着将这 b 个数据乘以一个平衡常数$\frac{1-a}{b}$ ，这样算法就会更关注训练不足的样本，而不会过多改变原数据集的分布。最后使用这 (a+b) 个数据来计算信息增益。下图是GOSS的具体算法

![img](/assets/Gradient%20boosting.assets/v2-79c3e6d91863f2512105f86dde65807d_720w.webp)

### **互斥特征捆绑算法(EFB)**

高维度的数据往往是稀疏的，**这种稀疏性启发我们设计一种无损的方法来减少特征的维度**。通常被 捆绑的特征都是互斥的 (即特征不会同时为非零值，像one-hot)，这样两个特征捆绑起来才不会 丟失信息。**如果两个特征并不是完全互斥（部分情况下两个特征都是非零值），可以用一个指标对 特征不互斥程度进行衡量，称之为冲突比率**，当这个值较小时，我们可以选择把不完全互斥的两个 特征捆绑，而不影响最后的精度。

**互斥特征捆绑算法 (Exclusive Feature Bundling, EFB) 指出如 果将一些特征进行融合绑定，则可以降低特征数量。这样在构建直方图时的时间复杂度从 $O(\# d a t a * \#$ feature $)$ 变为 $O(\#$ data $*$ \#bundle $)$ ，这里 \#bundle 指特征融合绑定后 特征包的个数，且 \#bundle 远小于 \# feature。**

针对这种想法，我们会遇到两个问题:

- 怎么判定哪些特征应该绑在一起 (build bundled) ?
- 怎么把特征绑为一个（merge feature）？



#### **解决哪些特征应该绑在一起**

将相互独立的特征进行绑定是一个 NP-Hard 问题，LightGBM的EFB算法将这个问题转化为图着色的问题来求解，将所有的特征视为图的各个顶点，将不是相互独立的特征用一条边连接起来，边的权重就是两个相连接的特征的总冲突值，这样需要绑定的特征就是在图着色问题中要涂上同一种颜色的那些点（特征）。此外，我们注意到通常有很多特征，尽管不是100％相互排斥，但也很少同时取非零值。 如果我们的算法可以允许一小部分的冲突，我们可以得到更少的特征包，进一步提高计算效率。经过简单的计算，随机污染小部分特征值将影响精度最多 $O\left([(1-\gamma) n]^{-2 / 3}\right)$， $\gamma$ 是每个绑定中的最大冲突比率，当其相对较小时，能够完成精度和效率之间的平衡。具体步骤可以总结如下：

1. 构造一个加权无向图，顶点是特征，边有权重，其权重与两个特征间冲突相关；
2. 根据节点的度进行降序排序，度越大，与其它特征的冲突越大；
3. 遍历每个特征，将它分配给现有特征包，或者新建一个特征包，使得总体冲突最小。

 ![img](/assets/Gradient%20boosting.assets/v2-1b2636a948ece17fae81be7f400fedfc_720w.webp)

算法3的时间复杂度是 $O(\#feature^2)$ ，训练之前只处理一次，其时间复杂度在特征不是特别多的情况下是可以接受的，但难以应对百万维度的特征。为了继续提高效率，LightGBM提出了一种更加高效的无图的排序策略：将特征按照非零值个数排序，这和使用图节点的度排序相似，因为更多的非零值通常会导致冲突，新算法在算法3基础上改变了排序策略。

#### **解决怎么把特征绑为一捆**

特征合并算法，其关键在于原始特征能从合并的特征中分离出来。绑定几个特征在同一个bundle里需要保证绑定前的原始特征的值可以在bundle中识别，考虑到histogram-based算法将连续的值保存为离散的bins，我们可以使得不同特征的值分到bundle中的不同bin（箱子）中，这可以通过在特征值中加一个偏置常量来解决。比如，我们在bundle中绑定了两个特征A和B，A特征的原始取值为区间[0,10)，B特征的原始取值为区间[0,20），我们可以在B特征的取值上加一个偏置常量10，将其取值范围变为[10,30），绑定后的特征取值范围为 [0, 30），这样就可以放心的融合特征A和B了。具体的特征合并算法如下所示：

![img](/assets/Gradient%20boosting.assets/v2-cb95a14f542d9cb791df65b63e3f7fdb_720w.webp)

## 