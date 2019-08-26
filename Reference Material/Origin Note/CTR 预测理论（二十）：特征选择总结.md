特征筛选作为一个老生常谈的问题，但自身一直缺乏一个较为完整的梳理，现结合现有资料，总结如下：

## 1. 特征工程与特征选择

首先，追本溯源，为什么特征工程和特征选择值得讨论？在实际的数据分析和建模中，我们通常要面对两种情况：

1. 数据集中已有的特征变量不够多，或者已有的特征变量不足以充分表征数据的特点；

2. 我们拥有大量的特征，需要判断出哪些是相关特征，哪些是不相关特征。特征工程解决的是第一个问题，而特征选择解决的是第二个问题。

对于特征工程来说，它的的难点在于找到好的思路，来产生能够表征数据特点的新特征变量；而特征选择的难点则在于，其本质是一个复杂的组合优化问题（combinatorial optimization）。例如，如果有 30 个特征变量，当我们进行建模的时候，每个特征变量有两种可能的状态：“保留”和“被剔除”。那么，这组特征维度的状态集合中的元素个数就是 $2^{30}$。更一般地，如果我们有 N 个特征变量，则特征变量的状态集合中的元素个数就是 $2^N$。

因此，从算法角度讲，通过穷举的方式进行求解的时间复杂度是指数级的（$O(2^N)$）。当 N 足够大时，特征筛选将会耗费大量的时间和计算资源。在实际应用中，为了减少运算量，目前特征子集的搜索策略大都采用贪心算法（greedy algorithm），其核心思想是在每一步选择中，都采纳当前条件下最好的选择，从而获得组合优化问题的近似最优解。

目前很多流行的机器学习的材料，都未能给出特征工程和特征选择的详细论述。其主要原因是，大部分机器学习算法有标准的推导过程，因而易于讲解。但是在很多实际问题中，寻找和筛选特征变量并没有普适的方法。 然而，特征工程和特征选择对于分析结果的影响，往往比之后的机器学习模型的选择更为重要。斯坦福大学教授，*Coursera* 上著名的机器学习课程主讲老师 *Andrew Ng* 就曾经表示：“基本上，所谓机器学习应用，就是进行特征工程。”

## 2. 特征选择综述

特征选择也可以说是特征工程中的重要问题（另一个重要的问题是特征提取），坊间常说：数据和特征决定了机器学习的上限，而模型和算法只是逼近这个上限而已。由此可见，特征工程尤其是特征选择在机器学习中占有相当重要的地位。
通常而言，**特征选择是指选择获得相应模型和算法最好性能的特征集**，工程上常用的方法有以下：

1. 计算每一个特征与响应变量的相关性：工程上常用的手段有计算皮尔逊系数和互信息系数，**皮尔逊系数（也就是相关性系数）只能衡量线性相关性**而**互信息系数能够很好地度量各种相关性**，但是计算相对复杂一些，好在很多toolkit里边都包含了这个工具（如sklearn的MINE），得到相关性之后就可以排序选择特征了；

2. 构建单个特征的模型，通过模型的准确性为特征排序，借此来选择特征，另外，记得JMLR'03上有一篇论文介绍了一种基于决策树的特征选择方法，本质上是等价的。当选择到了目标特征之后，再用来训练最终的模型；（对应 Wrapper 特征筛选法）

3. 通过 L1 正则项来选择特征：L1 正则方法具有稀疏解的特性，因此天然具备特征选择的特性，但是要注意，L1 没有选到的特征不代表不重要，原因是两个具有高相关性的特征可能只保留了一个，如果要确定哪个特征重要应再通过 L2 正则方法交叉检验；

   > 具体操作为：若一个特征在 L1 中的权值为 1，选择在 L2 中权值差别不大且在 L1 中权值为0的特征构成同类集合，将这一集合中的特征平分L1中的权值，故需要构建一个新的逻辑回归模型。

4. 训练能够对特征打分的预选模型：RandomForest和Logistic Regression等都能对模型的特征打分，通过打分获得相关性后再训练最终模型；

5. 通过特征组合后再来选择特征：如对用户id和用户特征最组合来获得较大的特征集再来选择特征，这种做法在推荐系统和广告系统中比较常见，这也是所谓亿级甚至十亿级特征的主要来源，原因是用户数据比较稀疏，组合特征能够同时兼顾全局模型和个性化模型。

6. 通过深度学习来进行特征选择：目前这种手段正在随着深度学习的流行而成为一种手段，尤其是在计算机视觉领域，原因是深度学习具有自动学习特征的能力，这也是深度学习又叫unsupervised feature learning的原因。从深度学习模型中选择某一神经层的特征后就可以用来进行最终目标模型的训练了。（其本质是数据（特征）表征）

整体上来说，特征选择是一个既有学术价值又有工程价值的问题，目前在研究领域也比较热，值得所有做机器学习的朋友重视。

## 3. 特征选择 sklearn 操作

　当数据预处理完成后，我们需要选择有意义的特征输入机器学习的算法和模型进行训练。通常来说，从两个方面考虑来选择特征：

- 特征是否发散：如果一个特征不发散，例如方差接近于0，也就是说样本在这个特征上基本上没有差异，这个特征对于样本的区分并没有什么用。
- 特征与目标的相关性：这点比较显见，与目标相关性高的特征，应当优选选择。除方差法外，本文介绍的其他方法均从相关性考虑。

　　根据特征选择的形式又可以将特征选择方法分为3种：

- Filter：过滤法，按照发散性或者相关性对各个特征进行评分，设定阈值或者待选择阈值的个数，选择特征。
- Wrapper：包装法，根据目标函数（通常是预测效果评分），每次选择若干特征，或者排除若干特征。
- Embedded：嵌入法，先使用某些机器学习的算法和模型进行训练，得到各个特征的权值系数，根据系数从大到小选择特征。类似于Filter方法，但是是通过训练来确定特征的优劣。

　　我们使用sklearn中的feature_selection库来进行特征选择。

### 3.1 Filter

#### 3.1.1 方差选择法

使用方差选择法，先要计算各个特征的方差，然后根据阈值，选择方差大于阈值的特征。

使用 feature_selection 库的 VarianceThreshold 类来选择特征的代码如下：

```python
1 from sklearn.feature_selection import VarianceThreshold
2 
3 #方差选择法，返回值为特征选择后的数据
4 #参数threshold为方差的阈值
5 VarianceThreshold(threshold=3).fit_transform(iris.data)
```

#### 3.1.2 相关系数法

皮尔森相关系数是一种最简单的，能帮助理解特征和响应变量之间关系的方法，该方法衡量的是变量之间的线性相关性，结果的取值区间为 [-1, 1] ，-1 表示完全的负相关(这个变量下降，那个就会上升)，+1 表示完全的正相关，0 表示没有线性相关。

Pearson Correlation 速度快、易于计算，经常在拿到数据(经过清洗和特征提取之后的)之后第一时间就执行。 Scipy 的 pearsonr 方法能够同时计算相关系数和 p-value，

使用相关系数法，先要计算各个特征对目标值的相关系数以及相关系数的 P 值。用 feature_selection库的SelectKBest 类结合相关系数来选择特征的代码如下：

```python
from sklearn.feature_selection import SelectKBest
from scipy.stats import pearsonr

#选择K个最好的特征，返回选择特征后的数据
#第一个参数为计算评估特征是否好的函数，该函数输入特征矩阵和目标向量，输出二元组（评分，P值）的数组，数组第i项为第i个特征的评分和P值。在此定义为计算相关系数
#参数k为选择的特征个数
SelectKBest(lambda X, Y: array(map(lambda x:pearsonr(x, Y), X.T)).T, k=2).fit_transform(iris.data, iris.target)
```

> Pearson相关系数的一个明显缺陷是，作为特征排序机制，**他只对线性关系敏感**。如果关系是非线性的，即便两个变量具有一一对应的关系，Pearson相关性也可能会接近 0。(相关系数（皮尔逊系数）只能衡量**线性相关性**而**互信息系数能够很好地度量各种相关性**，只是计算比较复杂；)



```
1 from sklearn.feature_selection import SelectKBest
2 from scipy.stats import pearsonr
3 
4 #选择K个最好的特征，返回选择特征后的数据
5 #第一个参数为计算评估特征是否好的函数，该函数输入特征矩阵和目标向量，输出二元组（评分，P值）的数组，数组第i项为第i个特征的评分和P值。在此定义为计算相关系数
6 #参数k为选择的特征个数
7 SelectKBest(lambda X, Y: array(map(lambda x:pearsonr(x, Y), X.T)).T, k=2).fit_transform(iris.data, iris.target)
```

[![复制代码](https://common.cnblogs.com/images/copycode.gif)](javascript:void(0);)

#### 3.1.3 卡方检验

经典的卡方检验是检验**定性自变量**对**定性因变量**的相关性。假设自变量有 N 种取值，因变量有 M 种取值，考虑自变量等于 i 且因变量等于 j 的样本频数的观察值与期望的差距，构建统计量：

![img](https://images2015.cnblogs.com/blog/927391/201605/927391-20160502144243326-2086446424.png)

[这个统计量的含义简而言之就是自变量对因变量的相关性](http://wiki.mbalib.com/wiki/卡方检验)。用feature_selection库的SelectKBest类结合卡方检验来选择特征的代码如下：

```
1 from sklearn.feature_selection import SelectKBest
2 from sklearn.feature_selection import chi2
3 
4 #选择K个最好的特征，返回选择特征后的数据
5 SelectKBest(chi2, k=2).fit_transform(iris.data, iris.target)
```

#### 3.1.4 互信息法

经典的互信息也是评价**定性自变量**对**定性因变量的相关性**的，互信息计算公式如下：
$$
I(X ; Y)=\sum_{x \in X} \sum_{y \in Y} p(x, y) \log \frac{p(x, y)}{p(x) p(y)}
$$
想把互信息直接用于特征选择其实不是太方便：

1. 它不属于度量方式，也没有办法归一化，在不同数据及上的结果无法做比较；
2. 对于连续变量的计算不是很方便（ X 和 Y 都是集合，$x_i$, $y$都是离散的取值），通常变量需要先离散化，而互信息的结果对离散化的方式很敏感。

最大信息系数克服了这两个问题。它首先寻找一种最优的离散化方式，然后把互信息取值转换成一种度量方式，取值区间在 [0, 1]。

下面我们来看下 $y = x^2$ 这个例子，MIC 算出来的互信息值为 1 (最大的取值)。

代码如下：

```python
from minepy import MINE

m = MINE()
x = np.random.uniform(-1, 1, 10000)
m.compute_score(x, x**2)
print(m.mic())
```

#### 3.1.5 **距离相关系数**

距离相关系数是为了克服 Pearson 相关系数的弱点而生的。在 $x$ 和 $x^2$ 这个例子中，即便 Pearson 相关系数是 0，我们也不能断定这两个变量是独立的（有可能是非线性相关）；但如果距离相关系数是 0，那么我们就可以说这两个变量是独立的。

### 3.2 Wrapper

Wrapper这里指不断地使用不同的特征组合来测试学习算法进行特征选择。先选定特定算法， 一般会选用普遍效果较好的算法， 例如Random Forest， SVM， kNN等等。

- **前向搜索**

前向搜索说白了就是每次增量地从剩余未选中的特征选出一个加入特征集中，待达到阈值或者 n 时，从所有的 $F$ 中选出错误率最小的。过程如下：

1. 初始化特征集 $F$ 为空。
2. 扫描 $i$ 从1 到 n
   如果第个 $i$ 特征不在中 $F$，那么特征 $i$ 和 $F$ 放在一起作为 $F_i$ (即 $F_{i}=F \cup\{i\}$)。
   在只使用 $F_i$ 中特征的情况下，利用交叉验证来得到 $F_i$ 的错误率。 
3. 从上步中得到的 n 个 $F_i$ 中选出错误率最小的 $F_i$ ,更新 $F$ 为 $F_i$。
4. 如果 $F$ 中的特征数达到了 $n$ 或者预定的阈值（如果有的话），
   那么输出整个搜索过程中最好的 ；若没达到，则转到 2，继续扫描。

- **后向搜索**

既然有增量加，那么也会有增量减，后者称为后向搜索。先将 $F$ 设置为 ${1,2,\dots,n}$ ，然后每次删除一个特征，并评价，直到达到阈值或者为空，然后选择最佳的 $F$。

这两种算法都可以工作，但是计算复杂度比较大。时间复杂度为
$$
O(n+(n-1)+(n-2)+\ldots+1)=O\left(n^{2}\right)
$$


#### 3.2.1 递归特征消除法

递归消除特征法**使用一个基模型来进行多轮训练**，每轮训练后，消除若干权值系数的特征，再基于新的特征集进行下一轮训练。使用 feature_selection 库的 RFE 类来选择特征的代码如下：

```python
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

#递归特征消除法，返回特征选择后的数据
#参数estimator为基模型
#参数n_features_to_select为选择的特征个数
RFE(estimator=LogisticRegression(), n_features_to_select=2).fit_transform(iris.data, iris.target)
```

### 3.3 Embedded

#### 3.3.1 基于惩罚项的特征选择法

使用带惩罚项的基模型，除了筛选出特征外，同时也进行了降维。使用 feature_selection 库的 SelectFromModel 类结合带 L1 惩罚项的逻辑回归模型，来选择特征的代码如下：

```
1 from sklearn.feature_selection import SelectFromModel
2 from sklearn.linear_model import LogisticRegression
3 
4 #带L1惩罚项的逻辑回归作为基模型的特征选择
5 SelectFromModel(LogisticRegression(penalty="l1", C=0.1)).fit_transform(iris.data, iris.target)
```

通过 L1 正则项来选择特征：L1 正则方法具有稀疏解的特性，因此天然具备特征选择的特性，但是要注意，L1 没有选到的特征不代表不重要，原因是两个具有高相关性的特征可能只保留了一个，如果要确定哪个特征重要应再通过 L2 正则方法交叉检验。

具体操作为：若一个特征在 L1 中的权值为 1，选择在 L2 中权值差别不大且在 L1 中权值为 0 的特征构成同类集合，将这一集合中的特征平分 L1 中的权值，故需要构建一个新的逻辑回归模型：

```python
from sklearn.linear_model import LogisticRegression

class LR(LogisticRegression):
    def __init__(self, threshold=0.01, dual=False, tol=1e-4, C=1.0,
                 fit_intercept=True, intercept_scaling=1, class_weight=None,
                 random_state=None, solver='liblinear', max_iter=100,
                 multi_class='ovr', verbose=0, warm_start=False, n_jobs=1):

        #权值相近的阈值
        self.threshold = threshold
        LogisticRegression.__init__(self, penalty='l1', dual=dual, tol=tol, C=C,
                 fit_intercept=fit_intercept, intercept_scaling=intercept_scaling, class_weight=class_weight,
                 random_state=random_state, solver=solver, max_iter=max_iter,
                 multi_class=multi_class, verbose=verbose, warm_start=warm_start, n_jobs=n_jobs)
        #使用同样的参数创建L2逻辑回归
        self.l2 = LogisticRegression(penalty='l2', dual=dual, tol=tol, C=C, fit_intercept=fit_intercept, intercept_scaling=intercept_scaling, class_weight = class_weight, random_state=random_state, solver=solver, max_iter=max_iter, multi_class=multi_class, verbose=verbose, warm_start=warm_start, n_jobs=n_jobs)

    def fit(self, X, y, sample_weight=None):
        #训练L1逻辑回归
        super(LR, self).fit(X, y, sample_weight=sample_weight)
        self.coef_old_ = self.coef_.copy()
        #训练L2逻辑回归
        self.l2.fit(X, y, sample_weight=sample_weight)

        cntOfRow, cntOfCol = self.coef_.shape
        #权值系数矩阵的行数对应目标值的种类数目
        for i in range(cntOfRow):
            for j in range(cntOfCol):
                coef = self.coef_[i][j]
                #L1逻辑回归的权值系数不为0
                if coef != 0:
                    idx = [j]
                    #对应在L2逻辑回归中的权值系数
                    coef1 = self.l2.coef_[i][j]
                    for k in range(cntOfCol):
                        coef2 = self.l2.coef_[i][k]
                        #在L2逻辑回归中，权值系数之差小于设定的阈值，且在L1中对应的权值为0
                        if abs(coef1-coef2) < self.threshold and j != k and self.coef_[i][k] == 0:
                            idx.append(k)
                    #计算这一类特征的权值系数均值
                    mean = coef / len(idx)
                    self.coef_[i][idx] = mean
        return self
```

使用 feature_selection 库的 SelectFromModel 类结合带L1以及L2惩罚项的逻辑回归模型，来选择特征的代码如下：

```
1 from sklearn.feature_selection import SelectFromModel
2 
3 #带L1和L2惩罚项的逻辑回归作为基模型的特征选择
4 #参数threshold为权值系数之差的阈值
5 SelectFromModel(LR(threshold=0.5, C=0.1)).fit_transform(iris.data, iris.target)
```

#### 3.3.2 基于树模型的特征选择法

这种方法的思路是直接使用你要用的机器学习算法，针对每个单独的特征和响应变量建立预测模型。

假如某个**特征和响应变量之间的关系是非线性的**，可以用基于树的方法（决策树、随机森林）、或者扩展的线性模型等。基于树的方法比较易于使用，因为他们对非线性关系的建模比较好，并且不需要太多的调试。但要注意过拟合问题，因此树的深度最好不要太大，再就是运用交叉验证。通过这种训练对特征进行打分获得相关性后再训练最终模型。

树模型中 GBDT 也可用来作为基模型进行特征选择，使用 feature_selection 库的 SelectFromModel 类结合 GBDT 模型，来选择特征的代码如下：

```
1 from sklearn.feature_selection import SelectFromModel
2 from sklearn.ensemble import GradientBoostingClassifier
3 
4 #GBDT作为基模型的特征选择
5 SelectFromModel(GradientBoostingClassifier()).fit_transform(iris.data, iris.target)
```

## 3.4 回顾

| 类                | 所属方式 | 说明                                                   |
| ----------------- | -------- | ------------------------------------------------------ |
| VarianceThreshold | Filter   | 方差选择法                                             |
| SelectKBest       | Filter   | 可选关联系数、卡方校验、最大信息系数作为得分计算的方法 |
| RFE               | Wrapper  | 递归地训练基模型，将权值系数较小的特征从特征集合中消除 |
| SelectFromModel   | Embedded | 训练基模型，选择权值系数较高的特征                     |





## 参考文献

[1] [使用sklearn做单机特征工程](https://www.cnblogs.com/jasonfreak/p/5448385.html)

[2] [特征选择](https://zhuanlan.zhihu.com/p/32749489)

[3] [机器学习中，有哪些特征选择的工程方法](https://www.zhihu.com/question/28641663/answer/41653367)

