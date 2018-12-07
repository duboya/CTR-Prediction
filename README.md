
# CTR Prediction 论文、个人学习笔记分享

动态更新学习中实现或者阅读过的计算广告相关论文、学习资料和业界分享，作为自己学习的总结，也希望能为计算广告相关行业的同学带来便利。
同时欢迎对CTR Prediction感兴趣的同学与我(杜博亚)讨论相关问题，我的联系方式如下：

* Email: duboyabz@163.com
* 知乎私信: [杜博亚的知乎](https://www.zhihu.com/people/freedom_forever/activities)

## 目录

### Optimization Method
Online Optimization，Parallel SGD，FTRL等优化方法，实用并且能够给出直观解释的文章
* [Google Vizier A Service for Black-Box Optimization.pdf](https://github.com/wzhe06/Ad-papers/blob/master/Optimization%20Method/Google%20Vizier%20A%20Service%20for%20Black-Box%20Optimization.pdf) 
Google的深度学习自动调参框架Vizier
* [在线最优化求解(Online Optimization)-冯扬.pdf](https://github.com/wzhe06/Ad-papers/blob/master/Optimization%20Method/%E5%9C%A8%E7%BA%BF%E6%9C%80%E4%BC%98%E5%8C%96%E6%B1%82%E8%A7%A3%28Online%20Optimization%29-%E5%86%AF%E6%89%AC.pdf) 
非常推荐冯扬的这个教程，把在线优化问题讲的非常透
* [Hogwild A Lock-Free Approach to Parallelizing Stochastic Gradient Descent.pdf](https://github.com/wzhe06/Ad-papers/blob/master/Optimization%20Method/Hogwild%20A%20Lock-Free%20Approach%20to%20Parallelizing%20Stochastic%20Gradient%20Descent.pdf) 
* [Parallelized Stochastic Gradient Descent.pdf](https://github.com/wzhe06/Ad-papers/blob/master/Optimization%20Method/Parallelized%20Stochastic%20Gradient%20Descent.pdf) 
* [A Survey on Algorithms of the Regularized Convex Optimization Problem.pptx](https://github.com/wzhe06/Ad-papers/blob/master/Optimization%20Method/A%20Survey%20on%20Algorithms%20of%20the%20Regularized%20Convex%20Optimization%20Problem.pptx) 
* [Follow-the-Regularized-Leader and Mirror Descent- Equivalence Theorems and L1 Regularization.pdf](https://github.com/wzhe06/Ad-papers/blob/master/Optimization%20Method/Follow-the-Regularized-Leader%20and%20Mirror%20Descent-%20Equivalence%20Theorems%20and%20L1%20Regularization.pdf) 
* [A Review of Bayesian Optimization.pdf](https://github.com/wzhe06/Ad-papers/blob/master/Optimization%20Method/A%20Review%20of%20Bayesian%20Optimization.pdf) 
* [Taking the Human Out of the Loop- A Review of Bayesian Optimization.pdf](https://github.com/wzhe06/Ad-papers/blob/master/Optimization%20Method/Taking%20the%20Human%20Out%20of%20the%20Loop-%20A%20Review%20of%20Bayesian%20Optimization.pdf) 
* [非线性规划.doc](https://github.com/wzhe06/Ad-papers/blob/master/Optimization%20Method/%E9%9D%9E%E7%BA%BF%E6%80%A7%E8%A7%84%E5%88%92.doc) 


### CTR Prediction
作为计算广告的核心，CTR预估永远是研究的热点，下面每一篇都是非常流行的文章，推荐逐一精读
* [Deep Crossing- Web-Scale Modeling without Manually Crafted Combinatorial Features.pdf](https://github.com/wzhe06/Ad-papers/blob/master/CTR%20Prediction/Deep%20Crossing-%20Web-Scale%20Modeling%20without%20Manually%20Crafted%20Combinatorial%20Features.pdf) <br />
* [Learning Piece-wise Linear Models from Large Scale Data for Ad Click Prediction.pdf](https://github.com/wzhe06/Ad-papers/blob/master/CTR%20Prediction/Learning%20Piece-wise%20Linear%20Models%20from%20Large%20Scale%20Data%20for%20Ad%20Click%20Prediction.pdf) <br />
阿里提出的Large Scale Piece-wise Linear Model (LS-PLM) CTR预估模型
* [[GBDT+LR]Practical Lessons from Predicting Clicks on Ads at Facebook.pdf](https://github.com/wzhe06/Ad-papers/blob/master/CTR%20Prediction/%5BGBDT%2BLR%5DPractical%20Lessons%20from%20Predicting%20Clicks%20on%20Ads%20at%20Facebook.pdf) <br />
* [[FNN]Deep Learning over Multi-field Categorical Data.pdf](https://github.com/wzhe06/Ad-papers/blob/master/CTR%20Prediction/%5BFNN%5DDeep%20Learning%20over%20Multi-field%20Categorical%20Data.pdf) <br />
* [Entire Space Multi-Task Model_ An Effective Approach for Estimating Post-Click Conversion Rate.pdf](https://github.com/wzhe06/Ad-papers/blob/master/CTR%20Prediction/Entire%20Space%20Multi-Task%20Model_%20An%20Effective%20Approach%20for%20Estimating%20Post-Click%20Conversion%20Rate.pdf) <br />
* [Deep Interest Network for Click-Through Rate Prediction.pdf](https://github.com/wzhe06/Ad-papers/blob/master/CTR%20Prediction/Deep%20Interest%20Network%20for%20Click-Through%20Rate%20Prediction.pdf) <br />
* [Bid-aware Gradient Descent for Unbiased Learning with Censored Data in Display Advertising.pdf](https://github.com/wzhe06/Ad-papers/blob/master/CTR%20Prediction/Bid-aware%20Gradient%20Descent%20for%20Unbiased%20Learning%20with%20Censored%20Data%20in%20Display%20Advertising.pdf) <br />
RTB 中训练 CTR 模型数据集是赢得出价的广告，预测时的样本却是所有候选的广告，也就是训练集和测试集的分布不一致，这篇文章就是要消除这样的 bias
* [[Multi-Task]An Overview of Multi-Task Learning in Deep Neural Networks.pdf](https://github.com/wzhe06/Ad-papers/blob/master/CTR%20Prediction/%5BMulti-Task%5DAn%20Overview%20of%20Multi-Task%20Learning%20in%20Deep%20Neural%20Networks.pdf) <br />
* [Ad Click Prediction a View from the Trenches.pdf](https://github.com/wzhe06/Ad-papers/blob/master/CTR%20Prediction/Ad%20Click%20Prediction%20a%20View%20from%20the%20Trenches.pdf) <br />
Google大名鼎鼎的用FTRL解决CTR在线预估的工程文章，非常经典。
* [[PNN]Product-based Neural Networks for User Response Prediction.pdf](https://github.com/wzhe06/Ad-papers/blob/master/CTR%20Prediction/%5BPNN%5DProduct-based%20Neural%20Networks%20for%20User%20Response%20Prediction.pdf) <br />
* [Image Matters- Visually modeling user behaviors using Advanced Model Server.pdf](https://github.com/wzhe06/Ad-papers/blob/master/CTR%20Prediction/Image%20Matters-%20Visually%20modeling%20user%20behaviors%20using%20Advanced%20Model%20Server.pdf) <br />
阿里提出引入商品图像特征的（Deep Image CTR Model）CTR预估模型，并介绍其分布式机器学习框架 Advanced Model Server (AMS)
* [[Wide & Deep]Wide & Deep Learning for Recommender Systems.pdf](https://github.com/wzhe06/Ad-papers/blob/master/CTR%20Prediction/%5BWide%20%26%20Deep%5DWide%20%26%20Deep%20Learning%20for%20Recommender%20Systems.pdf) <br />
* [[DeepFM]- A Factorization-Machine based Neural Network for CTR Prediction.pdf](https://github.com/wzhe06/Ad-papers/blob/master/CTR%20Prediction/%5BDeepFM%5D-%20A%20Factorization-Machine%20based%20Neural%20Network%20for%20CTR%20Prediction.pdf) <br />
* [Logistic Regression in Rare Events Data.pdf](https://github.com/wzhe06/Ad-papers/blob/master/CTR%20Prediction/Logistic%20Regression%20in%20Rare%20Events%20Data.pdf) <br />
样本稀少情况下的LR模型训练，讲的比较细
* [Deep & Cross Network for Ad Click Predictions.pdf](https://github.com/wzhe06/Ad-papers/blob/master/CTR%20Prediction/Deep%20%26%20Cross%20Network%20for%20Ad%20Click%20Predictions.pdf) <br />
Google 在17年发表的 Deep&Cross 网络，类似于 Wide&Deep, 比起 PNN 只做了特征二阶交叉，Deep&Cross 理论上能够做任意高阶的特征交叉
* [Learning Deep Structured Semantic Models for Web Search using Clickthrough Data.pdf](https://github.com/wzhe06/Ad-papers/blob/master/CTR%20Prediction/Learning%20Deep%20Structured%20Semantic%20Models%20for%20Web%20Search%20using%20Clickthrough%20Data.pdf) <br />
* [Adaptive Targeting for Online Advertisement.pdf](https://github.com/wzhe06/Ad-papers/blob/master/CTR%20Prediction/Adaptive%20Targeting%20for%20Online%20Advertisement.pdf) <br />
一篇比较简单但是全面的CTR预估的文章，有一定实用性


## 参考文献

[1] https://github.com/wzhe06/Ad-papers/blob/master/README.md
[2] https://github.com/duboya/ML_CIA