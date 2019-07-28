
# CTR Prediction 论文、个人学习笔记分享

动态更新学习中实现或者阅读过的计算广告相关论文、学习资料和业界分享，作为自己学习的总结，也希望能为计算广告相关行业的同学带来便利。
同时欢迎对CTR Prediction感兴趣的同学与我(杜博亚)讨论相关问题，我的联系方式如下：

* Email: duboyabz@163.com
* 知乎私信: [杜博亚的知乎](https://www.zhihu.com/people/freedom_forever/activities)
* 个人博客: https://blog.csdn.net/dby_freedom?t=1

其中，个人博客收录了本人关于CTR的理论、实践总结，欢迎访问、关注~

## 目录

### Optimization Method
Online Optimization，Parallel SGD，FTRL等优化方法，实用并且能够给出直观解释的文章
* [Google Vizier A Service for Black-Box Optimization.pdf](https://static.googleusercontent.com/media/research.google.com/zh-CN//pubs/archive/46180.pdf) 
Google的深度学习自动调参框架Vizier
* [在线最优化求解(Online Optimization)-冯扬.pdf](https://github.com/duboya/CTR-Prediction/blob/master/Reference%20Material/%E5%9C%A8%E7%BA%BF%E6%9C%80%E4%BC%98%E5%8C%96%E6%B1%82%E8%A7%A3.pdf) 
非常推荐冯扬的这个教程，把在线优化问题讲的非常透
* [Hogwild A Lock-Free Approach to Parallelizing Stochastic Gradient Descent.pdf](https://github.com/wzhe06/Ad-papers/blob/master/Optimization%20Method/Hogwild%20A%20Lock-Free%20Approach%20to%20Parallelizing%20Stochastic%20Gradient%20Descent.pdf) 
* [Parallelized Stochastic Gradient Descent.pdf](https://github.com/wzhe06/Ad-papers/blob/master/Optimization%20Method/Parallelized%20Stochastic%20Gradient%20Descent.pdf) 
* [A Survey on Algorithms of the Regularized Convex Optimization Problem.pptx](https://github.com/wzhe06/Ad-papers/blob/master/Optimization%20Method/A%20Survey%20on%20Algorithms%20of%20the%20Regularized%20Convex%20Optimization%20Problem.pptx) 
* [Follow-the-Regularized-Leader and Mirror Descent- Equivalence Theorems and L1 Regularization.pdf](https://github.com/wzhe06/Ad-papers/blob/master/Optimization%20Method/Follow-the-Regularized-Leader%20and%20Mirror%20Descent-%20Equivalence%20Theorems%20and%20L1%20Regularization.pdf) 
* [A Review of Bayesian Optimization.pdf](https://github.com/wzhe06/Ad-papers/blob/master/Optimization%20Method/A%20Review%20of%20Bayesian%20Optimization.pdf) 
* [Taking the Human Out of the Loop- A Review of Bayesian Optimization.pdf](https://github.com/wzhe06/Ad-papers/blob/master/Optimization%20Method/Taking%20the%20Human%20Out%20of%20the%20Loop-%20A%20Review%20of%20Bayesian%20Optimization.pdf) 
* [非线性规划.doc](https://github.com/wzhe06/Ad-papers/blob/master/Optimization%20Method/%E9%9D%9E%E7%BA%BF%E6%80%A7%E8%A7%84%E5%88%92.doc) 


### Classic CTR Prediction
* [[LR] Predicting Clicks - Estimating the Click-Through Rate for New Ads (Microsoft 2007)](https://github.com/wzhe06/Ad-papers/blob/master/Classic%20CTR%20Prediction/%5BLR%5D%20Predicting%20Clicks%20-%20Estimating%20the%20Click-Through%20Rate%20for%20New%20Ads%20%28Microsoft%202007%29.pdf) <br />
* [[FFM] Field-aware Factorization Machines for CTR Prediction (Criteo 2016)](https://github.com/wzhe06/Ad-papers/blob/master/Classic%20CTR%20Prediction/%5BFFM%5D%20Field-aware%20Factorization%20Machines%20for%20CTR%20Prediction%20%28Criteo%202016%29.pdf) <br />
* [[GBDT+LR] Practical Lessons from Predicting Clicks on Ads at Facebook (Facebook 2014)](https://github.com/wzhe06/Ad-papers/blob/master/Classic%20CTR%20Prediction/%5BGBDT%2BLR%5D%20Practical%20Lessons%20from%20Predicting%20Clicks%20on%20Ads%20at%20Facebook%20%28Facebook%202014%29.pdf) <br />
* [[PS-PLM] Learning Piece-wise Linear Models from Large Scale Data for Ad Click Prediction (Alibaba 2017)](https://github.com/wzhe06/Ad-papers/blob/master/Classic%20CTR%20Prediction/%5BPS-PLM%5D%20Learning%20Piece-wise%20Linear%20Models%20from%20Large%20Scale%20Data%20for%20Ad%20Click%20Prediction%20%28Alibaba%202017%29.pdf) <br />
* [[FTRL] Ad Click Prediction a View from the Trenches (Google 2013)](https://github.com/wzhe06/Ad-papers/blob/master/Classic%20CTR%20Prediction/%5BFTRL%5D%20Ad%20Click%20Prediction%20a%20View%20from%20the%20Trenches%20%28Google%202013%29.pdf) <br />
* [[FM] Fast Context-aware Recommendations with Factorization Machines (UKON 2011)](https://github.com/wzhe06/Ad-papers/blob/master/Classic%20CTR%20Prediction/%5BFM%5D%20Fast%20Context-aware%20Recommendations%20with%20Factorization%20Machines%20%28UKON%202011%29.pdf) <br />


### Deep Learning CTR Prediction
* [[DCN] Deep & Cross Network for Ad Click Predictions (Stanford 2017)](https://github.com/wzhe06/Ad-papers/blob/master/Deep%20Learning%20CTR%20Prediction/%5BDCN%5D%20Deep%20%26%20Cross%20Network%20for%20Ad%20Click%20Predictions%20%28Stanford%202017%29.pdf) <br />
* [[Deep Crossing] Deep Crossing - Web-Scale Modeling without Manually Crafted Combinatorial Features (Microsoft 2016)](https://github.com/wzhe06/Ad-papers/blob/master/Deep%20Learning%20CTR%20Prediction/%5BDeep%20Crossing%5D%20Deep%20Crossing%20-%20Web-Scale%20Modeling%20without%20Manually%20Crafted%20Combinatorial%20Features%20%28Microsoft%202016%29.pdf) <br />
* [[PNN] Product-based Neural Networks for User Response Prediction (SJTU 2016)](https://github.com/wzhe06/Ad-papers/blob/master/Deep%20Learning%20CTR%20Prediction/%5BPNN%5D%20Product-based%20Neural%20Networks%20for%20User%20Response%20Prediction%20%28SJTU%202016%29.pdf) <br />
* [[DIN] Deep Interest Network for Click-Through Rate Prediction (Alibaba 2018)](https://github.com/wzhe06/Ad-papers/blob/master/Deep%20Learning%20CTR%20Prediction/%5BDIN%5D%20Deep%20Interest%20Network%20for%20Click-Through%20Rate%20Prediction%20%28Alibaba%202018%29.pdf) <br />
* [[ESMM] Entire Space Multi-Task Model - An Effective Approach for Estimating Post-Click Conversion Rate (Alibaba 2018)](https://github.com/wzhe06/Ad-papers/blob/master/Deep%20Learning%20CTR%20Prediction/%5BESMM%5D%20Entire%20Space%20Multi-Task%20Model%20-%20An%20Effective%20Approach%20for%20Estimating%20Post-Click%20Conversion%20Rate%20%28Alibaba%202018%29.pdf) <br />
* [[Wide & Deep] Wide & Deep Learning for Recommender Systems (Google 2016)](https://github.com/wzhe06/Ad-papers/blob/master/Deep%20Learning%20CTR%20Prediction/%5BWide%20%26%20Deep%5D%20Wide%20%26%20Deep%20Learning%20for%20Recommender%20Systems%20%28Google%202016%29.pdf) <br />
* [[xDeepFM] xDeepFM - Combining Explicit and Implicit Feature Interactions for Recommender Systems (USTC 2018)](https://github.com/wzhe06/Ad-papers/blob/master/Deep%20Learning%20CTR%20Prediction/%5BxDeepFM%5D%20xDeepFM%20-%20Combining%20Explicit%20and%20Implicit%20Feature%20Interactions%20for%20Recommender%20Systems%20%28USTC%202018%29.pdf) <br />
* [[Image CTR] Image Matters - Visually modeling user behaviors using Advanced Model Server (Alibaba 2018)](https://github.com/wzhe06/Ad-papers/blob/master/Deep%20Learning%20CTR%20Prediction/%5BImage%20CTR%5D%20Image%20Matters%20-%20Visually%20modeling%20user%20behaviors%20using%20Advanced%20Model%20Server%20%28Alibaba%202018%29.pdf) <br />
* [[AFM] Attentional Factorization Machines - Learning the Weight of Feature Interactions via Attention Networks (ZJU 2017)](https://github.com/wzhe06/Ad-papers/blob/master/Deep%20Learning%20CTR%20Prediction/%5BAFM%5D%20Attentional%20Factorization%20Machines%20-%20Learning%20the%20Weight%20of%20Feature%20Interactions%20via%20Attention%20Networks%20%28ZJU%202017%29.pdf) <br />
* [[DIEN] Deep Interest Evolution Network for Click-Through Rate Prediction (Alibaba 2019)](https://github.com/wzhe06/Ad-papers/blob/master/Deep%20Learning%20CTR%20Prediction/%5BDIEN%5D%20Deep%20Interest%20Evolution%20Network%20for%20Click-Through%20Rate%20Prediction%20%28Alibaba%202019%29.pdf) <br />
* [[DSSM] Learning Deep Structured Semantic Models for Web Search using Clickthrough Data (UIUC 2013)](https://github.com/wzhe06/Ad-papers/blob/master/Deep%20Learning%20CTR%20Prediction/%5BDSSM%5D%20Learning%20Deep%20Structured%20Semantic%20Models%20for%20Web%20Search%20using%20Clickthrough%20Data%20%28UIUC%202013%29.pdf) <br />
* [[FNN] Deep Learning over Multi-field Categorical Data (UCL 2016)](https://github.com/wzhe06/Ad-papers/blob/master/Deep%20Learning%20CTR%20Prediction/%5BFNN%5D%20Deep%20Learning%20over%20Multi-field%20Categorical%20Data%20%28UCL%202016%29.pdf) <br />
* [[DeepFM] A Factorization-Machine based Neural Network for CTR Prediction (HIT-Huawei 2017)](https://github.com/wzhe06/Ad-papers/blob/master/Deep%20Learning%20CTR%20Prediction/%5BDeepFM%5D%20A%20Factorization-Machine%20based%20Neural%20Network%20for%20CTR%20Prediction%20%28HIT-Huawei%202017%29.pdf) <br />
* [[NFM] Neural Factorization Machines for Sparse Predictive Analytics (NUS 2017)](https://github.com/wzhe06/Ad-papers/blob/master/Deep%20Learning%20CTR%20Prediction/%5BNFM%5D%20Neural%20Factorization%20Machines%20for%20Sparse%20Predictive%20Analytics%20%28NUS%202017%29.pdf) <br />

### Embedding
* [[Negative Sampling] Word2vec Explained Negative-Sampling Word-Embedding Method (2014)](https://github.com/wzhe06/Ad-papers/blob/master/Embedding/%5BNegative%20Sampling%5D%20Word2vec%20Explained%20Negative-Sampling%20Word-Embedding%20Method%20%282014%29.pdf) <br />
* [[SDNE] Structural Deep Network Embedding (THU 2016)](https://github.com/wzhe06/Ad-papers/blob/master/Embedding/%5BSDNE%5D%20Structural%20Deep%20Network%20Embedding%20%28THU%202016%29.pdf) <br />
* [[Item2Vec] Item2Vec-Neural Item Embedding for Collaborative Filtering (Microsoft 2016)](https://github.com/wzhe06/Ad-papers/blob/master/Embedding/%5BItem2Vec%5D%20Item2Vec-Neural%20Item%20Embedding%20for%20Collaborative%20Filtering%20%28Microsoft%202016%29.pdf) <br />
* [[Word2Vec] Distributed Representations of Words and Phrases and their Compositionality (Google 2013)](https://github.com/wzhe06/Ad-papers/blob/master/Embedding/%5BWord2Vec%5D%20Distributed%20Representations%20of%20Words%20and%20Phrases%20and%20their%20Compositionality%20%28Google%202013%29.pdf) <br />
* [[Word2Vec] Word2vec Parameter Learning Explained (UMich 2016)](https://github.com/wzhe06/Ad-papers/blob/master/Embedding/%5BWord2Vec%5D%20Word2vec%20Parameter%20Learning%20Explained%20%28UMich%202016%29.pdf) <br />
* [[Node2vec] Node2vec - Scalable Feature Learning for Networks (Stanford 2016)](https://github.com/wzhe06/Ad-papers/blob/master/Embedding/%5BNode2vec%5D%20Node2vec%20-%20Scalable%20Feature%20Learning%20for%20Networks%20%28Stanford%202016%29.pdf) <br />
* [[Graph Embedding] DeepWalk- Online Learning of Social Representations (SBU 2014)](https://github.com/wzhe06/Ad-papers/blob/master/Embedding/%5BGraph%20Embedding%5D%20DeepWalk-%20Online%20Learning%20of%20Social%20Representations%20%28SBU%202014%29.pdf) <br />
* [[Airbnb Embedding] Real-time Personalization using Embeddings for Search Ranking at Airbnb (Airbnb 2018)](https://github.com/wzhe06/Ad-papers/blob/master/Embedding/%5BAirbnb%20Embedding%5D%20Real-time%20Personalization%20using%20Embeddings%20for%20Search%20Ranking%20at%20Airbnb%20%28Airbnb%202018%29.pdf) <br />
* [[Alibaba Embedding] Billion-scale Commodity Embedding for E-commerce Recommendation in Alibaba (Alibaba 2018)](https://github.com/wzhe06/Ad-papers/blob/master/Embedding/%5BAlibaba%20Embedding%5D%20Billion-scale%20Commodity%20Embedding%20for%20E-commerce%20Recommendation%20in%20Alibaba%20%28Alibaba%202018%29.pdf) <br />
* [[Word2Vec] Efficient Estimation of Word Representations in Vector Space (Google 2013)](https://github.com/wzhe06/Ad-papers/blob/master/Embedding/%5BWord2Vec%5D%20Efficient%20Estimation%20of%20Word%20Representations%20in%20Vector%20Space%20%28Google%202013%29.pdf) <br />
* [[LINE] LINE - Large-scale Information Network Embedding (MSRA 2015)](https://github.com/wzhe06/Ad-papers/blob/master/Embedding/%5BLINE%5D%20LINE%20-%20Large-scale%20Information%20Network%20Embedding%20%28MSRA%202015%29.pdf) <br />


### Tree Model

树模型和基于树模型的boosting模型，树模型的效果在大部分问题上非常好，在CTR，CVR预估及特征工程方面的应用非常广

* [Introduction to Boosted Trees](https://github.com/wzhe06/Ad-papers/blob/master/Tree%20Model/Introduction%20to%20Boosted%20Trees.pdf) <br />
* [Classification and Regression Trees](https://github.com/wzhe06/Ad-papers/blob/master/Tree%20Model/Classification%20and%20Regression%20Trees.pdf) <br />
* [Greedy Function Approximation A Gradient Boosting Machine](https://github.com/wzhe06/Ad-papers/blob/master/Tree%20Model/Greedy%20Function%20Approximation%20A%20Gradient%20Boosting%20Machine.pdf) <br />
* [Classification and Regression Trees](https://github.com/wzhe06/Ad-papers/blob/master/Tree%20Model/Classification%20and%20Regression%20Trees.ppt) <br />

### Computational Advertising Architect
广告系统的架构问题
* [[TensorFlow Whitepaper]TensorFlow- Large-Scale Machine Learning on Heterogeneous Distributed Systems](https://github.com/wzhe06/Ad-papers/blob/master/Computational%20Advertising%20Architect/%5BTensorFlow%20Whitepaper%5DTensorFlow-%20Large-Scale%20Machine%20Learning%20on%20Heterogeneous%20Distributed%20Systems.pdf) <br />
* [大数据下的广告排序技术及实践](https://github.com/wzhe06/Ad-papers/blob/master/Computational%20Advertising%20Architect/%E5%A4%A7%E6%95%B0%E6%8D%AE%E4%B8%8B%E7%9A%84%E5%B9%BF%E5%91%8A%E6%8E%92%E5%BA%8F%E6%8A%80%E6%9C%AF%E5%8F%8A%E5%AE%9E%E8%B7%B5.pdf) <br />
* [美团机器学习 吃喝玩乐中的算法问题](https://github.com/wzhe06/Ad-papers/blob/master/Computational%20Advertising%20Architect/%E7%BE%8E%E5%9B%A2%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%20%E5%90%83%E5%96%9D%E7%8E%A9%E4%B9%90%E4%B8%AD%E7%9A%84%E7%AE%97%E6%B3%95%E9%97%AE%E9%A2%98.pdf) <br />
* [[Parameter Server]Scaling Distributed Machine Learning with the Parameter Server](https://github.com/wzhe06/Ad-papers/blob/master/Computational%20Advertising%20Architect/%5BParameter%20Server%5DScaling%20Distributed%20Machine%20Learning%20with%20the%20Parameter%20Server.pdf) <br />
* [Display Advertising with Real-Time Bidding (RTB) and Behavioural Targeting](https://github.com/wzhe06/Ad-papers/blob/master/Computational%20Advertising%20Architect/Display%20Advertising%20with%20Real-Time%20Bidding%20%28RTB%29%20and%20Behavioural%20Targeting.pdf) <br />
* [A Comparison of Distributed Machine Learning Platforms](https://github.com/wzhe06/Ad-papers/blob/master/Computational%20Advertising%20Architect/A%20Comparison%20of%20Distributed%20Machine%20Learning%20Platforms.pdf) <br />
* [Efficient Query Evaluation using a Two-Level Retrieval Process](https://github.com/wzhe06/Ad-papers/blob/master/Computational%20Advertising%20Architect/Efficient%20Query%20Evaluation%20using%20a%20Two-Level%20Retrieval%20Process.pdf) <br />
* [[TensorFlow Whitepaper]TensorFlow- A System for Large-Scale Machine Learning](https://github.com/wzhe06/Ad-papers/blob/master/Computational%20Advertising%20Architect/%5BTensorFlow%20Whitepaper%5DTensorFlow-%20A%20System%20for%20Large-Scale%20Machine%20Learning.pdf) <br />
* [[Parameter Server]Parameter Server for Distributed Machine Learning](https://github.com/wzhe06/Ad-papers/blob/master/Computational%20Advertising%20Architect/%5BParameter%20Server%5DParameter%20Server%20for%20Distributed%20Machine%20Learning.pdf) <br />
* [Overlapping Experiment Infrastructure More, Better, Faster Experimentation](https://github.com/wzhe06/Ad-papers/blob/master/Computational%20Advertising%20Architect/Overlapping%20Experiment%20Infrastructure%20More%2C%20Better%2C%20Faster%20Experimentation.pdf) <br />

## 参考文献

[1] https://github.com/wzhe06/Ad-papers/blob/master/README.md
[2] https://github.com/duboya/ML_CIA