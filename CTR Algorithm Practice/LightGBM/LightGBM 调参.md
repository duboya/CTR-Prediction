**Step1. 学习率和估计器及其数目**

不管怎么样，我们先把学习率先定一个较高的值，这里取 `learning_rate = 0.1`，其次确定估计器`boosting/boost/boosting_type`的类型，不过默认都会选`gbdt`。

为了确定估计器的数目，也就是boosting迭代的次数，也可以说是残差树的数目，参数名为`n_estimators/num_iterations/num_round/num_boost_round`。我们可以先将该参数设成一个较大的数，然后在cv结果中查看最优的迭代次数，具体如代码。

在这之前，我们必须给其他重要的参数一个初始值。初始值的意义不大，只是为了方便确定其他参数。下面先给定一下初始值：

以下参数根据具体项目要求定：

```
'boosting_type'/'boosting': 'gbdt'
'objective': 'regression'
'metric': 'rmse'
```

以下参数我选择的初始值，你可以根据自己的情况来选择：

```
'max_depth': 6     ###   根据问题来定咯，由于我的数据集不是很大，所以选择了一个适中的值，其实4-10都无所谓。
'num_leaves': 50  ###   由于lightGBM是leaves_wise生长，官方说法是要小于2^max_depth
'subsample'/'bagging_fraction':0.8           ###  数据采样
'colsample_bytree'/'feature_fraction': 0.8  ###  特征采样
```

下面我是用LightGBM的cv函数进行演示：

```
params = {
    'boosting_type': 'gbdt', 
    'objective': 'regression', 

    'learning_rate': 0.1, 
    'num_leaves': 50, 
    'max_depth': 6,

    'subsample': 0.8, 
    'colsample_bytree': 0.8, 
    }
data_train = lgb.Dataset(df_train, y_train, silent=True)
cv_results = lgb.cv(
    params, data_train, num_boost_round=1000, nfold=5, stratified=False, shuffle=True, metrics='rmse',
    early_stopping_rounds=50, verbose_eval=50, show_stdv=True, seed=0)

print('best n_estimators:', len(cv_results['rmse-mean']))
print('best cv score:', cv_results['rmse-mean'][-1])
[50]    cv_agg's rmse: 1.38497 + 0.0202823
best n_estimators: 43
best cv score: 1.3838664241
```

由于我的数据集不是很大，所以在学习率为0.1时，最优的迭代次数只有43。那么现在，我们就代入(0.1, 43)进入其他参数的tuning。但是还是建议，在硬件条件允许的条件下，学习率还是越小越好。

**Step2. max_depth 和 num_leaves**

这是提高精确度的最重要的参数。

`max_depth` ：设置树深度，深度越大可能过拟合

`num_leaves`：因为 LightGBM 使用的是 leaf-wise 的算法，因此在调节树的复杂程度时，使用的是 num_leaves 而不是 max_depth。大致换算关系：num_leaves = 2^(max_depth)，但是它的值的设置应该小于 2^(max_depth)，否则可能会导致过拟合。

我们也可以同时调节这两个参数，对于这两个参数调优，我们先粗调，再细调：

这里我们引入`sklearn`里的`GridSearchCV()`函数进行搜索。不知道怎的，这个函数特别耗内存，特别耗时间，特别耗精力。

```
from sklearn.model_selection import GridSearchCV
### 我们可以创建lgb的sklearn模型，使用上面选择的(学习率，评估器数目)
model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=50,
                              learning_rate=0.1, n_estimators=43, max_depth=6,
                              metric='rmse', bagging_fraction = 0.8,feature_fraction = 0.8)

params_test1={
    'max_depth': range(3,8,2),
    'num_leaves':range(50, 170, 30)
}
gsearch1 = GridSearchCV(estimator=model_lgb, param_grid=params_test1, scoring='neg_mean_squared_error', cv=5, verbose=1, n_jobs=4)
gsearch1.fit(df_train, y_train)
gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_
Fitting 5 folds for each of 12 candidates, totalling 60 fits


[Parallel(n_jobs=4)]: Done  42 tasks      | elapsed:  2.0min
[Parallel(n_jobs=4)]: Done  60 out of  60 | elapsed:  3.1min finished


([mean: -1.88629, std: 0.13750, params: {'max_depth': 3, 'num_leaves': 50},
  mean: -1.88629, std: 0.13750, params: {'max_depth': 3, 'num_leaves': 80},
  mean: -1.88629, std: 0.13750, params: {'max_depth': 3, 'num_leaves': 110},
  mean: -1.88629, std: 0.13750, params: {'max_depth': 3, 'num_leaves': 140},
  mean: -1.86917, std: 0.12590, params: {'max_depth': 5, 'num_leaves': 50},
  mean: -1.86917, std: 0.12590, params: {'max_depth': 5, 'num_leaves': 80},
  mean: -1.86917, std: 0.12590, params: {'max_depth': 5, 'num_leaves': 110},
  mean: -1.86917, std: 0.12590, params: {'max_depth': 5, 'num_leaves': 140},
  mean: -1.89254, std: 0.10904, params: {'max_depth': 7, 'num_leaves': 50},
  mean: -1.86024, std: 0.11364, params: {'max_depth': 7, 'num_leaves': 80},
  mean: -1.86024, std: 0.11364, params: {'max_depth': 7, 'num_leaves': 110},
  mean: -1.86024, std: 0.11364, params: {'max_depth': 7, 'num_leaves': 140}],
 {'max_depth': 7, 'num_leaves': 80},
 -1.8602436718814157)
```

这里，我们运行了12个参数组合，得到的最优解是在max_depth为7，num_leaves为80的情况下，分数为-1.860。

这里必须说一下，sklearn模型评估里的scoring参数都是采用的**higher return values are better than lower return values（较高的返回值优于较低的返回值）**。

但是，我采用的metric策略采用的是均方误差(rmse)，越低越好，所以sklearn就提供了`neg_mean_squared_erro`参数，也就是返回metric的负数，所以就均方差来说，也就变成负数越大越好了。

所以，可以看到，最优解的分数为-1.860，转化为均方差为np.sqrt(-(-1.860)) = 1.3639，明显比step1的分数要好很多。

至此，我们将我们这步得到的最优解代入第三步。其实，我这里只进行了粗调，如果要得到更好的效果，可以将max_depth在7附近多取几个值，num_leaves在80附近多取几个值。千万不要怕麻烦，虽然这确实很麻烦。

```
params_test2={
    'max_depth': [6,7,8],
    'num_leaves':[68,74,80,86,92]
}

gsearch2 = GridSearchCV(estimator=model_lgb, param_grid=params_test2, scoring='neg_mean_squared_error', cv=5, verbose=1, n_jobs=4)
gsearch2.fit(df_train, y_train)
gsearch2.grid_scores_, gsearch2.best_params_, gsearch2.best_score_
Fitting 5 folds for each of 15 candidates, totalling 75 fits


[Parallel(n_jobs=4)]: Done  42 tasks      | elapsed:  2.8min
[Parallel(n_jobs=4)]: Done  75 out of  75 | elapsed:  5.1min finished


([mean: -1.87506, std: 0.11369, params: {'max_depth': 6, 'num_leaves': 68},
  mean: -1.87506, std: 0.11369, params: {'max_depth': 6, 'num_leaves': 74},
  mean: -1.87506, std: 0.11369, params: {'max_depth': 6, 'num_leaves': 80},
  mean: -1.87506, std: 0.11369, params: {'max_depth': 6, 'num_leaves': 86},
  mean: -1.87506, std: 0.11369, params: {'max_depth': 6, 'num_leaves': 92},
  mean: -1.86024, std: 0.11364, params: {'max_depth': 7, 'num_leaves': 68},
  mean: -1.86024, std: 0.11364, params: {'max_depth': 7, 'num_leaves': 74},
  mean: -1.86024, std: 0.11364, params: {'max_depth': 7, 'num_leaves': 80},
  mean: -1.86024, std: 0.11364, params: {'max_depth': 7, 'num_leaves': 86},
  mean: -1.86024, std: 0.11364, params: {'max_depth': 7, 'num_leaves': 92},
  mean: -1.88197, std: 0.11295, params: {'max_depth': 8, 'num_leaves': 68},
  mean: -1.89117, std: 0.12686, params: {'max_depth': 8, 'num_leaves': 74},
  mean: -1.86390, std: 0.12259, params: {'max_depth': 8, 'num_leaves': 80},
  mean: -1.86733, std: 0.12159, params: {'max_depth': 8, 'num_leaves': 86},
  mean: -1.86665, std: 0.12174, params: {'max_depth': 8, 'num_leaves': 92}],
 {'max_depth': 7, 'num_leaves': 68},
 -1.8602436718814157)
```

可见最大深度7是没问题的，但是看细节的话，发现在最大深度为7的情况下，叶结点的数量对分数并没有影响。

**Step3: min_data_in_leaf 和 min_sum_hessian_in_leaf**

说到这里，就该降低过拟合了。

`min_data_in_leaf` 是一个很重要的参数, 也叫min_child_samples，它的值取决于训练数据的样本个树和num_leaves. 将其设置的较大可以避免生成一个过深的树, 但有可能导致欠拟合。

`min_sum_hessian_in_leaf`：也叫min_child_weight，使一个结点分裂的最小海森值之和，真拗口（Minimum sum of hessians in one leaf to allow a split. Higher values potentially decrease overfitting）

我们采用跟上面相同的方法进行：

```
params_test3={
    'min_child_samples': [18, 19, 20, 21, 22],
    'min_child_weight':[0.001, 0.002]
}
model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=80,
                              learning_rate=0.1, n_estimators=43, max_depth=7, 
                              metric='rmse', bagging_fraction = 0.8, feature_fraction = 0.8)
gsearch3 = GridSearchCV(estimator=model_lgb, param_grid=params_test3, scoring='neg_mean_squared_error', cv=5, verbose=1, n_jobs=4)
gsearch3.fit(df_train, y_train)
gsearch3.grid_scores_, gsearch3.best_params_, gsearch3.best_score_
Fitting 5 folds for each of 10 candidates, totalling 50 fits


[Parallel(n_jobs=4)]: Done  42 tasks      | elapsed:  2.9min
[Parallel(n_jobs=4)]: Done  50 out of  50 | elapsed:  3.3min finished


([mean: -1.88057, std: 0.13948, params: {'min_child_samples': 18, 'min_child_weight': 0.001},
  mean: -1.88057, std: 0.13948, params: {'min_child_samples': 18, 'min_child_weight': 0.002},
  mean: -1.88365, std: 0.13650, params: {'min_child_samples': 19, 'min_child_weight': 0.001},
  mean: -1.88365, std: 0.13650, params: {'min_child_samples': 19, 'min_child_weight': 0.002},
  mean: -1.86024, std: 0.11364, params: {'min_child_samples': 20, 'min_child_weight': 0.001},
  mean: -1.86024, std: 0.11364, params: {'min_child_samples': 20, 'min_child_weight': 0.002},
  mean: -1.86980, std: 0.14251, params: {'min_child_samples': 21, 'min_child_weight': 0.001},
  mean: -1.86980, std: 0.14251, params: {'min_child_samples': 21, 'min_child_weight': 0.002},
  mean: -1.86750, std: 0.13898, params: {'min_child_samples': 22, 'min_child_weight': 0.001},
  mean: -1.86750, std: 0.13898, params: {'min_child_samples': 22, 'min_child_weight': 0.002}],
 {'min_child_samples': 20, 'min_child_weight': 0.001},
 -1.8602436718814157)
```

这是我经过粗调后细调的结果，可以看到，min_data_in_leaf的最优值为20，而min_sum_hessian_in_leaf对最后的值几乎没有影响。且这里调参之后，最后的值没有进行优化，说明之前的默认值即为20，0.001。

**Step4: feature_fraction 和 bagging_fraction**

这两个参数都是为了降低过拟合的。

feature_fraction参数来进行特征的子抽样。这个参数可以用来防止过拟合及提高训练速度。

bagging_fraction+bagging_freq参数必须同时设置，bagging_fraction相当于subsample样本采样，可以使bagging更快的运行，同时也可以降拟合。bagging_freq默认0，表示bagging的频率，0意味着没有使用bagging，k意味着每k轮迭代进行一次bagging。

不同的参数，同样的方法。

```
params_test4={
    'feature_fraction': [0.5, 0.6, 0.7, 0.8, 0.9],
    'bagging_fraction': [0.6, 0.7, 0.8, 0.9, 1.0]
}
model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=80,
                              learning_rate=0.1, n_estimators=43, max_depth=7, 
                              metric='rmse', bagging_freq = 5,  min_child_samples=20)
gsearch4 = GridSearchCV(estimator=model_lgb, param_grid=params_test4, scoring='neg_mean_squared_error', cv=5, verbose=1, n_jobs=4)
gsearch4.fit(df_train, y_train)
gsearch4.grid_scores_, gsearch4.best_params_, gsearch4.best_score_
Fitting 5 folds for each of 25 candidates, totalling 125 fits


[Parallel(n_jobs=4)]: Done  42 tasks      | elapsed:  2.6min
[Parallel(n_jobs=4)]: Done 125 out of 125 | elapsed:  7.1min finished


([mean: -1.90447, std: 0.15841, params: {'bagging_fraction': 0.6, 'feature_fraction': 0.5},
  mean: -1.90846, std: 0.13925, params: {'bagging_fraction': 0.6, 'feature_fraction': 0.6},
  mean: -1.91695, std: 0.14121, params: {'bagging_fraction': 0.6, 'feature_fraction': 0.7},
  mean: -1.90115, std: 0.12625, params: {'bagging_fraction': 0.6, 'feature_fraction': 0.8},
  mean: -1.92586, std: 0.15220, params: {'bagging_fraction': 0.6, 'feature_fraction': 0.9},
  mean: -1.88031, std: 0.17157, params: {'bagging_fraction': 0.7, 'feature_fraction': 0.5},
  mean: -1.89513, std: 0.13718, params: {'bagging_fraction': 0.7, 'feature_fraction': 0.6},
  mean: -1.88845, std: 0.13864, params: {'bagging_fraction': 0.7, 'feature_fraction': 0.7},
  mean: -1.89297, std: 0.12374, params: {'bagging_fraction': 0.7, 'feature_fraction': 0.8},
  mean: -1.89432, std: 0.14353, params: {'bagging_fraction': 0.7, 'feature_fraction': 0.9},
  mean: -1.88088, std: 0.14247, params: {'bagging_fraction': 0.8, 'feature_fraction': 0.5},
  mean: -1.90080, std: 0.13174, params: {'bagging_fraction': 0.8, 'feature_fraction': 0.6},
  mean: -1.88364, std: 0.14732, params: {'bagging_fraction': 0.8, 'feature_fraction': 0.7},
  mean: -1.88987, std: 0.13344, params: {'bagging_fraction': 0.8, 'feature_fraction': 0.8},
  mean: -1.87752, std: 0.14802, params: {'bagging_fraction': 0.8, 'feature_fraction': 0.9},
  mean: -1.88348, std: 0.13925, params: {'bagging_fraction': 0.9, 'feature_fraction': 0.5},
  mean: -1.87472, std: 0.13301, params: {'bagging_fraction': 0.9, 'feature_fraction': 0.6},
  mean: -1.88656, std: 0.12241, params: {'bagging_fraction': 0.9, 'feature_fraction': 0.7},
  mean: -1.89029, std: 0.10776, params: {'bagging_fraction': 0.9, 'feature_fraction': 0.8},
  mean: -1.88719, std: 0.11915, params: {'bagging_fraction': 0.9, 'feature_fraction': 0.9},
  mean: -1.86170, std: 0.12544, params: {'bagging_fraction': 1.0, 'feature_fraction': 0.5},
  mean: -1.87334, std: 0.13099, params: {'bagging_fraction': 1.0, 'feature_fraction': 0.6},
  mean: -1.85412, std: 0.12698, params: {'bagging_fraction': 1.0, 'feature_fraction': 0.7},
  mean: -1.86024, std: 0.11364, params: {'bagging_fraction': 1.0, 'feature_fraction': 0.8},
  mean: -1.87266, std: 0.12271, params: {'bagging_fraction': 1.0, 'feature_fraction': 0.9}],
 {'bagging_fraction': 1.0, 'feature_fraction': 0.7},
 -1.8541224387666373)
```

从这里可以看出来，bagging_feaction和feature_fraction的理想值分别是1.0和0.7，一个很重要原因就是，我的样本数量比较小(4000+)，但是特征数量很多(1000+)。所以，这里我们取更小的步长，对feature_fraction进行更细致的取值。

```
params_test5={
    'feature_fraction': [0.62, 0.65, 0.68, 0.7, 0.72, 0.75, 0.78 ]
}
model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=80,
                              learning_rate=0.1, n_estimators=43, max_depth=7, 
                              metric='rmse',  min_child_samples=20)
gsearch5 = GridSearchCV(estimator=model_lgb, param_grid=params_test5, scoring='neg_mean_squared_error', cv=5, verbose=1, n_jobs=4)
gsearch5.fit(df_train, y_train)
gsearch5.grid_scores_, gsearch5.best_params_, gsearch5.best_score_
Fitting 5 folds for each of 7 candidates, totalling 35 fits


[Parallel(n_jobs=4)]: Done  35 out of  35 | elapsed:  2.3min finished


([mean: -1.86696, std: 0.12658, params: {'feature_fraction': 0.62},
  mean: -1.88337, std: 0.13215, params: {'feature_fraction': 0.65},
  mean: -1.87282, std: 0.13193, params: {'feature_fraction': 0.68},
  mean: -1.85412, std: 0.12698, params: {'feature_fraction': 0.7},
  mean: -1.88235, std: 0.12682, params: {'feature_fraction': 0.72},
  mean: -1.86329, std: 0.12757, params: {'feature_fraction': 0.75},
  mean: -1.87943, std: 0.12107, params: {'feature_fraction': 0.78}],
 {'feature_fraction': 0.7},
 -1.8541224387666373)
```

好吧，feature_fraction就是0.7了

**Step5: 正则化参数**

正则化参数lambda_l1(reg_alpha), lambda_l2(reg_lambda)，毫无疑问，是降低过拟合的，两者分别对应l1正则化和l2正则化。我们也来尝试一下使用这两个参数。

```
params_test6={
    'reg_alpha': [0, 0.001, 0.01, 0.03, 0.08, 0.3, 0.5],
    'reg_lambda': [0, 0.001, 0.01, 0.03, 0.08, 0.3, 0.5]
}
model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=80,
                              learning_rate=0.b1, n_estimators=43, max_depth=7, 
                              metric='rmse',  min_child_samples=20, feature_fraction=0.7)
gsearch6 = GridSearchCV(estimator=model_lgb, param_grid=params_test6, scoring='neg_mean_squared_error', cv=5, verbose=1, n_jobs=4)
gsearch6.fit(df_train, y_train)
gsearch6.grid_scores_, gsearch6.best_params_, gsearch6.best_score_
Fitting 5 folds for each of 49 candidates, totalling 245 fits


[Parallel(n_jobs=4)]: Done  42 tasks      | elapsed:  2.8min
[Parallel(n_jobs=4)]: Done 192 tasks      | elapsed: 10.6min
[Parallel(n_jobs=4)]: Done 245 out of 245 | elapsed: 13.3min finished


([mean: -1.85412, std: 0.12698, params: {'reg_alpha': 0, 'reg_lambda': 0},
  mean: -1.85990, std: 0.13296, params: {'reg_alpha': 0, 'reg_lambda': 0.001},
  mean: -1.86367, std: 0.13634, params: {'reg_alpha': 0, 'reg_lambda': 0.01},
  mean: -1.86787, std: 0.13881, params: {'reg_alpha': 0, 'reg_lambda': 0.03},
  mean: -1.87099, std: 0.12476, params: {'reg_alpha': 0, 'reg_lambda': 0.08},
  mean: -1.87670, std: 0.11849, params: {'reg_alpha': 0, 'reg_lambda': 0.3},
  mean: -1.88278, std: 0.13064, params: {'reg_alpha': 0, 'reg_lambda': 0.5},
  mean: -1.86190, std: 0.13613, params: {'reg_alpha': 0.001, 'reg_lambda': 0},
  mean: -1.86190, std: 0.13613, params: {'reg_alpha': 0.001, 'reg_lambda': 0.001},
  mean: -1.86515, std: 0.14116, params: {'reg_alpha': 0.001, 'reg_lambda': 0.01},
  mean: -1.86908, std: 0.13668, params: {'reg_alpha': 0.001, 'reg_lambda': 0.03},
  mean: -1.86852, std: 0.12289, params: {'reg_alpha': 0.001, 'reg_lambda': 0.08},
  mean: -1.88076, std: 0.11710, params: {'reg_alpha': 0.001, 'reg_lambda': 0.3},
  mean: -1.88278, std: 0.13064, params: {'reg_alpha': 0.001, 'reg_lambda': 0.5},
  mean: -1.87480, std: 0.13889, params: {'reg_alpha': 0.01, 'reg_lambda': 0},
  mean: -1.87284, std: 0.14138, params: {'reg_alpha': 0.01, 'reg_lambda': 0.001},
  mean: -1.86030, std: 0.13332, params: {'reg_alpha': 0.01, 'reg_lambda': 0.01},
  mean: -1.86695, std: 0.12587, params: {'reg_alpha': 0.01, 'reg_lambda': 0.03},
  mean: -1.87415, std: 0.13100, params: {'reg_alpha': 0.01, 'reg_lambda': 0.08},
  mean: -1.88543, std: 0.13195, params: {'reg_alpha': 0.01, 'reg_lambda': 0.3},
  mean: -1.88076, std: 0.13502, params: {'reg_alpha': 0.01, 'reg_lambda': 0.5},
  mean: -1.87729, std: 0.12533, params: {'reg_alpha': 0.03, 'reg_lambda': 0},
  mean: -1.87435, std: 0.12034, params: {'reg_alpha': 0.03, 'reg_lambda': 0.001},
  mean: -1.87513, std: 0.12579, params: {'reg_alpha': 0.03, 'reg_lambda': 0.01},
  mean: -1.88116, std: 0.12218, params: {'reg_alpha': 0.03, 'reg_lambda': 0.03},
  mean: -1.88052, std: 0.13585, params: {'reg_alpha': 0.03, 'reg_lambda': 0.08},
  mean: -1.87565, std: 0.12200, params: {'reg_alpha': 0.03, 'reg_lambda': 0.3},
  mean: -1.87935, std: 0.13817, params: {'reg_alpha': 0.03, 'reg_lambda': 0.5},
  mean: -1.87774, std: 0.12477, params: {'reg_alpha': 0.08, 'reg_lambda': 0},
  mean: -1.87774, std: 0.12477, params: {'reg_alpha': 0.08, 'reg_lambda': 0.001},
  mean: -1.87911, std: 0.12027, params: {'reg_alpha': 0.08, 'reg_lambda': 0.01},
  mean: -1.86978, std: 0.12478, params: {'reg_alpha': 0.08, 'reg_lambda': 0.03},
  mean: -1.87217, std: 0.12159, params: {'reg_alpha': 0.08, 'reg_lambda': 0.08},
  mean: -1.87573, std: 0.14137, params: {'reg_alpha': 0.08, 'reg_lambda': 0.3},
  mean: -1.85969, std: 0.13109, params: {'reg_alpha': 0.08, 'reg_lambda': 0.5},
  mean: -1.87632, std: 0.12398, params: {'reg_alpha': 0.3, 'reg_lambda': 0},
  mean: -1.86995, std: 0.12651, params: {'reg_alpha': 0.3, 'reg_lambda': 0.001},
  mean: -1.86380, std: 0.12793, params: {'reg_alpha': 0.3, 'reg_lambda': 0.01},
  mean: -1.87577, std: 0.13002, params: {'reg_alpha': 0.3, 'reg_lambda': 0.03},
  mean: -1.87402, std: 0.13496, params: {'reg_alpha': 0.3, 'reg_lambda': 0.08},
  mean: -1.87032, std: 0.12504, params: {'reg_alpha': 0.3, 'reg_lambda': 0.3},
  mean: -1.88329, std: 0.13237, params: {'reg_alpha': 0.3, 'reg_lambda': 0.5},
  mean: -1.87196, std: 0.13099, params: {'reg_alpha': 0.5, 'reg_lambda': 0},
  mean: -1.87196, std: 0.13099, params: {'reg_alpha': 0.5, 'reg_lambda': 0.001},
  mean: -1.88222, std: 0.14735, params: {'reg_alpha': 0.5, 'reg_lambda': 0.01},
  mean: -1.86618, std: 0.14006, params: {'reg_alpha': 0.5, 'reg_lambda': 0.03},
  mean: -1.88579, std: 0.12398, params: {'reg_alpha': 0.5, 'reg_lambda': 0.08},
  mean: -1.88297, std: 0.12307, params: {'reg_alpha': 0.5, 'reg_lambda': 0.3},
  mean: -1.88148, std: 0.12622, params: {'reg_alpha': 0.5, 'reg_lambda': 0.5}],
 {'reg_alpha': 0, 'reg_lambda': 0},
 -1.8541224387666373)
```

哈哈，看来我多此一举了。

**step6: 降低learning_rate**

之前使用较高的学习速率是因为可以让收敛更快，但是准确度肯定没有细水长流来的好。最后，我们使用较低的学习速率，以及使用更多的决策树n_estimators来训练数据，看能不能可以进一步的优化分数。

我们可以用回lightGBM的cv函数了 ，我们代入之前优化好的参数。

```
params = {
    'boosting_type': 'gbdt', 
    'objective': 'regression', 

    'learning_rate': 0.005, 
    'num_leaves': 80, 
    'max_depth': 7,
    'min_data_in_leaf': 20,

    'subsample': 1, 
    'colsample_bytree': 0.7, 
    }

data_train = lgb.Dataset(df_train, y_train, silent=True)
cv_results = lgb.cv(
    params, data_train, num_boost_round=10000, nfold=5, stratified=False, shuffle=True, metrics='rmse',
    early_stopping_rounds=50, verbose_eval=100, show_stdv=True)

print('best n_estimators:', len(cv_results['rmse-mean']))
print('best cv score:', cv_results['rmse-mean'][-1])
[100]   cv_agg's rmse: 1.52939 + 0.0261756
[200]   cv_agg's rmse: 1.43535 + 0.0187243
[300]   cv_agg's rmse: 1.39584 + 0.0157521
[400]   cv_agg's rmse: 1.37935 + 0.0157429
[500]   cv_agg's rmse: 1.37313 + 0.0164503
[600]   cv_agg's rmse: 1.37081 + 0.0172752
[700]   cv_agg's rmse: 1.36942 + 0.0177888
[800]   cv_agg's rmse: 1.36854 + 0.0180575
[900]   cv_agg's rmse: 1.36817 + 0.0188776
[1000]  cv_agg's rmse: 1.36796 + 0.0190279
[1100]  cv_agg's rmse: 1.36783 + 0.0195969
best n_estimators: 1079
best cv score: 1.36772351783
```

这就是一个大概过程吧，其实也有更高级的方法，但是这种基本的对于GBM模型的调参方法也是需要了解的吧。如有问题，请多指教。



