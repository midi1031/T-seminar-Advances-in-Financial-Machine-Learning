# imports

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold

# 7.4 パージされたパージされたk-分割交差検証法

# スニペット7.1 訓練データセットの観測データのパージング
def getTrainTimes(t1, testTimes):
  '''
  testTimesを所与として、訓練データの時点を探す
  -t1.index:観測データが開始した時点
  -t1.vakue:観測データが終了した時点
  -testTimes:テストデータの時点
  '''

  trn = t1.copy(deep=True)
  for i,j in testTimes.iteritems():
    # テストデータセットのなかで開始する訓練データ
    df0=trn[(i<=trn.index)&(trn.index<=j)].index
    # テストデータセットのなかで終了する訓練データ
    df1=trn[(i<=trn)&(trn<=j)].index
    # テストデータセットを覆う訓練データ
    df2=trn[(trn.index<=i)&(j<=trn)].index
    trn=trn.drop(df0.union(df1).union(df2))
  return trn


# スニペット7.2 訓練データのエンバーゴ
def getEmbargoTimes(times, pctEmbargo):
  # 各データのエンバーゴ時点を取得
  step=int(times.shape[0]*pctEmbargo)
  if step==0:
    mbrg=pd.Series(times[step:],index=times[:-step])
    mbrg=mbrg.append(pd.Series(times[-1], index=times[-step:]))
  return mbrg

#-----------------------------------------------------------------
# パージング前のエンバーゴを含める
# testTimes=pd.Series(mbrg[dt1], index=[dt0])
# trainTimes=getTrainTimes(t1, testTimes)
# testTimes=t1.loc[dt0:dt1].index


# スニペット7.3 観測データが重複するときの交差検証クラス
class PurgedKFold(KFold):
  '''
  区間にまたがるラベルに対して機能するようにkFoldクラスを拡張する
  訓練データのうちテストラベル区間と重複する観測値がパージされる
  テストデータセットは連続的(shuffle=False)で、間に訓練データがないとする
  '''

  def __init__(self, n_splits=3, t1=None, pctEmbargo=0.):
    if not isinstance(t1, pd.Series):
      raise ValueError('Label Through Dates must be a pd.Series')
    super(PurgedKFold, self).__init__(n_splits,shuffle=False,random_state=None)
    self.t1=t1
    self.pctEmbargo=pctEmbargo

  def split(self,X,y=None,groups=None):
    if (X.index==self.t1.index).sum()!=len(self.t1):
      raise ValueError('X and ThruDateValues must have the same index')
    indices=np.arange(X.shape[0])
    mbrg=int(X.shape[0]*self.pctEmbargo)
    test_starts=[(i[0],i[-1]+1) for i in np.array_split(np.arange(X.shape[0]),self.n_splits)]
    for i,j in test_starts:
      t0=self.t1.index[i] #テストデータセットの始まり
      test_indices=indices[i:j]
      maxT1Idx=self.t1.index.searchsorted(self.t1[test_indices].max())
      train_indices=self.t1.index.searchsorted(self.t1[self.t1<=t0].index)
      train_indices=np.concatenate((train_indices, indices[maxT1Idx+mbrg:]))
    yield train_indices, test_indices


# スニペット7.4 Purged K-Foldクラスの使用
def cvScore(clf,X,y,sample_weight,scoring='neg_log_loss',t1=None,cv=None,cvGen=None,pctEmbargo=None):
  if scoring not in ['neg_log_loss', 'accuracy']:
    raise Exception('wrong scoring method.')
  from sklearn.metrics import log_loss, accuracy_score
  # from clfSequential import PurgedKFold

  if cvGen is None:
    #パージ
    cvGen=PurgedKFold(n_splits=cv,t1=t1,pctEmbargo=pctEmbargo)
  score=[]
  for train,test in cvGen.split(X = X):
    fit = clf.fit(X = X.iloc[train, : ], y = y.iloc[train], sample_weight = sample_weight.iloc[train].values)
    if scoring == 'neg_log_loss':
      prob = fit.predict_proba(X.iloc[test, : ])
      score_ = -1 * log_loss(y.iloc[test], prob, sample_weight = sample_weight.iloc[test].values, labels = clf.classes_)
    else:
      pred = fit.predict(X.iloc[test, : ])
      score_ = accuracy_score(y.iloc[test], pred, sample_weight = sample_weight.iloc[test].values)
    score.append(score_)
  return np.array(score)