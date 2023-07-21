# ICR
This repository is for ICR competition of Kaggle

# feature selection
* lightgbmのfeature_importancesの変動係数、SHAPのスコア、permutation importanceなどをそれぞれCVで計算してそれらを足し合わせて得たスコアで比較する
* greedy forward
* 復元抽出

# consideration from experiment
- feature selectionで行った4-foldのlightgbmでの特徴量重要度を見るとfold3だけ他のfoldと比べ重要度のブレが大きかった
  - 別の特徴を持ったレコードがあり、クラスタリングしてそれぞれに対して予測を行うことでスコアが上がるかも？
    
