# _ring_radius_cache/README.md

## キャッシュファイルの内容
- 複数のシミュレーション条件（Vtip, Rtip, Htip）ごとの水平方向ポテンシャル分布データ
- 各キャッシュファイルには以下の配列が格納されています：
    - `Vtip`, `Rtip`, `Htip`: 各パラメータの値リスト
    - `r`: 距離（nm）
    - `data`: ポテンシャル分布データ（V）

## 使い方
1. 読み込み例：
```python
import numpy as np
cache = np.load('キャッシュファイル名.npz')
Vtip = cache['Vtip']
Rtip = cache['Rtip']
Htip = cache['Htip']
r = cache['r']
data = cache['data']
```
2. `data`の形状は `(Vtip数, Rtip数, Htip数, r数)` です。
3.　`data[i,j,k,l]` には、`Vtip[i]`, `Rtip[j]`, `Htip[k]` に対応する距離 `r[l]` でのポテンシャル値が格納されています。計算が収束しなかった部分は `np.nan` となっています。