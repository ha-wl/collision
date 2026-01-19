dataフォルダにはデータが、scriptフォルダにはコードがまとめられている。

dataフォルダ内にはおおまかにall, train, val, test, plainという5種類のデータが含まれており、実験ごとにそれぞれナンバリングがなされている。
allはその時点での全データ、trainは学習用データ、valは検証用データ、testはテスト用データ、plainは「やや」「とても」などの程度の表現を含めたすべての条件を網羅したテスト時に使うデータである。

scriptフォルダにはおおまかにkaiki.py, inference.py, distribution.pyという種類のデータが含まれており、実験ごとにそれぞれナンバリングがなされている。
kaiki.pyで学習を行い、inference.pyで推論を行う。また、distribution.pyで各データに対しメンバーシップ関数を付与する。
