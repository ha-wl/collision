import re
from pathlib import Path
import json
import pandas as pd


class Args:
  def __init__(self):
    self.dataset_dir: Path = Path("../data")
    self.data_dir: Path = Path("../lancers")
    self.train_data: str = "plain.jsonl"
    self.output_data: str = "plain2.jsonl"

def main(args: Args):
  p = [[1, 3], [1, 1], [1, 2], [2, 1], [2, 3], 
       [2, 2], [1, 2], [2, 2], [3, 1], [1, 2],
       [2, 2], [3, 1], [1, 2], [2, 2], [3, 1],
       [1, 1], [1, -1], [1, 0], [2, -1], [2, 1],
       [2, 0], [1, 0], [2, 0], [3, -1], [1, 0],
       [2, 0], [3, -1], [1, 0], [2, 0], [3, -1],
       [1, 2], [1, 0], [1, 1], [2, 0], [2, 2],
       [2, 1], [1, 1], [2, 1], [3, 0], [1, 1],
       [2, 1], [3, 0], [1, 1], [2, 1], [3, 0]]
  df = pd.read_json(args.dataset_dir / args.train_data, orient='records', lines=True)
  df2 = pd.read_csv(args.data_dir / 'train_predict_翻訳付.csv', nrows=200)
  distribution = []

  for row in df.itertuples():
    id2 = row.ID
    dis = []
    pattern = row.pattern
    index = df2.loc[df2["id"]==id2].index[0]
    st = df2.iat[index, 3]
    match = re.search(r'(.+)[は|と|が](.+)[が|に|と](.*)', st)
    obj1 = match.groups()[0]
    obj2 = match.groups()[1]
    distance = p[pattern][1]

    if p[pattern][0]==1:
      if("球体" in obj1)==True :
        distance = distance + 1
    elif p[pattern][0]==2:
      if("球体" in obj2)==True :
        distance = distance + 1
    else :
      if("球体" in obj1) or ("球体" in obj2):
        distance = distance + 1

    if distance == -1:
      dis = [4, 1]
    elif distance == 0:
      dis = [6, 1]
    elif distance == 1:
      dis = [8, 1]
    elif distance == 2:
      dis = [10, 1]
    elif distance == 3:
      dis = [12, 1]
    else:
      dis = [14, 1]
    distribution.append(dis)


  df["distribution"] = distribution
  df.to_json(args.dataset_dir / args.output_data, orient='records', force_ascii=False, lines=True)





if __name__=="__main__":
  args = Args()
  main(args)
