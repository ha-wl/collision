import re
from pathlib import Path
import json
import pandas as pd


class Args:
  def __init__(self):
    self.dataset_dir: Path = Path("../data")
    self.data_dir: Path = Path("../lancers")
    self.train_data: str = "plain5.jsonl"
    self.output_data: str = "plain6.jsonl"

def main(args: Args):
  #物体1の質量,物体2の質量 1が重いなら0 1が軽いなら1 等しいなら2  記載がない場合は3
  weight = [0, 0, 0, 1, 1,
            1, 2, 2, 2, 0,
            1, 2, 3, 3, 3, 
            0, 0, 0, 1, 1,
            1, 2, 2, 2, 0,
            1, 2, 3, 3, 3,
            0, 0, 0, 1, 1,
            1, 2, 2, 2, 0,
            1, 2, 3, 3, 3]
  #物体1のスピード,物体2のスピード 1が速いなら0 1が遅いなら1 等しいなら2 記載がない場合は3 あとは床の状態で3パターン
  speed = [0, 1, 2, 0, 1,
           2, 0, 1, 2, 3,
           3, 3, 0, 1, 2,
           4, 5, 6, 4, 5,
           6, 4, 5, 6, 7,
           7, 7, 4, 5, 6,
           8, 9, 10, 8, 9,
           10, 8, 9, 10, 11,
           11, 11, 8, 9, 10]
  #飛ぶ距離 飛ばされる対象と飛ばされる距離
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
  weightspeed_list = []
  distribution = []

  for row in df.itertuples():
    id2 = row.ID
    we = []
    sp = []
    dis = []
    pattern = row.pattern
    distance = p[pattern][1]
    #atoduke[0]は質量, atoduke[1]はスピード, atoduke[2]は床
    #1は「とても」がついていて、-1は「やや」がついている
    atoduke = row.atoduke
    index = df2.loc[df2["id"]==id2].index[0]

    if weight[pattern]== 0:
      if atoduke[0]==1:
        we = [40, 4, 20, 4]
      elif atoduke[0]==-1:
        we = [32, 4, 28, 4]
      else:
        we = [35, 4, 25, 4]
    elif weight[pattern]== 1:
      if atoduke[0]==1:
        we = [20, 4, 40, 4]
      elif atoduke[0]==-1:
        we = [28, 4, 32, 4]
      else:
        we = [25, 4, 35, 4]
    else:
      we = [30, 4, 30, 4]


    #床がツルツルしている
    if speed[pattern]== 0:#物体1のほうが速い
      if atoduke[1]==1:#スピードに「かなり」という表現がついている
        if atoduke[2]==1:#床の表現に「かなり」という表現がついている
          sp = [20, 4, 8, 4]
        elif atoduke[2]==-1:
          sp = [16, 4, 4, 4]
        else:
          sp = [18, 4, 6, 4]
      elif atoduke[1]==-1:
        if atoduke[2]==1:
          sp = [15, 4, 13, 4]
        elif atoduke[2]==-1:
          sp = [11, 4, 9, 4]
        else:
          sp = [13, 4, 11, 4]
      else:
        if atoduke[2]==1:
          sp = [16, 4, 12, 4]
        elif atoduke[2]==-1:
          sp = [12, 4, 8, 4]
        else:
          sp = [14, 4, 10, 4]
    elif speed[pattern]== 1:
      if atoduke[1]==1:
        if atoduke[2]==1:#床の表現に「かなり」という表現がついている
          sp = [8, 4, 20, 4]
        elif atoduke[2]==-1:
          sp = [4, 4, 16, 4]
        else:
          sp = [6, 4, 18, 4]
      elif atoduke[1]==-1:
        if atoduke[2]==1:
          sp = [13, 4, 15, 4]
        elif atoduke[2]==-1:
          sp = [9, 4, 11, 4]
        else:
          sp = [11, 4, 13, 4]
      else:
        if atoduke[2]==1:
          sp = [12, 4, 16, 4]
        elif atoduke[2]==-1:
          sp = [8, 4, 12, 4]
        else:
          sp = [10, 4, 14, 4]
    elif (speed[pattern]== 2) or (speed[pattern]== 3) :
      if atoduke[2]==1:
        sp = [14, 4, 14, 4]
      elif atoduke[2]==-1:
        sp = [10, 4, 10, 4]
      else:
        sp = [12, 4, 12, 4]
    #床がザラザラしている
    elif speed[pattern]== 4:
      if atoduke[1]==1:
        if atoduke[2]==1:
          sp = [12, 4, 8, 4]
        elif atoduke[2]==-1:
          sp = [16, 4, 12, 4]
        else:
          sp = [14, 4, 10, 4]
      elif atoduke[1]==-1:
        if atoduke[2]==1:
          sp = [7, 4, 5, 4]
        elif atoduke[2]==-1:
          sp = [11, 4, 9, 4]
        else:
          sp = [9, 4, 7, 4]
      else:
        if atoduke[2]==1:
          sp = [8, 4, 4, 4]
        elif atoduke[2]==-1:
          sp = [12, 4, 8, 4]
        else:
          sp = [10, 4, 6, 4]
    elif speed[pattern]== 5:
      if atoduke[1]==1:
        if atoduke[2]==1:
          sp = [8, 4, 12, 4]
        elif atoduke[2]==-1:
          sp = [12, 4, 16, 4]
        else:
          sp = [10, 4, 14, 4]
      elif atoduke[1]==-1:
        if atoduke[2]==1:
          sp = [5, 4, 7, 4]
        elif atoduke[2]==-1:
          sp = [9, 4, 11, 4]
        else:
          sp = [7, 4, 9, 4]
      else:
        if atoduke[2]==1:
          sp = [4, 4, 8, 4]
        elif atoduke[2]==-1:
          sp = [8, 4, 12, 4]
        else:
          sp = [6, 4, 10, 4]
    elif (speed[pattern]== 6) or (speed[pattern]== 7) :
      if atoduke[2]==1:
        sp = [6, 4, 6, 4]
      elif atoduke[2]==-1:
        sp = [10, 4, 10, 4]
      else:
        sp = [8, 4, 8, 4]
    elif speed[pattern]== 8:
      if atoduke[1]==1:
        sp = [16, 4, 4, 4]
      elif atoduke[1]==-1:
        sp = [11, 4, 9, 4]
      else:
        sp = [12, 4, 8, 4]
    elif speed[pattern]== 9:
      if atoduke[1]==1:
        sp = [4, 4, 16, 4]
      elif atoduke[1]==-1:
        sp = [9, 4, 11, 4]
      else:
        sp = [8, 4, 12, 4]
    else:
      sp = [10, 4, 10, 4]
    we += sp
    weightspeed_list.append(we)

    st = df2.iat[index, 3]
    match = re.search(r'(.+)[は|と|が](.+)[が|に|と](.*)', st)
    obj1 = match.groups()[0]
    obj2 = match.groups()[1]
    #distance = p[pattern][1]
    if p[pattern][0]==1:
      if("球体" in obj1) :
        distance = distance + 1
    elif p[pattern][0]==2:
      if("球体" in obj2) :
        distance = distance + 1
    else :
      if ("球体" in obj1) or ("球体" in obj2):
        distance = distance + 1

    #atodukeで距離を変える処理
    #質量
    if atoduke[0]==1:
      distance = distance + 1
    elif atoduke[0]==-1:
      distance = distance - 0.5
    else:
      pass

    #スピード
    if atoduke[1]==1:
      distance = distance + 1
    elif atoduke[1]==-1:
      distance = distance - 0.5
    else:
      pass

    #床
    if atoduke[2]==1:#かなりがついている
      if speed[pattern]== 0 or speed[pattern]== 1 or speed[pattern]== 2 or speed[pattern]== 3:
        distance = distance + 1
      elif speed[pattern]== 4 or speed[pattern]== 5 or speed[pattern]== 6 or speed[pattern]== 7:
        distance = distance - 0.5
      else:
        pass
    elif atoduke[2]==-1:#ややがついている
      if speed[pattern]== 0 or speed[pattern]== 1 or speed[pattern]== 2 or speed[pattern]== 3:
        distance = distance - 0.5
      elif speed[pattern]== 4 or speed[pattern]== 5 or speed[pattern]== 6 or speed[pattern]== 7:
        distance = distance + 0.5
      else:
        pass
    else:
      pass

    a = 6 + distance * 2
    dis = [a, 1]
    distribution.append(dis)

  df["weightspeed"] = weightspeed_list
  df["distribution"] = distribution
  df.to_json(args.dataset_dir / args.output_data, orient='records', force_ascii=False, lines=True)



if __name__=="__main__":
  args = Args()
  main(args)
