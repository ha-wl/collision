import re
from pathlib import Path
import json
import pandas as pd


class Args:
  def __init__(self):
    self.dataset_dir: Path = Path("../data")
    self.data_dir: Path = Path("../lancers")
    self.train_data: str = "test6-2.jsonl"
    self.output_data: str = "test8-1.jsonl"

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
  df = pd.read_json(args.dataset_dir / args.train_data, orient='records', lines=True)
  df2 = pd.read_csv(args.data_dir / 'train_predict_翻訳付.csv', nrows=200)
  weightspeed_list = []
  distribution = []
  target = []
  label = []

  for row in df.itertuples():
    id2 = row.ID
    we = []
    sp = []
    dis = []
    pattern = row.pattern
    atoduke = row.atoduke
    input = row.input
    #index = df2.loc[df2["id"]==id2].index[0]
    #weightの定義
    if weight[pattern]== 0:
      we = [35, 4, 25, 4]
    elif weight[pattern]== 1:
      we = [25, 4, 35, 4]
    else:
      we = [30, 4, 30, 4]
    #ここで調節
    if atoduke[0]==1:
      if weight[pattern]== 0:
        we[0] += 5
        we[2] -= 5
      else:
        we[0] -= 5
        we[2] += 5
    elif atoduke[0]==-1:
      if weight[pattern]== 0:
        we[0] -= 3
        we[2] += 3
      else:
        we[0] += 3
        we[2] -= 3
    else:
      pass
    
    #speedの定義
    if speed[pattern]== 0:
      sp = [15, 4, 9, 4]
    elif speed[pattern]== 1:
      sp = [9, 4, 15, 4]
    elif (speed[pattern]== 2) or (speed[pattern]== 3) :
      sp = [12, 4, 12, 4]
    elif speed[pattern]== 4:
      sp = [11, 4, 5, 4]
    elif speed[pattern]== 5:
      sp = [5, 4, 11, 4]
    elif (speed[pattern]== 6) or (speed[pattern]== 7) :
      sp = [8, 4, 8, 4]
    elif speed[pattern]== 8:
      sp = [13, 4, 7, 4]
    elif speed[pattern]== 9:
      sp = [7, 4, 13, 4]
    else:
      sp = [10, 4, 10, 4]

    #ここで調節
    if atoduke[1]==1:#スピードにかなりがついている
      if speed[pattern]%4==0:#物体1が速い
        sp[0] += 3
        sp[2] -= 3
      elif speed[pattern]%4==1:
        sp[0] -= 3
        sp[2] += 3
      else:
        pass
    elif atoduke[1]==-1:#やや
      if speed[pattern]%4==0:
        sp[0] -= 2
        sp[2] += 2
      elif speed[pattern]%4==1:
        sp[0] += 2
        sp[2] -= 2
      else:
        pass
    #床の条件
    if atoduke[2]==1:
      if speed[pattern]//4==0:
        sp[0] += 2
        sp[2] += 2
      elif speed[pattern]//4==1:
        sp[0] -= 2
        sp[2] -= 2
      else:
        pass
    elif atoduke[2]==-1:
      if speed[pattern]//4==0:
        sp[0] -= 1
        sp[2] -= 1
      elif speed[pattern]//4==1:
        sp[0] += 1
        sp[2] += 1
      else:
        pass
    else:
      pass
    we += sp
    #ノイズを追加する

    weightspeed_list.append(we)

    #st = df2.iat[index, 3]
    st = re.findall("[^。]+[。]?", input)[0]
    match = re.search(r'(.+)[は|と|が](.+)[が|に|と](.*)', st)
    obj1 = match.groups()[0]
    obj2 = match.groups()[1]
    distance = 0
    distance2 = 0
    flag = 0 #球体という判定をしたか
    tar = [0,0]
    lab = -1
    
    #飛ばされる側と距離をポイント制で決定する
    o1 = 0
    o2 = 0
    #重さ
    if weight[pattern]== 0:#物体1が重い
      if("球体" in obj2):
        distance += 1
        flag = 1
      else:
        pass
        
      if atoduke[0]==1:
        o1 += 4
        o2 -= 4
        distance2 =-2
      elif atoduke[0]==-1:
        o1 += 1
        o2 -= 1
        distance2 =1
      else:
        o1 += 2
        o2 -= 2
    elif weight[pattern]== 1:
      if("球体" in obj1) :
        distance += 1
        flag = 1
      else:
        pass

      if atoduke[0]==1:
        o1 -= 4
        o2 += 4
        distance2 =-2
      elif atoduke[0]==-1:
        o1 -= 1
        o2 += 1
        distance2 =1
      else:
        o1 -= 2
        o2 += 2
    else:
      pass
    #スピード
    if speed[pattern]%4== 0:#物体1が速い
      if (flag==0) and ("球体" in obj2):
        distance += 1
        flag = 1
      else:
        pass    

      if atoduke[1]==1:
        o1 += 4
        o2 -= 4
        distance2 =-2
      elif atoduke[1]==-1:
        o1 += 1
        o2 -= 1
        distance2 =1
      else:
        o1 += 2
        o2 -= 2
    elif speed[pattern]%4== 1:
      if (flag==0) and ("球体" in obj1):
        distance += 1
        flag = 1
      else:
        pass 

      if atoduke[1]==1:
        o1 -= 4
        o2 += 4
        distance2 =-2
      elif atoduke[1]==-1:
        o1 -= 1
        o2 += 1
        distance2 =1
      else:
        o1 -= 2
        o2 += 2
    else:
      pass

    if (flag ==0) and (("球体" in obj1) or ("球体" in obj2)):
      distance += 1
    else:
      pass

    if o1>o2:
      distance += o1 
    elif o2>o1:
      distance += o2
    else:
      distance += o1
      

    #床の条件を追加
    if speed[pattern]//4 == 0:
      if atoduke[2]==1:
        distance += 2
        distance2 =-1
      elif atoduke[2]==-1:
        distance += 0.5
        distance2 =0.5
      else:
        distance += 1
    elif speed[pattern]//4 == 1:
      if atoduke[2]==1:
        distance -= 2
        distance2 = 1
      elif atoduke[2]==-1:
        distance -= 0.5
        distance2 = 0.5
      else:
        distance -= 1
    else:
      pass

    dis = [distance*2+6, 1]
    distribution.append(dis)

    if(atoduke[0]==1):
      tar = [(distance-2)*2+6, 1]
      lab = 1
    elif(atoduke[0]==-1):
      tar = [(distance+1)*2+6, 1]
      lab = 2
    elif(atoduke[1]==1):
      tar = [(distance-2)*2+6, 1]
      lab = 1
    elif(atoduke[1]==-1):
      tar = [(distance+1)*2+6, 1]
      lab = 2
    elif(atoduke[2]==1):
      if speed[pattern]//4 == 0:#床がツルツルしている
        tar = [(distance-1)*2+6, 1]
        lab = 1
      elif speed[pattern]//4 == 1:#床がザラザラしている
        tar = [(distance+1)*2+6, 1]
        lab = 2
      else:
        pass
    elif(atoduke[2]==-1):
      if speed[pattern]//4 == 0:#床がツルツルしている
        tar = [(distance+0.5)*2+6, 1]
        lab = 2
      elif speed[pattern]//4 == 1:#床がザラザラしている
        tar = [(distance-0.5)*2+6, 1]
        lab = 1
      else:
        pass
    else:
      tar = dis
      lab = 0
    """  
    if(distance2<0):
      lab = 1
      distance2 +=distance
      tar = [distance2*2+6, 1]
    elif(distance2>0):
      lab = 2
      distance2 +=distance
      tar = [distance2*2+6, 1]
    else:
      tar = dis
      lab=0
    """
    target.append(tar)
    label.append(lab)

  df["weightspeed"] = weightspeed_list
  df["distribution"] = distribution
  df["teido_nashi"] = target
  df["kigou"] = label
  df.to_json(args.dataset_dir / args.output_data, orient='records', force_ascii=False, lines=True)


if __name__=="__main__":
  args = Args()
  main(args)
