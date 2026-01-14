import pandas as pd
import numpy as np
import seaborn as sns
from pathlib import Path
from datetime import datetime
import re
import matplotlib.pyplot as plt
from tensorflow import keras 
from keras.models import Sequential
from keras.layers import Dense
import embedding


class Args:
  def __init__(self):
    self.model_name: str = "sonoisa/t5-base-japanese"
    self.model_date: str = ("2024-07-07/11-03-24")
    self.pretrained_model: str = Path("../outputs/weightspeed_nn_model")/ self.model_date

    self.dataset_dir: Path = Path("../data")
    self.date = datetime.now().strftime("%Y-%m-%d/%H-%M-%S")
    self.traindata = self.dataset_dir / "train6.jsonl"
    self.df = pd.read_json(self.traindata, orient='records', lines=True)
    self.output_data: str = ("train7.jsonl")
    #self.valdata = args.dataset_dir / "val6.jsonl"
    #self.testdata = rgs.dataset_dir / "test6.jsonl"
    #self.plaindata = rgs.dataset_dir / "plain6.jsonl"
    #self.df1 = pd.read_json(self.traindata, orient='records', lines=True)
    #self.df2 = pd.read_json(self.valdata, orient='records', lines=True)
    #self.df3 = pd.read_json(self.testdata, orient='records', lines=True)
    #self.df4 = pd.read_json(self.plaindata, orient='records', lines=True)
    #self.answer = df.weightspeed_answer #正解データ
    #self.input = df.input #条件文  

def main(args: Args):
  em_model = embedding.SentenceT5(args.model_name, args.model_name)
  weight_model = keras.models.load_model(args.pretrained_model / "weight")
  speed_model = keras.models.load_model(args.pretrained_model / "speed")  
  #物体1の質量,物体2の質量 1が重いなら0 1が軽いなら1 等しいなら2  記載がない場合は3
  weight_lst = [0, 0, 0, 1, 1,
          1, 2, 2, 2, 0,
          1, 2, 3, 3, 3, 
          0, 0, 0, 1, 1,
          1, 2, 2, 2, 0,
          1, 2, 3, 3, 3,
          0, 0, 0, 1, 1,
          1, 2, 2, 2, 0,
          1, 2, 3, 3, 3]
  #物体1のスピード,物体2のスピード 1が速いなら0 1が遅いなら1 等しいなら2 記載がない場合は3 あとは床の状態で3パターン
  speed_lst = [0, 1, 2, 0, 1,
          2, 0, 1, 2, 3,
          3, 3, 0, 1, 2,
          4, 5, 6, 4, 5,
          6, 4, 5, 6, 7,
          7, 7, 4, 5, 6,
          8, 9, 10, 8, 9,
          10, 8, 9, 10, 11,
          11, 11, 8, 9, 10]
  weight_sentence = []
  speed_sentence = []
  weight_inference = [] #0なら推論の必要なし、1なら推論の必要あり
  speed_inference = []
  weight_number = []
  speed_number = []

  for row in args.df.itertuples():
    pattern = row.pattern
    atoduke = row.atoduke
    input = row.input
    input_str = str(input)
    we_inf = 1
    sp_inf = 1
    
    weightspeed_answer = row.weightspeed
    
    if (pattern>29):#30以上であれば床の条件がない
      if(weight_lst[pattern]==2):#等しいので一文だけ
        weight = re.findall("[^。]+[。]?", input_str)[0:2]
        if((speed_lst[pattern] %4)==2):#スピードに関してもここでとっていく
          speed = list(re.findall("[^。]+[。]?", input_str)[0]) + list(re.findall("[^。]+[。]?", input_str)[2])
        elif((speed_lst[pattern] %4)==0 or (speed_lst[pattern] %4)==1):
          speed = list(re.findall("[^。]+[。]?", input_str)[0]) + re.findall("[^。]+[。]?", input_str)[2:4]
        else:
          pass
      elif(weight_lst[pattern]==0 or weight_lst[pattern]==1):#等しくないので二文だけ
        weight = re.findall("[^。]+[。]?", input_str)[0:3]
        if((speed_lst[pattern] %4)==2):
          speed = list(re.findall("[^。]+[。]?", input_str)[0]) + list(re.findall("[^。]+[。]?", input_str)[3])
        elif((speed_lst[pattern] %4)==0 or (speed_lst[pattern] %4)==1):
          speed = list(re.findall("[^。]+[。]?", input_str)[0]) + re.findall("[^。]+[。]?", input_str)[3:5]
        else:
          pass
      else:#重さに関する言及はないけどスピードに関して書くべきところ
        if((speed_lst[pattern] %4)==2):
          speed = re.findall("[^。]+[。]?", input_str)[0:2]
        elif((speed_lst[pattern] %4)==0 or (speed_lst[pattern] %4)==1):
          speed = re.findall("[^。]+[。]?", input_str)[0:3]
        else:
          pass
    else:#床の条件がある場合
      if(weight_lst[pattern]==2):#等しいので一文だけ
        weight = list(re.findall("[^。]+[。]?", input_str)[0])  + list(re.findall("[^。]+[。]?", input_str)[2])
        if((speed_lst[pattern] %4)==2):
          speed = re.findall("[^。]+[。]?", input_str)[0:2] + list(re.findall("[^。]+[。]?", input_str)[3])
        elif((speed_lst[pattern] %4)==0 or (speed_lst[pattern] %4)==1):
          speed = re.findall("[^。]+[。]?", input_str)[0:2] + re.findall("[^。]+[。]?", input_str)[3:5]
        else:
          pass
      elif(weight_lst[pattern]==0 or weight_lst[pattern]==1):#等しくないので二文だけ
        weight = list(re.findall("[^。]+[。]?", input_str)[0]) + re.findall("[^。]+[。]?", input_str)[2:4]
        if((speed_lst[pattern] %4)==2):
          speed = re.findall("[^。]+[。]?", input_str)[0:2] + list(re.findall("[^。]+[。]?", input_str)[4])
        elif((speed_lst[pattern] %4)==0 or (speed_lst[pattern] %4)==1):
          speed = re.findall("[^。]+[。]?", input_str)[0:2] + re.findall("[^。]+[。]?", input_str)[4:6]
        else:
          pass
      else:#重さに関する言及はないけどスピードに関して書くべきところ
        if((speed_lst[pattern] %4)==2):
          speed = re.findall("[^。]+[。]?", input_str)[0:3]
        elif((speed_lst[pattern] %4)==0 or (speed_lst[pattern] %4)==1):
          speed = re.findall("[^。]+[。]?", input_str)[0:4]
        else:
          pass

    if(weight_lst[pattern]==3):
      we_inf = 0
      weight = "なし。"
    else:
      weight = "".join(weight)
    weight_sentence.append(weight)
    weight_inference.append(we_inf)
    weight_number.append(weightspeed_answer[0:4])

    if ((speed_lst[pattern] %4)==3):
      sp_inf = 0
      speed = "なし。"
    else:
      speed = "".join(speed)
    speed_sentence.append(speed)
    speed_inference.append(sp_inf)
    speed_number.append(weightspeed_answer[4:8])


  sentence_embeddings_weight = em_model.encode(weight_sentence, batch_size=8)
  em_we = sentence_embeddings_weight.clone().detach()
  a_we = np.array(em_we) #入力する768次元のベクトル
  nn_pred_we = weight_model.predict(a_we) #四桁の数字が出てくる

  sentence_embeddings_speed = em_model.encode(speed_sentence, batch_size=8)
  em_sp = sentence_embeddings_speed.clone().detach()
  a_sp = np.array(em_sp) #入力する768次元のベクトル
  nn_pred_sp = speed_model.predict(a_sp) #四桁の数字が出てくる

  i=0
  weightspeed_list = []
  for row in args.df.itertuples():
    if(weight_inference[i] == 0):
      we = weight_number[i]
    else:
      we = nn_pred_we[i]
    
    if(speed_inference[i] == 0):
      sp = speed_number[i]
    else:
      sp = nn_pred_sp[i]
    
    wese = np.concatenate([we, sp], 0)
    weightspeed_list.append(wese)
    i+=1

  args.df["weightspeed_inf"] = weightspeed_list
  args.df.to_json(args.dataset_dir / args.output_data, orient='records', force_ascii=False, lines=True)

if __name__=="__main__":
  args = Args()
  main(args)
