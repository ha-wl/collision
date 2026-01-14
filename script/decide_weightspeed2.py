from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import seaborn as sns
import re
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
import embedding


class Args:
  def __init__(self):
    self.model_name: str = "sonoisa/t5-base-japanese"
    self.dataset_dir: Path = Path("../data")
    self.date = datetime.now().strftime("%Y-%m-%d/%H-%M-%S")
    self.traindata = self.dataset_dir / "train6.jsonl"
    self.valdata = self.dataset_dir / "val6.jsonl"
    self.testdata = self.dataset_dir / "test6.jsonl"
    self.plaindata = self.dataset_dir / "plain6.jsonl"
    #self.plaindata2 = self.dataset_dir / "plain7-6.jsonl"
    #self.df1 = pd.read_json(self.traindata, orient='records', lines=True)
    #self.df2 = pd.read_json(self.valdata, orient='records', lines=True)
    #self.df3 = pd.read_json(self.testdata, orient='records', lines=True)
    self.df1 = pd.read_json(self.plaindata, orient='records', lines=True)
    #self.df2 = pd.read_json(self.plaindata2, orient='records', lines=True)
    #self.answer = df.weightspeed_answer #正解データ
    #self.input = df.input #条件文
    self.em_model = embedding.SentenceT5(self.model_name, self.model_name)
    self.output_dir = Path("../outputs") / "weightspeed_nn_model" / self.date
    self.output_dir.mkdir(parents=True, exist_ok=True)
    
    

def shori(data):
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
  weight_number = []
  speed_number = []

  for row in data.itertuples():
    we = []
    sp = []
    pattern = row.pattern
    atoduke = row.atoduke
    input = row.input
    input_str = str(input)
    
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
      pass
    else:
      weight = "".join(weight)
      weight_sentence.append(weight)
      weight_number.append(weightspeed_answer[0:4])

    if ((speed_lst[pattern] %4)==3):
      pass
    else:
      speed = "".join(speed)
      speed_sentence.append(speed)
      speed_number.append(weightspeed_answer[4:8])
  
  return weight_sentence, weight_number, speed_sentence, speed_number

def main(args: Args):
  wese, wenu, spse, spnu, wese2, wenu2, spse2, spnu2 = [], [], [], [], [], [], [], []
  nn1 = 64
  nn2 = 32
  we_model = Sequential()
  we_model.add(Dense(nn1, activation='relu', input_dim=768))
  we_model.add(Dense(nn2, activation='relu'))
  we_model.add(Dense(4, activation='linear'))
  we_model.compile(
      optimizer='rmsprop',
      loss='mean_squared_error',
      metrics=['accuracy']
  )
  sp_model = Sequential()
  sp_model.add(Dense(nn1, activation='relu', input_dim=768))
  sp_model.add(Dense(nn2, activation='relu'))
  sp_model.add(Dense(4, activation='linear'))
  sp_model.compile(
      optimizer='rmsprop',
      loss='mean_squared_error',
      metrics=['accuracy']
  )
  
  # val = shori(args.df1)
  # wese += val[0]
  # wenu += val[1]
  # spse += val[2]
  # spnu += val[3]
  # val = shori(args.df2)
  # wese += val[0]
  # wenu += val[1]
  # spse += val[2]
  # spnu += val[3]
  # val = shori(args.df3)
  # wese += val[0]
  # wenu += val[1]
  # spse += val[2]
  # spnu += val[3]
  val = shori(args.df1)
  #val = shori(args.df2)
  wese += val[0]
  wenu += val[1]
  spse += val[2]
  spnu += val[3]
  print(len(wese))
  print(len(wenu))
  print(len(spse))
  print(len(spnu))

  sentence_embeddings_weight = args.em_model.encode(wese, batch_size=8)
  em_we = sentence_embeddings_weight.clone().detach()
  a_we = np.array(em_we) #入力する768次元のベクトル
  gtruth_we = np.array(wenu) #正解ラベル

  train_history_we = we_model.fit(a_we,gtruth_we,
    batch_size=50,
    epochs=1000,
    verbose=1)

  fig, ax = plt.subplots(1, 1, figsize=(6, 4))
  ax.plot(train_history_we.history['loss'])
  ax.set_xlabel('Epoch')
  ax.set_ylabel('loss')
  ax.set_xlim(0, 1000)
  ax.set_ylim(-0.01, 10)
  # plt.savefig('nn-regression-train-1.png', dpi=300, facecolor='white')
  plt.show()

  sentence_embeddings_speed = args.em_model.encode(spse, batch_size=8)
  em_sp = sentence_embeddings_speed.clone().detach()
  a_sp = np.array(em_sp) #入力する768次元のベクトル
  gtruth_sp = np.array(spnu) #正解ラベル

  train_history_sp = sp_model.fit(a_sp,gtruth_sp,
    batch_size=50,
    epochs=1000,
    verbose=1)
  
  fig, ax = plt.subplots(1, 1, figsize=(6, 4))
  ax.plot(train_history_sp.history['loss'])
  ax.set_xlabel('Epoch')
  ax.set_ylabel('loss')
  ax.set_xlim(0, 1000)
  ax.set_ylim(-0.01, 10)
  # plt.savefig('nn-regression-train-1.png', dpi=300, facecolor='white')
  plt.show()

  we_model.save(args.output_dir/ "weight")
  sp_model.save(args.output_dir/ "speed")


if __name__=="__main__":
  args = Args()
  main(args)
