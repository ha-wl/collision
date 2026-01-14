from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import seaborn as sns
import re
import matplotlib.pyplot as plt

import tensorflow as tf
import keras
from tensorflow import keras

from tensorflow.keras import layers
import math
from scipy.stats import norm

import embedding
import torch
from torch.distributions import Normal

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
  #許容範囲を決めておく
  difference1 = 1
  difference2 = 3 #かなり
  difference3 = 0.5 #やや
  weight_difference = []
  weight_taisho = []
  speed_difference = []
  speed_taisho = []


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

      if(weight_lst[pattern]==2): #質量が等しい場合
        weight_difference.append(difference1)
      elif(weight_lst[pattern]==0 or weight_lst[pattern]==1):
        if(atoduke[0]==1): #かなりがついている場合
          weight_difference.append(difference2)
        elif(atoduke[0]==-1):
          weight_difference.append(difference3)
        else:
          weight_difference.append(difference1)
      else: #質量の条件が与えられていない場合
        weight_difference.append(difference1)

      weight_taisho.append(weight_lst[pattern])

    if ((speed_lst[pattern] %4)==3):
      pass
    else:
      speed = "".join(speed)
      speed_sentence.append(speed)
      speed_number.append(weightspeed_answer[4:8])

      if((speed_lst[pattern]%4)==2): #スピードが等しい場合
        speed_difference.append(difference1)
      elif((speed_lst[pattern]%4)==0 or (speed_lst[pattern]%4)==1):
        if(atoduke[0]==1): #かなりがついている場合
          speed_difference.append(difference2)
        elif(atoduke[0]==-1):
          speed_difference.append(difference3)
        else:
          speed_difference.append(difference1)
      else: #スピードの条件が与えられていない場合
        speed_difference.append(difference1)
      
      speed_taisho.append((speed_lst[pattern])%4)

  return weight_sentence, weight_number, speed_sentence, speed_number, weight_difference, weight_taisho, speed_difference, speed_taisho


def custom_loss(y_pred, difference, taisho):
  def f1(): return y_pred[0,0]-y_pred[0,2]
  def f2(): return y_pred[0,2]-y_pred[0,0]
  def f3(): return tf.cond(y_pred[0,0]>y_pred[0,2],f1,f2)

  def f4(): return (y_pred[0,0]-y_pred[0,2]) - difference[0,0]
  def f5(): return (y_pred[0,2]-y_pred[0,0]) - difference[0,0]
  def f6(): return tf.cond(taisho[0,0]==0,f4,f5)

  loss = tf.cond(taisho[0,0]==1,f3,f6)
  #print(a2.shape)
  #tf.print(a2)
  #tf.print(loss)
  #tf.cond(taisho==1,tf.cond(y_pred[0]>y_pred[2],loss = y_pred[0]-y_pred[2],loss = y_pred[2]-y_pred[0]),tf.cond(taisho==0,loss = (y_pred[0]-y_pred[2]) - difference,loss = (y_pred[2]-y_pred[0]) - difference))

  def f7(): return loss*(-1)
  def f8(): return loss
  loss = tf.cond(loss<0,f7, f8)
  tf.print(loss)
  #loss = tf.math.reduce_mean((y_pred[0,0]-difference)**2)
  #print(loss.shape)
  return loss

def custom_loss2(y_pred, correct, taisho):
  #a_mean = y_pred[0,0].float()
  #print(a_mean)
  a=y_pred[0,0]-correct[0,0]
  b=correct[0,0]-y_pred[0,0]
  c=y_pred[0,1]**2+correct[0,0]**2
  d = (1+tf.math.erf((0-a)/tf.math.sqrt(2*c)))
  e = (1+tf.math.erf((0-b)/tf.math.sqrt(2*c)))

  
  def f0(): return 1.0/(1.0 - d/2)
  def f1(): return 1.0/(1.0 - e/2)
  def f2(): return 1.0/(1.5*tf.math.exp(-1.5*(y_pred[0,0]-correct[0,0])**2))
  def f3(): return tf.cond(taisho[0,0]==1,f1,f2)
  def f4(): return tf.cond(taisho[0,0]==0,f0,f0)

  loss = tf.cond(taisho[0,0]==0,f4,f3)

  return loss


class ModelCustomLoss(keras.Model):
  def __init__(self):
    super(ModelCustomLoss, self).__init__()
    self.layer_dense_1 = layers.Dense(768)
    self.layer_dense_2 = layers.Dense(64, activation="relu")
    self.layer_dense_3 = layers.Dense(32, activation="relu")
    self.layer_dense_out = layers.Dense(2, activation='linear')

  def call(self, inputs):
    dense_1 = self.layer_dense_1(inputs[0])
    dense_2 = self.layer_dense_2(dense_1)
    dense_3 = self.layer_dense_3(dense_2)
    out = self.layer_dense_out(dense_3)

    dif = inputs[1]
    tai = inputs[2]
    #tf.print(out)
    #print(out.shape)
    self.add_loss(custom_loss2(out, dif, tai))

    return out

def main(args: Args):
  wese, wenu, spse, spnu, wedf, weta, spdf, spta = [], [], [], [], [], [], [], []
  
  we_model = ModelCustomLoss()
  we_model.compile(
    #run_eagerly=True,
    optimizer='rmsprop',
    #loss='mean_squared_error',
    #metrics=['accuracy'],
  )
  
  val = shori(args.df1)
  wese += val[0]
  wenu += val[1]
  spse += val[2]
  spnu += val[3]
  wedf += val[4]
  weta += val[5]
  spdf += val[6]
  spta += val[7]

  sentence_embeddings_weight = args.em_model.encode(wese, batch_size=8)
  em_we = sentence_embeddings_weight.clone().detach()
  a_we = np.array(em_we) #入力する768次元のベクトル
  a_wedf = np.array(wedf)
  b_wedf = a_wedf.reshape([a_wedf.shape[0],1])
  a_weta = np.array(weta)
  b_weta = a_weta.reshape([a_weta.shape[0],1])
  a_wenu = np.array(wenu)
  b_wenu = a_wenu.reshape([a_wenu.shape[0],4])
  print(a_we.shape)
  print(b_wedf.shape)
  print(b_weta.shape)
  print(a_wenu.shape)
  
  """
  wese = np.full((3,768),2.0)
  #b_wese = a_wese.reshape([a_wese.shape[0],1])
  wedf = np.full((3,2),2.0)
  b_wedf = wedf.reshape([wedf.shape[0],2])
  weta = np.full((3,1),1)
  b_weta = weta.reshape([weta.shape[0],1])
  wenu = np.full((3,2),2)"""

  train_history_we = we_model.fit([a_we, b_wedf, b_weta],b_wenu,
    batch_size=32,
    epochs=3,
    verbose=1)
  
  """
  train_history_we = we_model.fit([a_we, b_wedf, b_weta],a_wenu,
    batch_size=32,
    epochs=3,
    verbose=1)"""


  we_model.save(args.output_dir/ "weight")
  #sp_model.save(args.output_dir/ "speed")

if __name__=="__main__":
  args = Args()
  main(args)
