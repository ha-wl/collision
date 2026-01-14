import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
import embedding


class Decidewesp:
  def __init__(self, df):
    self.model_name: str = "sonoisa/t5-base-japanese"
    self.answer = df.weightspeed_answer #正解データ
    self.input = df.input #条件文
    self.em_model = embedding.SentenceT5(self.model_name, self.model_name)

  def main(self):
    nn1 = 64
    nn2 = 32
    nn_model = Sequential()
    nn_model.add(Dense(nn1, activation='relu', input_dim=768))
    nn_model.add(Dense(nn2, activation='relu'))
    nn_model.add(Dense(2, activation='linear'))
    nn_model.compile(
        optimizer='rmsprop',
        loss='mean_squared_error',
        metrics=['accuracy']
    )

    sentence_embeddings = em_model.encode(self.input, batch_size=8)
    se = sentence_embeddings.clone().detach()
    a = np.array(se) #入力する768次元のベクトル
    gtruth = np.array(self.answer) #正解ラベル

    nn_model_epochs = 300
    train_history = nn_model.fit(se,gtruth,
      batch_size=50,
      epochs=nn_model_epochs,
      verbose=1)
    
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    ax.plot(train_history.history['loss'])
    ax.set_xlabel('Epoch')
    ax.set_ylabel('loss')
    ax.set_xlim(0, 300)
    ax.set_ylim(-0.01, 0.5)
    plt.show()

    pred = nn_model.predict(self.input)

    return pred

