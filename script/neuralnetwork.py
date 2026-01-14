import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense

class Args:
    def __init__(self): #学習データの初期化
        self.model_name: str = "sonoisa/t5-base-japanese" 
        #self.model_name: str = "megagonlabs/t5-base-japanese-web"
        # self.model_name: str = "nlp-waseda/comet-t5-base-japanese"
        self.second_finetune = True
        self.pretrained_model: str = "../outputs/cae/sonoisa__t5-base-japanese/2024-05-30/16-41-02"
        #self.pretrained_model: str = "sonoisa/t5-base-japanese"
        # self.pretrained_model: str = "../outputs/cae/megagonlabs__t5-base-japanese-web/2023-02-26/22-05-20"
        # self.pretrained_model: str = "../outputs/cae/nlp-waseda__comet-t5-base-japanese/2023-11-16/15-00-59"
        self.dataset_dir: Path = Path("../data")
        # self.model_save_dir: str = "/Storage/yuki/Research/models"

        self.batch_size: int = 32
        #100に直すこと
        self.epochs: int = 200
        self.lr: float = 1e-4
        self.num_warmup_epochs: int = 10
        self.max_seq_len: int = 512
        self.nn_model_epochs: int = 50

        self.max_length: int = 100
        self.num_beams: int = 4 # 予測時
        # self.repetition_penalty: float = 2.5
        # self.length_penalty: int = 1

        self.patience = 10
        self.date = datetime.now().strftime("%Y-%m-%d/%H-%M-%S")
        self.device: str =  "cuda:0"  
        self.seed: int = 42
        utils.set_seed(self.seed) #乱数シードを固定

        model_name = self.model_name.replace("/", "__")
        self.output_dir = Path("../outputs") / model_name / self.date #保存先の指定
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir2 = Path("../outputs") / "nn_model" / self.date
        self.output_dir2.mkdir(parents=True, exist_ok=True)


  nn1 = 64
    nn2 = 32
    nn_model = Sequential()
    nn_model.add(Dense(nn1, activation='relu', input_dim=8))
    nn_model.add(Dense(nn2, activation='relu'))
    nn_model.add(Dense(2, activation='linear'))
    nn_model.compile(
        optimizer='rmsprop',
        loss='mean_squared_error',
        metrics=['accuracy']
    )


train_history = nn_model.fit(
            seikai,
            gtruth,
            batch_size=20,
            epochs=args.nn_model_epochs,
            verbose=1)
        


        weightspeed = np.array(val_metrics[2]) #質量・スピード

        gtruth = np.array(val_metrics[3]) #正解ラベル
        seikai = weightspeed
