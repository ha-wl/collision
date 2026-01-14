from pathlib import Path
from datetime import datetime
import random
import numpy as np
import nltk
from nltk import bleu_score

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm, trange
from transformers import (
    T5Model,
    T5Tokenizer,
    T5ForConditionalGeneration,
    PreTrainedModel,
)
from transformers.optimization import get_linear_schedule_with_warmup
from transformers.tokenization_utils import PreTrainedTokenizer, BatchEncoding

import utils as utils

import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

import embedding
import tensorflow as tf
import keras
from tensorflow import keras
from tensorflow.keras import layers
import math
from scipy.stats import norm

from torch.distributions import Normal


class Args:
    def __init__(self): #学習データの初期化
        self.model_name: str = "sonoisa/t5-base-japanese" 
        # self.model_name: str = "megagonlabs/t5-base-japanese-web"
        # self.model_name: str = "nlp-waseda/comet-t5-base-japanese"
        self.second_finetune = False
        self.pretrained_model: str = "../outputs/cae/sonoisa__t5-base-japanese/2024-05-30/16-41-02"
        #self.pretrained_model: str = "sonoisa/t5-base-japanese"
        # self.pretrained_model: str = "../outputs/cae/megagonlabs__t5-base-japanese-web/2023-02-26/22-05-20"
        # self.pretrained_model: str = "../outputs/cae/nlp-waseda__comet-t5-base-japanese/2023-11-16/15-00-59"
        self.dataset_dir: Path = Path("../data")
        # self.model_save_dir: str = "/Storage/yuki/Research/models"

        self.batch_size: int = 32
        #100に直すこと
        self.epochs: int = 100
        self.lr: float = 1e-5
        self.num_warmup_epochs: int = 10
        self.max_seq_len: int = 512
        self.nn_model_epochs: int = 50
        self.nn_learning_rate: int = 1e-6

        self.max_length: int = 100
        self.num_beams: int = 4 # 予測時
        # self.repetition_penalty: float = 2.5
        # self.length_penalty: int = 1

        self.patience = 10
        self.date = datetime.now().strftime("%Y-%m-%d/%H-%M-%S")
        self.device: str =  "cuda:3"  
        self.seed: int = 42
        utils.set_seed(self.seed) #乱数シードを固定

        model_name = self.model_name.replace("/", "__")
        self.output_dir = Path("../outputs") / model_name / self.date #保存先の指定
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir2 = Path("../outputs") / "nn_model" / self.date
        self.output_dir2.mkdir(parents=True, exist_ok=True)


def load_dataset(path: Path) -> list:
    return utils.load_jsonl(path).to_dict(orient="records")

def custom_loss(y_pred, target, taisho):
  #taisho 0は等しい 1はy_predのほうが大きい 2はtargetの方が大きい
  a=y_pred[0,0]-target[0,0]
  b=target[0,0]-y_pred[0,0]
  c=y_pred[0,1]**2+target[0,1]**2
  d = tf.math.sqrt(2.0*c)
  e = tf.math.erf(-a/d)#小さくなってほしい
  f = tf.math.erf(-b/d) #小さくなってほしい

  i = tf.clip_by_value(1.0-((1.0+e)/2.0),clip_value_max=1e+5,clip_value_min=1e-5)
  j = tf.clip_by_value(1.0-((1.0+f)/2.0),clip_value_max=1e+5,clip_value_min=1e-5)
  """
  d = 1+tf.math.erf(-a/tf.math.sqrt(2*c))
  d2 = 1+tf.math.erf(-a/tf.math.sqrt(2*c))
  e = 1+tf.math.erf(-b/tf.math.sqrt(2*c))
  e2 = 1+tf.math.erf(-b/tf.math.sqrt(2*c))
  """
  g = tf.clip_by_value(1.5*tf.math.exp(-1.5*a**2),clip_value_max=1e+5,clip_value_min=1e-5)
  h = tf.clip_by_value(1.0/g,clip_value_max=1e+5,clip_value_min=1e-5)
  #g = 1.0/(1.0 - d/2)
  #h = 1.0/(1.0 - e/2)
  
  def f0(): return tf.clip_by_value(1.0/i,clip_value_max=1e+5,clip_value_min=1e-5) #小さくなるためにはeが小さくなってほしい
  def f1(): return tf.clip_by_value(1.0/j,clip_value_max=1e+5,clip_value_min=1e-5)
  def f2(): return h
  def f3(): return tf.cond(taisho[0,0]==1,f0,f1)
  def f4(): return tf.cond(taisho[0,0]==0,f2,f2)

  #def f4(): return tf.cond(taisho[0,0]==0,f0,f0))

  loss = tf.cond(taisho[0,0]==0,f4,f3)
  loss2 = tf.clip_by_value(tf.math.exp(loss),clip_value_max=1e+5,clip_value_min=1e-5)

  return loss

class ModelCustomLoss(keras.Model):
  def __init__(self):
    super(ModelCustomLoss, self).__init__()
    self.layer_dense_1 = layers.Dense(776)
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
    self.add_loss(custom_loss(out, dif, tai))

    return out

def main(args: Args):
    tokenizer: PreTrainedTokenizer = T5Tokenizer.from_pretrained(
        args.model_name,
        model_max_length=args.max_seq_len,
    )
    if args.second_finetune:
        print("Yes")
        model: PreTrainedModel = (
            T5ForConditionalGeneration.from_pretrained(
            args.pretrained_model,
        )
        .eval()
        .to(args.device, non_blocking=True)
        )
    else:
        model: PreTrainedModel = (
            T5ForConditionalGeneration.from_pretrained(
            args.model_name,
        )
        .eval()
        .to(args.device, non_blocking=True)
        )

    train_dataset: list[dict] = load_dataset(args.dataset_dir / "train7.jsonl")
    val_dataset: list[dict] = load_dataset(args.dataset_dir / "val7.jsonl")
    test_dataset: list[dict] = load_dataset(args.dataset_dir / "test7.jsonl")
    em_model = embedding.SentenceT5(args.pretrained_model, args.model_name)

    def collate_fn(dataset: list) -> BatchEncoding:
        ids = [d["ID"] for d in dataset]
        patterns = [d["pattern"] for d in dataset]
        input = [d["input"] for d in dataset]
        weightspeeds = [d["weightspeed"] for d in dataset]
        distributions = [d["distribution"] for d in dataset]
        targets = [d["target"] for d in dataset]
        label2s = [d["label"] for d in dataset]
        # 正解データをランダムに選択
        # output = [d["output"][random.randint(0, 4)] for d in dataset]

        inputs: BatchEncoding = tokenizer(
            input,
            padding=True,
            return_tensors="pt",
            max_length=args.max_seq_len,       
        )
        labels = []
        for i in range(5):
            outputs = [d["output"][i] for d in dataset]
            labels.append(tokenizer(
                outputs,
                padding=True,
                return_tensors="pt",
                max_length=args.max_seq_len,
            ))
        return ids, patterns, inputs, labels, weightspeeds, distributions, targets, label2s

    def create_loader(dataset, batch_size=None, shuffle=False):
        return DataLoader(
            dataset,
            collate_fn=collate_fn,
            batch_size=batch_size or args.batch_size,
            shuffle=shuffle,
            num_workers=0,
            pin_memory=True,
        )

    train_dataloader: DataLoader = create_loader(train_dataset, shuffle=True)
    val_dataloader: DataLoader = create_loader(val_dataset, shuffle=False)
    test_dataloader: DataLoader = create_loader(test_dataset, shuffle=False)

    optimizer = torch.optim.AdamW(params=model.parameters(), lr=args.lr)

    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=len(train_dataloader) * args.num_warmup_epochs,
        num_training_steps=len(train_dataloader) * args.epochs,
    )

    def clone_state_dict() -> dict:
        return { k: v.detach().clone().cpu() for k, v in model.state_dict().items()}
    
    @torch.no_grad()
    def evaluate(dataloader: DataLoader):
        model.eval()
        gold_labels, pred_labels, id_list, pattern_list, weightspeed_list, distribution_list, target_list, label2_list = [], [], [], [], [], [], [], []
        output_list = []
        prediction_list = []
        fn = bleu_score.SmoothingFunction().method1
        bleu_scores = 0
        data_size = 0
        
        for batch in tqdm(dataloader, total=len(dataloader), dynamic_ncols=True, leave=False):
            # max_length, num_beams引数確認
            ids, patterns, input_ids, outputs, weightspeeds, distributions, targets, label2s = batch

            # out = model(input_ids=input_ids["input_ids"].to(args.device), attention_mask=input_ids["attention_mask"].to(args.device), labels=labels["input_ids"].to(args.device))
            out = model.generate(input_ids["input_ids"].to(args.device), max_length=args.max_seq_len, num_beams=args.num_beams)
            for i, o in enumerate(out):
                mask_o = o.ne(0)
                o = torch.masked_select(o, mask_o).cpu().tolist()
                ref = []
                for j in range(5):
                    labels = outputs[j]["input_ids"]
                    l = labels[i]
                    mask_l = l.ne(0)
                    l = torch.masked_select(l, mask_l).tolist()
                    ref.append(l)
                scores = bleu_score.sentence_bleu(ref, o, smoothing_function=fn)
                bleu_scores += scores

            # Lossではない！ BLUEスコアで検証する！
            data_size += input_ids["input_ids"].size(0)
            # loss += out.loss.item() * batch_size
            # pred_labels += out.logits.argmax(dim=-1).tolist()
            pred_labels += out.tolist()
            golds = []
            id_list += ids
            pattern_list += patterns
            weightspeed_list += weightspeeds
            distribution_list += distributions
            target_list += targets
            label2_list += label2s

        preds = [tokenizer.decode(p, skip_special_tokens=True, clean_up_tokenization_spaces=True) for p in pred_labels]

        ##predを入れて
        for i in range(len(id_list)):
            output_list.append({"bleu": bleu_scores/data_size, "ID":id_list[i], "pattern":pattern_list[i], "pred":preds[i]}) #"gold": golds[:, i]
            prediction_list.append(preds[i])

        return output_list, prediction_list, weightspeed_list, distribution_list, target_list, label2_list

    
    @torch.no_grad()
    def evaluate2(dataloader: DataLoader, nn_model):
        model.eval()
        gold_labels, pred_labels, id_list, pattern_list, weightspeed_list, distribution_list, target_list, label2_list = [], [], [], [], [], [], [], []
        output_list = []
        nn_dist1 = []
        nn_dist2 = []
        fn = bleu_score.SmoothingFunction().method1
        bleu_scores = 0
        data_size = 0
        
        for batch in tqdm(dataloader, total=len(dataloader), dynamic_ncols=True, leave=False):
            # max_length, num_beams引数確認
            ids, patterns, input_ids, outputs, weightspeeds, distributions, targets, label2s = batch

            # out = model(input_ids=input_ids["input_ids"].to(args.device), attention_mask=input_ids["attention_mask"].to(args.device), labels=labels["input_ids"].to(args.device))
            out = model.generate(input_ids["input_ids"].to(args.device), max_length=args.max_seq_len, num_beams=args.num_beams)
            for i, o in enumerate(out):
                mask_o = o.ne(0)
                o = torch.masked_select(o, mask_o).cpu().tolist()
                ref = []
                for j in range(5):
                    labels = outputs[j]["input_ids"]
                    l = labels[i]
                    mask_l = l.ne(0)
                    l = torch.masked_select(l, mask_l).tolist()
                    ref.append(l)
                scores = bleu_score.sentence_bleu(ref, o, smoothing_function=fn)
                bleu_scores += scores

            # Lossではない！ BLUEスコアで検証する！
            data_size += input_ids["input_ids"].size(0)
            # loss += out.loss.item() * batch_size
            # pred_labels += out.logits.argmax(dim=-1).tolist()
            pred_labels += out.tolist()
            golds = []
            id_list += ids
            pattern_list += patterns
            weightspeed_list += weightspeeds
            distribution_list += distributions
            target_list += targets
            label2_list += label2s

        preds = [tokenizer.decode(p, skip_special_tokens=True, clean_up_tokenization_spaces=True) for p in pred_labels]
        sentence_embeddings = em_model.encode(preds, batch_size=8)
        se = sentence_embeddings.clone().detach()
        a = np.array(se) #768次元のベクトル
        ws = np.array(weightspeed_list) #質量とスピード
        seikai = np.concatenate([ws, a], 1)

        tar = np.array(target_list) 
        target = tar.reshape([tar.shape[0],2])
        lab = np.array(label2_list) 
        label2 = lab.reshape([lab.shape[0],1])

        nn_pred = nn_model.predict([seikai,target,label2])


        ##predを入れて
        for i in range(len(id_list)):
            output_list.append({"bleu": bleu_scores/data_size, "ID":id_list[i], "pattern":pattern_list[i], "pred":preds[i], "dist_pred":nn_pred[i], "dist_lst":distribution_list[i]}) #"gold": golds[:, i]
            nn_dist1.append(nn_pred[i][0])
            nn_dist2.append(nn_pred[i][1])

        return output_list, preds, distribution_list

    def log(metrics: dict) -> None:
        utils.log(metrics, args.output_dir / "log.csv")
        tqdm.write(
            f"epoch: {metrics['epoch']}")# \t loss: {metrics['loss']:2.6f} \t"

    # val_metrics = {"epoch": None, "loss": None, "perlexity": None}#, **evaluate(val_dataloader)}
    patience_counter = 0
    best_score = 0
    best_state_dict = clone_state_dict()
    # log(val_metrics)
    loss2_lst = []
    """nn1 = 64
    nn2 = 32
    nn_model = Sequential()
    nn_model.add(Dense(nn1, activation='relu', input_dim=776))
    nn_model.add(Dense(nn2, activation='relu'))
    nn_model.add(Dense(2, activation='linear'))
    nn_model.compile(
        optimizer='rmsprop',
        loss='mean_squared_error',
        metrics=['accuracy']
    )"""
    nn_model = ModelCustomLoss()
    opt = keras.optimizers.Adam(learning_rate=args.nn_learning_rate)
    nn_model.compile(
        optimizer = opt,
        #optimizer='rmsprop',
        metrics=['accuracy'],
    )
  

    for epoch in trange(args.epochs, dynamic_ncols=True):
        model.eval()
        val_metrics = evaluate(val_dataloader)
        #em_model = embedding.SentenceT5(args.pretrained_model, args.model_name, args.device)
        
        sentence_embeddings = em_model.encode(val_metrics[1], batch_size=8)
        se = sentence_embeddings.clone().detach()
        a = np.array(se) #入力する768次元のベクトル
        weightspeed = np.array(val_metrics[2]) #質量・スピード

        gtruth = np.array(val_metrics[3]) #正解ラベル
        seikai = np.concatenate([weightspeed,a], 1)

        tar = np.array(val_metrics[4]) 
        target = tar.reshape([tar.shape[0],2])
        lab = np.array(val_metrics[5]) 
        label2 = lab.reshape([lab.shape[0],1])

        loss2 = 0
        train_history = nn_model.fit(
            [seikai,target,label2],
            gtruth,
            batch_size=32,
            epochs=args.nn_model_epochs,
            verbose=1)
        
        loss2 = sum(train_history.history['loss'])
        loss2 /= args.nn_model_epochs
        loss2_lst.append(loss2)

        model.train()
        loss_for_log = 0

        data_size = 0
        for batch in tqdm(
            train_dataloader,
            total=len(train_dataloader),
            dynamic_ncols=True,
            leave=False,
        ):
            ids, pattern, input_ids, outputs, weightspeeds, distributions, targets, label2s = batch

            labels = outputs[random.randint(0, 4)]
            out = model(input_ids=input_ids["input_ids"].to(args.device), attention_mask=input_ids["attention_mask"].to(args.device), labels=labels["input_ids"].to(args.device))
            loss: torch.FloatTensor = out.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            batch_size: int = input_ids["input_ids"].size(0)
            loss_for_log += loss.item() * batch_size

            data_size += batch_size
        # loss_for_log = loss_for_log.cpu()
        model.eval()
        # train_metrics = evaluate(train_dataloader)
        # val_metrics = evaluate(val_dataloader)
        utils.save_jsonl(val_metrics[0], args.output_dir / "val-metrics.jsonl")
        log({"epoch": epoch, "train_loss": loss_for_log/data_size, "train_perplexity":np.exp(loss_for_log/data_size), \
             #train_metrics[0]["loss"], "train_perplexity":np.exp(train_metrics[0]["loss"]), \
            "val_bleu": val_metrics[0][0]["bleu"], "loss2": loss2})#, "val_perlexity": np.exp(val_metrics[0]["loss"])})

        if val_metrics[0][0]["bleu"] > best_score:
            best_score = val_metrics[0][0]["bleu"]
            best_epoch = epoch
            best_state_dict = clone_state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            # if patience_counter > args.patience:
                # break
        # if epoch % 200 == 0:
        #     model.save_pretrained(f"{args.model_save_dir}/{epoch}")

    model.load_state_dict(best_state_dict)
    model.eval().to(args.device, non_blocking=True)

    nn_model.save(args.output_dir2)

    scr = evaluate2(val_dataloader, nn_model)

    val_metrics = [{"best-epoch": best_epoch}] + scr[0]
    utils.save_jsonl(val_metrics, args.output_dir / "val-metrics.jsonl")
        
    test_metrics = evaluate2(test_dataloader, nn_model)#, mode="test")
    utils.save_jsonl(test_metrics[0], args.output_dir / "test-metrics.jsonl")

    utils.save_config(args, args.output_dir / "config.json")

    model.save_pretrained(args.output_dir)

    

if __name__=="__main__":
    args = Args()
    main(args)
