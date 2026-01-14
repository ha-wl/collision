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
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
import embedding


class Args:
    def __init__(self): #学習データの初期化
        self.model_name: str = "sonoisa/t5-base-japanese" 
        # self.model_name: str = "megagonlabs/t5-base-japanese-web"
        # self.model_name: str = "nlp-waseda/comet-t5-base-japanese"
        self.second_finetune = True
        self.pretrained_model: str = "../outputs/cae/sonoisa__t5-base-japanese/2024-05-30/16-41-02"
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


def load_dataset(path: Path) -> list:
    return utils.load_jsonl(path).to_dict(orient="records")


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

    train_dataset: list[dict] = load_dataset(args.dataset_dir / "train8.jsonl")
    val_dataset: list[dict] = load_dataset(args.dataset_dir / "val8.jsonl")
    test_dataset: list[dict] = load_dataset(args.dataset_dir / "test8.jsonl")
    em_model = embedding.SentenceT5(args.pretrained_model, args.model_name)

    def collate_fn(dataset: list) -> BatchEncoding:
        ids = [d["ID"] for d in dataset]
        patterns = [d["pattern"] for d in dataset]
        input = [d["input"] for d in dataset]
        distributions = [d["distribution"] for d in dataset]
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
        return ids, patterns, inputs, labels, distributions

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
        gold_labels, pred_labels, id_list, pattern_list, distribution_list = [], [], [], [], []
        output_list = []
        prediction_list = []
        fn = bleu_score.SmoothingFunction().method1
        bleu_scores = 0
        data_size = 0
        
        for batch in tqdm(dataloader, total=len(dataloader), dynamic_ncols=True, leave=False):
            # max_length, num_beams引数確認
            ids, patterns, input_ids, outputs, distributions = batch

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
            distribution_list += distributions

        preds = [tokenizer.decode(p, skip_special_tokens=True, clean_up_tokenization_spaces=True) for p in pred_labels]

        ##predを入れて
        for i in range(len(id_list)):
            output_list.append({"bleu": bleu_scores/data_size, "ID":id_list[i], "pattern":pattern_list[i], "pred":preds[i]}) #"gold": golds[:, i]
            prediction_list.append(preds[i])

        return output_list, prediction_list, distribution_list

    
    @torch.no_grad()
    def evaluate2(dataloader: DataLoader, nn_model):
        model.eval()
        gold_labels, pred_labels, id_list, pattern_list, distribution_list = [], [], [], [], []
        output_list = []
        nn_dist1 = []
        nn_dist2 = []
        fn = bleu_score.SmoothingFunction().method1
        bleu_scores = 0
        data_size = 0
        
        for batch in tqdm(dataloader, total=len(dataloader), dynamic_ncols=True, leave=False):
            # max_length, num_beams引数確認
            ids, patterns, input_ids, outputs, distributions = batch

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
            distribution_list += distributions

        preds = [tokenizer.decode(p, skip_special_tokens=True, clean_up_tokenization_spaces=True) for p in pred_labels]
        sentence_embeddings = em_model.encode(preds, batch_size=8)
        se = sentence_embeddings.clone().detach()
        a = np.array(se)
        nn_pred = nn_model.predict(a)


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
  

    for epoch in trange(args.epochs, dynamic_ncols=True):
        model.eval()
        val_metrics = evaluate(val_dataloader)
        #em_model = embedding.SentenceT5(args.pretrained_model, args.model_name, args.device)
        
        sentence_embeddings = em_model.encode(val_metrics[1], batch_size=8)
        se = sentence_embeddings.clone().detach()
        a = np.array(se)
        #a = se.to('cpu').detach().numpy().copy()
        #a = a.to(device)
        gtruth = np.array(val_metrics[2])
        loss2 = 0
        train_history = nn_model.fit(
            a,
            gtruth,
            batch_size=20,
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
            ids, pattern, input_ids, outputs, distributions = batch

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
