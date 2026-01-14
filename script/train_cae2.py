from pathlib import Path
from datetime import datetime
import random
import numpy as np
from nltk import bleu_score

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm, trange
from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    PreTrainedModel,
)
from transformers.optimization import get_linear_schedule_with_warmup
from transformers.tokenization_utils import PreTrainedTokenizer, BatchEncoding

import utils as utils


class Args:
    def __init__(self):
        self.model_name: str = "sonoisa/t5-base-japanese" 
        # self.model_name: str = "megagonlabs/t5-base-japanese-web"
        # self.model_name: str = "nlp-waseda/comet-t5-base-japanese"
        self.second_finetune = True
        self.pretrained_model: str = "../outputs/cae/sonoisa__t5-base-japanese/2024-05-30/16-28-38"
        self.dataset_dir: Path = Path("../data/cae")
        # self.model_save_dir: str = "/Storage/yuki/Research/models"

        self.batch_size: int = 32
        self.epochs: int = 100
        self.lr: float = 5e-5
        self.num_warmup_epochs: int = 10
        self.max_seq_len: int = 512

        self.max_length: int = 100
        self.num_beams: int = 4 # 予測時
        # self.repetition_penalty: float = 2.5
        # self.length_penalty: int = 1

        self.patience = 10
        self.date = datetime.now().strftime("%Y-%m-%d/%H-%M-%S")
        self.device: str =  "cuda:0"
        self.seed: int = 42
        utils.set_seed(self.seed)

        model_name = self.model_name.replace("/", "__")
        self.output_dir = Path("../outputs/cae") / model_name / self.date
        self.output_dir.mkdir(parents=True, exist_ok=True)


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

    train_dataset: list[dict] = load_dataset(args.dataset_dir / "train2-1.jsonl")
    val_dataset: list[dict] = load_dataset(args.dataset_dir / "val2-1.jsonl")
    test_dataset: list[dict] = load_dataset(args.dataset_dir / "test2-1.jsonl")

    def collate_fn(dataset: list) -> BatchEncoding:
        input = [d["input"] for d in dataset]
        output = [d["output"] for d in dataset]

        inputs: BatchEncoding = tokenizer(
            input,
            padding=True,
            return_tensors="pt",
            max_length=args.max_seq_len,       
        )
        labels: BatchEncoding = tokenizer(
            output,
            padding=True,
            return_tensors="pt",
            max_length=args.max_seq_len,
        )
        return inputs, labels


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
        gold_labels, pred_labels, input_labels = [], [], [],
        output_list = []
        fn = bleu_score.SmoothingFunction().method1
        bleu_scores = 0
        data_size = 0
        for batch in tqdm(dataloader, total=len(dataloader), dynamic_ncols=True, leave=False):
            # max_length, num_beams引数確認
            input_ids, outputs = batch

            # out = model(input_ids=input_ids["input_ids"].to(args.device), attention_mask=input_ids["attention_mask"].to(args.device), labels=labels["input_ids"].to(args.device))
            out = model.generate(input_ids["input_ids"].to(args.device), max_length=args.max_seq_len, num_beams=args.num_beams)
            for i, o in enumerate(out):
                mask_o = o.ne(0)
                o = torch.masked_select(o, mask_o).cpu().tolist()
                labels = outputs["input_ids"]
                l = labels[i]
                mask_l = l.ne(0)
                l = torch.masked_select(l, mask_l).tolist()
                ref = [l]
                scores = bleu_score.sentence_bleu(ref, o, smoothing_function=fn)
                bleu_scores += scores

            # Lossではない！ BLUEスコアで検証する！
            data_size += input_ids["input_ids"].size(0)
            pred_labels += out.tolist()
            gold_labels += outputs["input_ids"].tolist()
            input_labels += input_ids["input_ids"].tolist()

        inputs = [tokenizer.decode(i, skip_special_tokens=True, clean_up_tokenization_spaces=True) for i in input_labels]
        golds = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in gold_labels]
        preds = [tokenizer.decode(p, skip_special_tokens=True, clean_up_tokenization_spaces=True) for p in pred_labels]

        for i in range(len(pred_labels)):
            output_list.append({"bleu": bleu_scores/data_size, "input": inputs[i], "gold": golds[i], "pred":preds[i]})
        return output_list

    def log(metrics: dict) -> None:
        utils.log(metrics, args.output_dir / "log.csv")
        tqdm.write(
            f"epoch: {metrics['epoch']}")# \t loss: {metrics['loss']:2.6f} \t"

    # val_metrics = {"epoch": None, "loss": None, "perlexity": None}#, **evaluate(val_dataloader)}
    patience_counter = 0
    best_bleu = 0
    best_state_dict = clone_state_dict()
    # log(val_metrics)

    for epoch in trange(args.epochs, dynamic_ncols=True):
        model.train()
        loss_for_log = 0
        data_size = 0
        for batch in tqdm(
            train_dataloader,
            total=len(train_dataloader),
            dynamic_ncols=True,
            leave=False,
        ):
            input_ids, labels = batch
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
        val_metrics = evaluate(val_dataloader)
        utils.save_jsonl(val_metrics, args.output_dir / "val-metrics.jsonl")
        log({"epoch": epoch, "train_loss": loss_for_log/data_size, "train_perplexity":np.exp(loss_for_log/data_size), \
             #train_metrics[0]["loss"], "train_perplexity":np.exp(train_metrics[0]["loss"]), \
            "val_bleu": val_metrics[0]["bleu"]})#, "val_perlexity": np.exp(val_metrics[0]["loss"])})


        if val_metrics[0]["bleu"] > best_bleu:
            best_bleu = val_metrics[0]["bleu"]
            best_epoch = epoch
            best_state_dict = clone_state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            # if patience_counter > args.patience:
            #     break
        # if epoch % 200 == 0:
        #     model.save_pretrained(f"{args.model_save_dir}/{epoch}")

    model.load_state_dict(best_state_dict)
    model.eval().to(args.device, non_blocking=True)

    val_metrics = [{"best-epoch": best_epoch}] + evaluate(val_dataloader)
    utils.save_jsonl(val_metrics, args.output_dir / "val-metrics.jsonl")
        
    test_metrics = evaluate(test_dataloader)#, mode="test")
    utils.save_jsonl(test_metrics, args.output_dir / "test-metrics.jsonl")

    utils.save_config(args, args.output_dir / "config.json")

    model.save_pretrained(args.output_dir)

if __name__=="__main__":
    args = Args()
    main(args)
