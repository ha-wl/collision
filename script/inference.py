from pathlib import Path
from datetime import datetime
import random
import numpy as np
from nltk import bleu_score
from sumeval.metrics.rouge import RougeCalculator

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

        self.pretrained_model: str = "../outputs/sonoisa__t5-base-japanese/2024-12-10/21-42-01"
        # self.pretrained_model: str = "../outputs/megagonlabs__t5-base-japanese-web/2023-07-16/18-28-05"
        # self.pretrained_model: str = "../outputs/nlp-waseda__comet-t5-base-japanese/2023-11-18/10-00-12"
        # self.pretrained_model: str = "/Storage/yuki/Research/models"
        self.dataset_dir: Path = Path("../data")

        self.batch_size: int = 32
        self.epochs: int = 5
        self.lr: float = 1e-5
        self.num_warmup_epochs: int = 5
        self.max_seq_len: int = 512

        self.max_length: int = 100
        # 予測時
        self.num_beams: int = 4 
        # self.repetition_penalty: float = 2.5
        # self.length_penalty: int = 1

        self.date = datetime.now().strftime("%Y-%m-%d/%H-%M-%S")
        self.device: str =  "cuda:1"
        self.seed: int = 42
        utils.set_seed(self.seed)

        model_name = self.model_name.replace("/", "__")
        self.output_dir = Path("../gen_outputs") / model_name / self.date
        self.output_dir.mkdir(parents=True, exist_ok=True)


def load_dataset(path: Path) -> list:
    return utils.load_jsonl(path).to_dict(orient="records")


def main(args: Args):
    tokenizer: PreTrainedTokenizer = T5Tokenizer.from_pretrained(
        args.model_name,
        model_max_length=args.max_seq_len,
    )
    model: PreTrainedModel = (
        T5ForConditionalGeneration.from_pretrained(
        args.pretrained_model,
    )
    .eval()
    .to(args.device, non_blocking=True)
    )


    val_dataset: list[dict] = load_dataset(args.dataset_dir / "val8.jsonl")
    test_dataset: list[dict] = load_dataset(args.dataset_dir / "test8-1.jsonl")
    plain_dataset: list[dict] = load_dataset(args.dataset_dir / "plain8-1.jsonl")

    def collate_fn(dataset: list) -> BatchEncoding:
        ids = [d["ID"] for d in dataset]
        patterns = [d["pattern"] for d in dataset]
        input = [d["input"] for d in dataset]
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
        return ids, patterns, inputs, labels



    def create_loader(dataset, batch_size=None, shuffle=False):
        return DataLoader(
            dataset,
            collate_fn=collate_fn,
            batch_size=batch_size or args.batch_size,
            shuffle=shuffle,
            num_workers=0,
            pin_memory=True,
        )

    val_dataloader: DataLoader = create_loader(val_dataset, shuffle=False)
    test_dataloader: DataLoader = create_loader(test_dataset, shuffle=False)
    plain_dataloader: DataLoader = create_loader(plain_dataset, shuffle=False)

    @torch.no_grad()
    def evaluate(dataloader: DataLoader):
        model.eval()
        # loss = 0
        gold_labels, pred_labels, id_list, pattern_list = [], [], [], []
        output_list = []
        fn = bleu_score.SmoothingFunction().method1
        bleu_scores = 0
        rouge2_scores = 0
        rougel_scores = 0
        data_size = 0

        scorer = RougeCalculator(stopwords=True, lang="ja") # tokenizeしないように変更している
        # /raid/yuki/.local/lib/python3.6/site-packages/sumeval/metrics/rouge.py 
        for batch in tqdm(dataloader, total=len(dataloader), dynamic_ncols=True, leave=False):
            ids, patterns, input_ids, outputs = batch

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
                bleu = bleu_score.sentence_bleu(ref, o, smoothing_function=fn)
                rouge_2 = scorer.rouge_n(summary=o, references=ref, n=2)
                rouge_l = scorer.rouge_l(summary=o, references=ref)

                if bleu > 1:
                    print(bleu)
                bleu_scores += bleu
                rouge2_scores += rouge_2
                rougel_scores += rouge_l

            # Lossではない！ BLUEスコアで検証する！
            data_size += input_ids["input_ids"].size(0)
            pred_labels += out.tolist()
            id_list += ids
            pattern_list += patterns

        preds = [tokenizer.decode(p, skip_special_tokens=True, clean_up_tokenization_spaces=True) for p in pred_labels]

        for i in range(len(id_list)):
            output_list.append({"bleu": bleu_scores/data_size, "ROUGE-2":rouge2_scores/data_size, "ROUGE-L":rougel_scores/data_size,
                                "ID":id_list[i], "pattern":pattern_list[i], "pred":preds[i]}) #"gold": golds[:, i]
        
        return output_list

    def log(metrics: dict) -> None:
        utils.log(metrics, args.output_dir / "log.csv")
        tqdm.write(
            f"epoch: {metrics['epoch']}")


    val_metrics =  evaluate(val_dataloader)
    utils.save_jsonl(val_metrics, args.output_dir / "val-metrics.jsonl")
        
    test_metrics = evaluate(test_dataloader)
    utils.save_jsonl(test_metrics, args.output_dir / "test-metrics.jsonl")

    plain_metrics = evaluate(plain_dataloader)
    utils.save_jsonl(test_metrics, args.output_dir / "plain-metrics.jsonl")

    utils.save_config(args, args.output_dir / "config.json")

if __name__=="__main__":
    args = Args()
    main(args)
