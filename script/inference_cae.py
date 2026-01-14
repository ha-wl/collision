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
        # self.pretrained_model: str = "../outputs/cae/sonoisa__t5-base-japanese/2023-02-20/23-10-04"
        self.pretrained_model: str = "../outputs/cae/sonoisa__t5-base-japanese/2024-05-30/21-44-58"
        # self.pretrained_model: str = "../outputs/cae/megagonlabs__t5-base-japanese-web/2023-02-26/22-05-20"
        # self.pretrained_model: str = "../outputs/cae/nlp-waseda__comet-t5-base-japanese/2023-02-26/22-05-58"
        # self.pretrained_model: str = "/Storage/yuki/Research/models/400"
        self.dataset_dir: Path = Path("../data/cae")

        self.batch_size: int = 32
        self.epochs: int = 5
        self.lr: float = 1e-5
        self.num_warmup_epochs: int = 5
        self.max_seq_len: int = 512

        self.max_length: int = 100
        # 予測時
        self.num_beams: int = 4
        self.repetition_penalty: float = 2.5
        self.length_penalty: int = 1

        self.date = datetime.now().strftime("%Y-%m-%d/%H-%M-%S")
        self.device: str =  "cuda:0"
        self.seed: int = 42
        utils.set_seed(self.seed)

        model_name = self.model_name.replace("/", "__")
        self.output_dir = Path("../gen_outputs/cae") / model_name / self.date
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

    val_dataset: list[dict] = load_dataset(args.dataset_dir / "val3.jsonl")
    test_dataset: list[dict] = load_dataset(args.dataset_dir / "test3.jsonl")


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

    val_dataloader: DataLoader = create_loader(val_dataset, shuffle=False)
    test_dataloader: DataLoader = create_loader(test_dataset, shuffle=False)
    @torch.no_grad()
    def evaluate(dataloader: DataLoader):
        model.eval()
        gold_labels, pred_labels, input_labels = [], [], [],
        output_list = []
        fn = bleu_score.SmoothingFunction().method1
        bleu_scores = 0
        rouge2_scores = 0
        rougel_scores = 0
        data_size = 0

        scorer = RougeCalculator(stopwords=True, lang="ja") # tokenizeしないように変更している
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
                rouge_2 = scorer.rouge_n(summary=o, references=ref, n=2)
                rouge_l = scorer.rouge_l(summary=o, references=ref)
                bleu_scores += scores
                rouge2_scores += rouge_2
                rougel_scores += rouge_l

            # Lossではない！ BLUEスコアで検証する！
            data_size += input_ids["input_ids"].size(0)
            pred_labels += out.tolist()
            gold_labels += outputs["input_ids"].tolist()
            input_labels += input_ids["input_ids"].tolist()

        inputs = [tokenizer.decode(i, skip_special_tokens=True, clean_up_tokenization_spaces=True) for i in input_labels]
        golds = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in gold_labels]
        preds = [tokenizer.decode(p, skip_special_tokens=True, clean_up_tokenization_spaces=True) for p in pred_labels]

        for i in range(len(pred_labels)):
            output_list.append({"bleu": bleu_scores/data_size, "ROUGE-2":rouge2_scores/data_size, "ROUGE-L":rougel_scores/data_size, "input":inputs[i],"pred":preds[i] ,"gold": golds[i]})
        
        return output_list
    """
    @torch.no_grad()
    def evaluate(dataloader: DataLoader):
        model.eval()
        loss = 0
        gold_labels, pred_labels, input_list, pattern_list = [], [], [], []
        output_list = []
        beam_outputs = []

        for batch in tqdm(dataloader, total=len(dataloader), dynamic_ncols=True, leave=False):
            # max_length, num_beams引数確認
            input_ids, labels = batch
   
            # out = model(input_ids=input_ids["input_ids"].to(args.device), attention_mask=input_ids["attention_mask"].to(args.device), labels=labels["input_ids"].to(args.device))
            batch_size: int = input_ids["input_ids"].size(0)
            # loss += out.loss.item() * batch_size
            # pred_labels += out.logits.argmax(dim=-1).tolist()
            gold_labels += labels["input_ids"].tolist()
            gen = model.generate(input_ids["input_ids"].to(args.device), max_length=args.max_seq_len, num_beams=4, repetition_penalty=args.repetition_penalty, length_penalty=args.length_penalty)
            beam_outputs += gen.tolist()
            input_list += input_ids["input_ids"].tolist()

        inputs_a =  [tokenizer.decode(inp, skip_special_tokens=True, clean_up_tokenization_spaces=True) for inp in input_list]
        golds = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in gold_labels]
        preds = [tokenizer.decode(p, skip_special_tokens=True, clean_up_tokenization_spaces=True) for p in pred_labels]
        gens = [tokenizer.decode(b, skip_special_tokens=True, clean_up_tokenization_spaces=True) for b in beam_outputs]
        for i in range(len(golds)):
            output_list.append({"input":inputs_a[i], "gold": golds[i], "pred":preds[i], "gen":gens[i]})
        
        return output_list
    """
    def log(metrics: dict) -> None:
        utils.log(metrics, args.output_dir / "log.csv")
        tqdm.write(
            f"epoch: {metrics['epoch']}")


    val_metrics =  evaluate(val_dataloader)
    utils.save_jsonl(val_metrics, args.output_dir / "val-metrics.jsonl")
        
    test_metrics = evaluate(test_dataloader)
    utils.save_jsonl(test_metrics, args.output_dir / "test-metrics.jsonl")

    utils.save_config(args, args.output_dir / "config.json")

if __name__=="__main__":
    args = Args()
    main(args)
