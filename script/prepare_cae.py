import random
import pandas as pd
import re
from pathlib import Path
import unicodedata

import utils as utils


class Args:
    def __init__(self, base_dir):
        self.lancers_dir: Path = Path(base_dir, "lancers")
        self.data_dir: Path = Path(base_dir, "data/cae")
        

def process_input(input: str) -> str:
    input = unicodedata.normalize("NFKC", input)
    # 文末の空白文字や改行・タブが取り除かれる
    input = input.strip().replace("\n", "").replace("\r", "")
    return input

def process_output(output: str) -> str:
    output = unicodedata.normalize("NFKC", output)
    output = output.strip().replace("\n", "").replace("\r", "")
    return output


# データを作成してjsonlの形式でtrain.jsonl, val.jsonl, test.jsonlを保存
def main(args: Args):

    df = pd.read_csv(args.lancers_dir / "cae.csv")

    output_list = []
    train = []
    val = []
    test = []

    for text_id in range(1, 26):            
        ans_df = pd.read_csv(args.lancers_dir / f"data/{text_id}.csv")

        for index, que in df.iterrows():
            for ans in ans_df[que["id"]]:
                dic = {}
                dic["input"] = que["text"]
                processed_ans = process_output(ans)
                if len(processed_ans.split("。")[:-1]) != 1:
                    for p_ans in processed_ans.split("。")[:-1]:          
                        dic["output"] = p_ans+"。"
                        output_list.append(dic)
                        if text_id  in [2, 13, 12]:
                            val.append(dic)
                        elif text_id in [6, 22, 25]:
                            test.append(dic)
                        else:
                            train.append(dic)

                else:
                    dic["output"] = processed_ans
                    output_list.append(dic)
                    if text_id  in [7, 9]:
                        val.append(dic)
                    elif text_id in [19, 21]:
                        test.append(dic)
                    else:
                        train.append(dic)
                

    utils.save_jsonl(output_list, args.data_dir / "all.jsonl")
    utils.save_jsonl(train, args.data_dir / "train.jsonl")
    utils.save_jsonl(val, args.data_dir / "val.jsonl")
    utils.save_jsonl(test, args.data_dir / "test.jsonl")

if __name__=="__main__":
    base_dir = "/home/yuki/Research/CLEVRER"
    args = Args(base_dir)
    main(args)
