import random
import pandas as pd
import re
from pathlib import Path
import unicodedata

import utils as utils


class Args:
    def __init__(self, base_dir):
        self.lancers_dir: Path = Path(base_dir, "lancers")
        self.data_dir: Path = Path(base_dir , "data_test")

def process_input(input: str) -> str:
    input = unicodedata.normalize("NFKC", input)
    # 文末の空白文字や改行・タブが取り除かれる
    input = input.strip().replace("\n", "").replace("\r", "")
    return input

def process_output(output: list) -> list:
    output = [unicodedata.normalize("NFKC", line) for line in output]
    output = [line.strip().replace("\n", "").replace("\r", "") for line in output]
    return output


# データを作成してjsonlの形式でtrain.jsonl, val.jsonl, test.jsonlを保存
def main(args: Args):

    df = pd.read_csv(args.lancers_dir / "45pattern.csv")
    text_df = pd.read_csv(args.lancers_dir / "train_predict_翻訳付.csv")

    output_list = []
    train = []
    val = []
    test = []


    for text_id in range(1, 26):
        start_index = 8*(text_id-1)+1
        end_index = 8*text_id

        ans_df = pd.read_csv(args.lancers_dir / f"data/{text_id}.csv")

        for i in range(start_index-1, end_index):
            random.seed(text_id+i)
            text = text_df.loc[i, "DeepL翻訳"]
            id = text_df.loc[i, "id"]
            match = re.search(r'(.+)[は|と|が](.+)[が|に|と](.*)', text)
            obj1 = match.groups()[0]
            obj2 = match.groups()[1]
            r_num_list = sorted(random.sample(range(0, 45), k=10))
            for r_num in r_num_list:
                dic = {}
                dic["ID"] = str(id)
                dic["pattern"] = str(r_num)
                input = text
                if df.loc[r_num]["床"] != "-":
                    input += f"床が{df.loc[r_num]['床']}している。"
                if df.loc[r_num]["質量1"] == "質量が等しい":
                    input += f"{obj1}と{obj2}の{df.loc[r_num]['質量1']}。"
                elif df.loc[r_num]["質量1"] != "-":
                    if df.loc[r_num]["スピード1"] == "-":
                        input += f"{obj1}の{df.loc[r_num]['質量1']}。{obj2}の{df.loc[r_num]['質量2']}。"
                    else:
                        input += f"{obj1}の{df.loc[r_num]['質量1']}。{obj2}の{df.loc[r_num]['質量2']}。"
                if df.loc[r_num]["スピード1"] == "スピードが等しい":
                    input += f"{obj1}と{obj2}の{df.loc[r_num]['スピード1']}。"
                elif df.loc[r_num]["スピード1"] != "-":
                    input += f"{obj1}の{df.loc[r_num]['スピード1']}。{obj2}の{df.loc[r_num]['スピード2']}。"
                dic["input"] = process_input(input)
                output = []
                for ans in ans_df.loc[:, f'{id}-{r_num}']:
                    output.append(ans)
                dic["output"] = process_output(output)
                if i == start_index-1:
                    val.append(dic)
                elif i == start_index:
                    test.append(dic)
                else:
                    train.append(dic)
                output_list.append(dic)

    utils.save_jsonl(output_list, args.data_dir / "all.jsonl")
    utils.save_jsonl(train, args.data_dir / "train.jsonl")
    utils.save_jsonl(val, args.data_dir / "val.jsonl")
    utils.save_jsonl(test, args.data_dir / "test.jsonl")

if __name__=="__main__":
    base_dir = "/home/yuki/Research/share"
    args = Args(base_dir)
    main(args)