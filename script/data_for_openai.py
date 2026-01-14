import re
from pathlib import Path
import json
import pandas as pd


class Args:
  def __init__(self):
    self.dataset_dir: Path = Path("../data")
    self.output_dir: Path = Path("../openai_data")
    self.train_data: str = "val8-1.jsonl"
    self.output_data: str = "openai_val8-1.jsonl"

def main(args: Args):
  df = pd.read_json(args.dataset_dir / args.train_data, orient='records', lines=True)
  df2 = pd.read_json(args.output_dir / args.output_data, orient='records', lines=True)
  inputlst = []
  outputlst = []

  for row in df.itertuples():
    input = row.input
    output = row.output

    inputlst.append(input)
    outputlst.append(output[0])


  df2["input"] = inputlst
  df2["output"] = outputlst
  df2.to_json(args.output_dir / args.output_data, orient='records', force_ascii=False, lines=True)


if __name__=="__main__":
  args = Args()
  main(args)

