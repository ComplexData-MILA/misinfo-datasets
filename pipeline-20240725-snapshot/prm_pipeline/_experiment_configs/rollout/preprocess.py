"""Preprocess json output from R."""
import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument("input_json_path")
parser.add_argument("output_jsonl_path")
args = parser.parse_args()


with open(args.input_json_path, "rb") as input_file:
    data_bytes = input_file.read()


data_str = data_bytes.decode("latin-1")
data = json.loads(data_str)
assert isinstance(data, list), type(data)

with open(args.output_jsonl_path, "w") as output_file:
    for item in data:
        output_file.write(json.dumps(item) + "\n")
