import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from dataset.example_datasets.Mmlu import  Mmlu
from datasets import load_dataset
import random
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch


mmlu = Mmlu()
tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-small")
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-small", torch_dtype=torch.bfloat16, device_map="auto")
batch_size = 16
method = "direct"
save_every = 500

def mmlu_eval(mmlu, model, tokenizer, batch_size, method, save_every):
    total_accuracy = {}
    for c in mmlu.classes:
        dataset = load_dataset("hendrycks_test", c)
        eval_dataset = random.sample([exp for exp in dataset["auxiliary_train"]], dataset.num_rows["auxiliary_train"], 50)  # FIXME: sample only 50 exmaples
        accuracy = mmlu.eval(model, tokenizer, eval_dataset, batch_size, class_name=c, method=method, save_every=save_every)
        total_accuracy[c] = accuracy
    print(total_accuracy)
    return total_accuracy


if __name__ == "__main__":
    mmlu_eval(mmlu, model, tokenizer, batch_size, method, save_every)