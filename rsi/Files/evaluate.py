import os, sys
from typing import Tuple, List
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dataset.Dataset import Dataset
from dataset.example_datasets.Tydiqa import Tydiqa
from dataset.example_datasets.Bbh import Bbh
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

def evaluate(eval_datasets: List[Tuple[Dataset, str]], model, tokenizer, batch_size, save_every=50, resume_from_checkpoint=False, checkpoint_dir=None):
    """
    eval_datasets: a list of tuples containing dataset object (ex. a Mmlu instance) and eval method ("cot" or "direct")
    """
    all_metrics = []
    for dataset, method in eval_datasets:
        data_accuracy = {}
        if hasattr(dataset, "classes"):
            for c in dataset.classes:
                accuracy = dataset.eval(model, tokenizer, dataset.train[c], batch_size, class_name=c, method=method, save_every=save_every, resume_from_checkpoint=resume_from_checkpoint, checkpoint_dir=checkpoint_dir)
                data_accuracy[f'{dataset.name}-{c}'] = accuracy
        else:
            accuracy = dataset.eval(model, tokenizer, dataset.train, batch_size, class_name=None, method=method, save_every=save_every, resume_from_checkpoint=resume_from_checkpoint, checkpoint_dir=checkpoint_dir)
            data_accuracy[dataset.name] = accuracy
        print(f'{dataset.name} accuracy: {data_accuracy}')
        all_metrics.append(data_accuracy)
    return all_metrics

def main():
    eval_datasets = [(Bbh(), "direct")]
    tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-small")
    model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-small", torch_dtype=torch.bfloat16, device_map="auto")
    batch_size = 16
    save_every = 10
    return evaluate(eval_datasets, model, tokenizer, batch_size, save_every, resume_from_checkpoint=False, checkpoint_dir=None)

if __name__ == "__main__":
  main()