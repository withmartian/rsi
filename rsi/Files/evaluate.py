import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dataset.example_datasets.Tydiqa import Tydiqa
from dataset.example_datasets.Bbh import Bbh
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

def evaluate(data_object, model, tokenizer, batch_size, method, save_every, resume_from_checkpoint=False, checkpoint_dir=None):
    total_accuracy = {}
    if hasattr(data_object, "classes"):
        for c in data_object.classes:
            accuracy = data_object.eval(model, tokenizer, data_object.train[c], batch_size, class_name=c, method=method, save_every=save_every, resume_from_checkpoint=resume_from_checkpoint, checkpoint_dir=checkpoint_dir)
            total_accuracy[f'{data_object.name}-{c}'] = accuracy
    else:
        accuracy = data_object.eval(model, tokenizer, data_object.train, batch_size, class_name=None, method=method, save_every=save_every, resume_from_checkpoint=resume_from_checkpoint, checkpoint_dir=checkpoint_dir)
        total_accuracy[data_object.name] = accuracy
    print(f'overall accuracy: {total_accuracy}')
    return total_accuracy

def main():
    data_object = Tydiqa()
    tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-small")
    model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-small", torch_dtype=torch.bfloat16, device_map="auto")
    batch_size = 16
    method = "direct"
    save_every = 10
    return evaluate(data_object, model, tokenizer, batch_size, method, save_every, resume_from_checkpoint=False, checkpoint_dir=None)

if __name__ == "__main__":
  main()