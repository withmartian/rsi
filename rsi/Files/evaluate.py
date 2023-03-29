import os, sys, torch, json, argparse
from typing import Tuple, List
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dataset.Dataset import Dataset
from dataset.example_datasets.Tydiqa import Tydiqa
from dataset.example_datasets.Aqua import Aqua
from dataset.example_datasets.Creak import Creak
from dataset.example_datasets.Ecqa import Ecqa
from dataset.example_datasets.Esnli import Esnli
from dataset.example_datasets.Gsm8k import Gsm8k
from dataset.example_datasets.Qasc import Qasc
from dataset.example_datasets.Strategyqa import Strategyqa
from transformers import T5Tokenizer, T5ForConditionalGeneration
from Files.rsi_utils.rsi_utils import str_to_bool, get_checkpoint_states


def get_checkpoint_states(checkpoint_dir, resume_from_checkpoint, iteration):
  states = {"iteration": iteration, "completed_datasets": []}
  if not os.path.exists(f'{checkpoint_dir}/states.json'):
    if not os.path.exists(f'{checkpoint_dir}'):
      os.mkdir(f'{checkpoint_dir}')
    with open(f'{checkpoint_dir}/states.json', "w") as f:
      json.dump(states, f)
  if resume_from_checkpoint:
    with open(f'{checkpoint_dir}/states.json', "r") as f:
        states = json.load(f)
  return states

def save_evaluation(performance, data_accuracy, performance_fp, dataset_slug, states, checkpoint_dir):
    performance.append(data_accuracy)
    # save performance
    with open(performance_fp, "w") as f:
        json.dump(performance, f)
    # update states
    states["completed_datasets"].append(dataset_slug)
    with open(f'{checkpoint_dir}/states.json', "w") as f:
        json.dump(states, f)

def evaluate(iteration, eval_datasets: List[Tuple[Dataset, str]], model, tokenizer, resume_from_checkpoint=False, checkpoint_dir="eval_checkpoints", batch_size=16, save_every=50):
    """
    eval_datasets: a list of tuples containing dataset object (ex. a Mmlu instance) and eval method ("cot" or "direct")
    """
    # checkpoint
    states = get_checkpoint_states(checkpoint_dir, resume_from_checkpoint, iteration)
    performance = []
    performance_fp = f'{checkpoint_dir}/iteration-{iteration}-performance.json'
    if resume_from_checkpoint and os.path.exists(performance_fp):
        with open(performance_fp, "r") as f:
            performance = json.load(f)

    for dataset, method in eval_datasets:
        data_accuracy = {}
        if hasattr(dataset, "classes"):
           for c in dataset.classes:
              if f'{dataset.name}-{c}' not in states["completed_datasets"]:
                accuracy = dataset.eval(model, tokenizer, dataset.train[c], batch_size, class_name=c, method=method, save_every=save_every, resume_from_checkpoint=resume_from_checkpoint, checkpoint_dir=checkpoint_dir)
                data_accuracy[f'{dataset.name}-{c}'] = accuracy
                save_evaluation(performance, data_accuracy, performance_fp, f'{dataset.name}-{c}', states, checkpoint_dir)
        else: # dataset don't have classes
           if dataset.name not in states["completed_datasets"]:
                accuracy = dataset.eval(model, tokenizer, dataset.train[:1000], batch_size, class_name=None, method=method, save_every=save_every, resume_from_checkpoint=resume_from_checkpoint, checkpoint_dir=checkpoint_dir)
                data_accuracy[dataset.name] = accuracy
                save_evaluation(performance, data_accuracy, performance_fp, dataset.name, states, checkpoint_dir)
        
        # if dataset.name not in states["completed_datasets"]:
        #     data_accuracy = {}
        #     if hasattr(dataset, "classes"):
        #         for c in dataset.classes:
        #             accuracy = dataset.eval(model, tokenizer, dataset.train[c], batch_size, class_name=c, method=method, save_every=save_every, resume_from_checkpoint=resume_from_checkpoint, checkpoint_dir=checkpoint_dir)
        #             data_accuracy[f'{dataset.name}-{c}'] = accuracy
        #     else:
        #         accuracy = dataset.eval(model, tokenizer, dataset.train, batch_size, class_name=None, method=method, save_every=save_every, resume_from_checkpoint=resume_from_checkpoint, checkpoint_dir=checkpoint_dir)
        #         data_accuracy[dataset.name] = accuracy
        #     print(f'{dataset.name} accuracy: {data_accuracy}')
        #     performance.append(data_accuracy)
        #     # save performance
        #     with open(performance_fp, "w") as f:
        #         json.dump(performance, f)
        #     # update states
        #     states["completed_datasets"].append(dataset.name)
        #     with open(f'{checkpoint_dir}/states.json', "w") as f:
        #         json.dump(states, f)
    return performance

def main():
    data_name = sys.argv[1]
    method = sys.argv[2]
    conversion = {"aqua": Aqua(), "creak": Creak(), "ecqa": Ecqa(), "esnli": Esnli(), "gsm8k": Gsm8k(), "qasc": Qasc(), "strategyqa": Strategyqa()}
    eval_datasets = [(conversion[data_name], method)]
    tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")
    model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base", torch_dtype=torch.bfloat16, device_map="auto")
    batch_size = 16
    save_every = 10
    iteration = 0

    print(eval_datasets[0][0].name, eval_datasets[0][1])
    print(evaluate(iteration, eval_datasets, model, tokenizer, checkpoint_dir="eval_checkpoints", batch_size=32, save_every=50))

if __name__ == "__main__":
  main()