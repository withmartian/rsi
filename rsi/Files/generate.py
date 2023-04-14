import sys, os, json, torch, random, argparse
from typing import Tuple, List
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dataset.utils.dataset_utils import generate_5way_finetune_mixture
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import LlamaForCausalLM, LlamaTokenizer
from dataset.example_datasets.Aqua import Aqua
from dataset.example_datasets.Creak import Creak
from dataset.example_datasets.Ecqa import Ecqa
from dataset.example_datasets.Esnli import Esnli
from dataset.example_datasets.Gsm8k import Gsm8k
from dataset.example_datasets.Qasc import Qasc
from dataset.example_datasets.Strategyqa import Strategyqa
from dataset.Dataset import Dataset
from Files.rsi_utils.rsi_utils import str_to_bool, get_checkpoint_states

def _generate_dataset(mixture, N, model, tokenizer, data_object, batch_size, num_pathways=32, method="direct"):
  """
  Generate logic paths, filter, and augment one dataset

  mixture: List. A list of existing mixtures recovered from checkpoints, or an empty list. 
  N: RSI step size. The desired length of the final dataset augmentation
  data_object: a instantiated dataset class object. Ex: Aqua()
  """
  last_sampled = 0
  data = data_object.train
  while len(mixture) < N:
    # each time, we generate pathways for batch_size number of examples
    if last_sampled + batch_size >= len(data):
      return mixture
    batch_data = data[last_sampled : last_sampled + batch_size]
    pathways = data_object.get_pathways(model, tokenizer, batch_data, batch_size, num_pathways, method=method)
    for exp, exp_paths in zip(batch_data, pathways):
      question = data_object.get_question(exp)
      filtered_paths, filtered_pred = data_object.filter_generated_paths(exp, exp_paths)
      mixture.extend(generate_5way_finetune_mixture(data_object.instruction, data_object.direct_prompts, data_object.cot_prompts, question, filtered_paths, filtered_pred))
  return mixture

def generate_training_dataset(iteration, N, model, tokenizer, datasets: List[Tuple[Dataset, List]], resume_from_checkpoint=False, batch_size=16, num_pathways=32, method="cot", checkpoint_dir="generate_checkpoints"):
  """
  One iteration of data generation. Returns the training dataset (List).
  N: RSI step size. The desired length of the final augmentation for each dataset that we use to generate.
  datasets: a list of tuples containing a Dataset instance and its dataset to be used for generating training data
  iteration: int. Used to find the corresponding iteration folder in checkpoint_dir to save checkpoints. 
  """
  # checkpointing
  states = get_checkpoint_states(checkpoint_dir, resume_from_checkpoint, iteration)
  path = f'{checkpoint_dir}/{iteration}'
  if not os.path.exists(path):
    os.mkdir(path)

  final_mixture = []
  if resume_from_checkpoint and os.path.exists(f'{path}/all_data.json'):
      with open(f'{path}/all_data.json', "r") as f:
        final_mixture = json.load(f)

  for data_object, data in datasets:
    if not data_object.name in states["completed_datasets"]:
      mixture = []
      mixture = _generate_dataset(mixture, N, model, tokenizer, data_object, batch_size, num_pathways, method)
      # save generated data
      with open(f'{path}/{data_object.name}.json', "w") as f:
        json.dump(mixture, f)
      final_mixture.extend(mixture[:N])
      with open(f'{path}/all_data.json', "w") as f:
        json.dump(final_mixture, f)
      # update states
      states["completed_datasets"].append(data_object.name)
      with open(f'{checkpoint_dir}/states.json', "w") as f:
        json.dump(states, f)
  
  return final_mixture


if __name__ == "__main__": 
  parser = argparse.ArgumentParser()
  parser.add_argument('--resume', type=str_to_bool, default=None)
  args = parser.parse_args()
  resume = args.resume if args.resume is not None else generate_training_dataset.__defaults__[-1]
  print(f'resume generation: {resume}')
  
  aqua = Aqua()
  # creak = Creak()
  # ecqa = Ecqa()
  # esnli = Esnli()
  # gsm8k = Gsm8k()
  # qasc = Qasc()
  # strategyqa = Strategyqa()
  datasets = [(aqua, aqua.train)]
  # datasets = [(aqua, aqua.train), (creak, creak.train), (ecqa, ecqa.train), (esnli, esnli.train), (gsm8k, gsm8k.train), (qasc, qasc.train), (strategyqa, strategyqa.train)]
  batch_size = 1
  N = 10
  iteration = 200
  tokenizer = LlamaTokenizer.from_pretrained("decapoda-research/llama-7b-hf")
  model = LlamaForCausalLM.from_pretrained("decapoda-research/llama-7b-hf").to("cuda")
  # tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-small")
  # model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-small", torch_dtype=torch.bfloat16, device_map="auto") #, cache_dir="drive/MyDrive/FLAN-T5-XXL"
  mix = generate_training_dataset(iteration, N, model, tokenizer, datasets, resume, batch_size, num_pathways=16, method="cot")
  print(len(mix))
  print(mix)