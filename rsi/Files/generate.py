import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dataset.utils.dataset_utils import generate_5way_finetune_mixture

def _generate_dataset(mixture, N, model, tokenizer, data_object, dataset, batch_size, num_pathways=32, method="direct"):
  """
  Generate logic paths, filter, and augment one dataset

  mixture: List. A list of existing mixtures recovered from checkpoints, or an empty list. 
  N: RSI step size. The desired length of the final dataset augmentation
  data_object: a instantiated dataset class object. Ex: Aqua()
  dataset: a specific dataset that belongs to the data_object
  """
  while len(mixture) < N:
    # each time, we generate pathways for batch_size number of examples
    start = data_object.last_sampled
    pathways = data_object.get_pathways(model, tokenizer, dataset, batch_size, num_pathways, num_samples=(start, start + batch_size), method=method)
    for i in range(batch_size):
      exp = dataset[start+i]
      question = data_object.get_question(exp)
      filtered_paths, filtered_pred = data_object.filter_generated_paths(exp, pathways[i])
      mixture.extend(generate_5way_finetune_mixture(data_object.instruction, data_object.direct_prompts, data_object.cot_prompts, question, filtered_paths, filtered_pred))

      # if len(curr_mixture) // save_every > last_saved:
      #   last_saved += 1
      #   with open(f'mixture-checkpoint/{dset}-checkpoint.json', "w") as f:
      #     json.dump(curr_mixture, f)
  return mixture

def generate_training_dataset(N, model, tokenizer, datasets, batch_size, num_pathways, method):
  """
  N: RSI step size. The desired length of the final augmentation for each dataset that we use to generate.
  datasets: a dictionary. Key: data objects. Value: dataset of the corresponding data object. 

  """
  # checkpointing
  final_mixture = []
  # if resume_from_checkpoint:
  #   with open(f'mixture-checkpoint/states.json', "r") as f:
  #       states = json.load(f)
  # else:
  #   states = create_checkpoint_files("mixture-checkpoint","states.json")
  # files = os.listdir("mixture-checkpoint")
  for data_object in datasets:
    mixture = []
    # checkpointing
    # if states[dset] == "incomplete" and f'{dset}-checkpoint.json' in files:
    #   with open(f'mixture-checkpoint/{dset}-checkpoint.json', "r") as f:
    #     curr_mixture = json.load(f)
    #   last_saved = len(curr_mixture) // save_every
    # elif states[dset] == "incomplete":
    #   curr_mixture = []
    #   last_saved = 0
    # else: # dataset status == complete
    #   continue
    mixture = _generate_dataset(mixture, N, model, tokenizer, data_object, datasets[data_object], batch_size)
    final_mixture.extend(mixture[:N])
    # checkpointing
    # save_generate_state("mixture-checkpoint/states.json", dset)
    # with open(f'mixture-checkpoint/all-mixture.json', "w") as f:
    #     json.dump(mixture, f)
  return final_mixture