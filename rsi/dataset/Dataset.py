from abc import ABC, abstractmethod, ABC
from typing import Optional, List, Dict, Tuple
from collections import Counter
import os, json, warnings

class Dataset(ABC):
  name = None
  instruction = None
  cot_prompts = None
  direct_prompts = None

  @abstractmethod
  def __init__(self):
    # initialize the dataset
    pass

  @abstractmethod
  def create_prompt(self, exp, method: str):
    # create the CoT or direct prompt used to generate inferences
    # exp: a singular example
    # method: "cot" or "direct"
    pass

  @abstractmethod
  def correct_answer(self, exp):
    # exp: a singular example
    pass

  @abstractmethod
  def extract_answer(self, output):
    # output: list or singular example??
    pass

  @staticmethod
  def get_batches(examples, batch_size):
    """
    Returns a generator that yiels examples in batches of batch_size

    examples: a list of dataset examples
    batch_size: batch size
    """
    for i in range(0, len(examples), batch_size):
        yield examples[i:i + batch_size]

  @staticmethod
  def generate_batched(model, tokenizer, batch, num_pathways: int, device="gpu", **gen_kwargs):
    """
    model, tokenizer: huggingface model, tokenizer
    batch: one singular batch of example
    num_pathways: number of inferences to generate per example
    device: "gpu" or "cpu"
    gen_kwargs: parameters to be passed into model.generate(); for example:
    - max_length
    - temperature
    - top_p
    """
    # Set kwargs for generation
    if "max_length" not in gen_kwargs: gen_kwargs["max_length"] = 100
    if "temperature" not in gen_kwargs:  gen_kwargs["temperature"] = 0.7
    if "top_p" not in gen_kwargs:  gen_kwargs["top_p"] = 0.95
    if num_pathways > 1: gen_kwargs["do_sample"] = True
    gen_kwargs["num_return_sequences"] = num_pathways

    print(f'batch.input_ids size: {len(batch.input_ids)}')
    print(f'num_pathways: {num_pathways}')

    batch_out = model.generate(batch.input_ids, **gen_kwargs)
    return [tokenizer.decode(seqs, skip_special_tokens=True) for seqs in batch_out] 

  def check_required_attributes(self):
    """
    Check required attributes for child dataset classes and throw errors or warnings.
    Required attributes:
      - name: dataset name
    Recommended attributes:
      - instruction: Not required for eval datasets. If missing, the dataset cannot use the default create_finetune_mixture function for data augmentation.
      - direct_prompts: Not required for eval datasets. If missing, the dataset cannot use the default create_finetune_mixture function for data augmentation.
      - cot_prompts: Recommended for both training and eval datasets. If missing, the dataset cannot be used as a training dataset or used as an eval dataset with eval method set to "cot".
    """
    assert self.name != None, "Class attribute name cannot be None. Please define the attribute in your custom Dataset class."
    if self.cot_prompts == None:
      warnings.warn("Class attribute `cot_prompts` is None. This dataset cannot be used used as a training dataset or used as an eval dataset with eval method set to 'cot'")
    if self.direct_prompts == None:
      warnings.warn("Class attribute `direct_prompt` is None. `direct_prompts` is not required for eval datasets. When `direct_prompts` is not defined, the dataset cannot use the default `create_finetune_mixture` function for data augmentation.")
    if self.instruction == None:
      warnings.warn("Class attribute `instruction` is None. `instruction` is not required for eval datasets. When `instruction` is not defined, the dataset cannot use the default `create_finetune_mixture` function for data augmentation.")

  def get_pathways(self, model, tokenizer, dataset: List, batch_size: int, num_pathways: int, method: str = "direct", device="gpu", **gen_kwargs):
    """
    Return inference for the data passed in. shape: num_examples x num_pathways.

    model: huggingface model
    dataset: dataset (List)
    batch_size: batch size for pathway generation
    num_pathways: number of inferences generated per example
    num_samples: (start, end) indicates the starting and ending index of the samples that we'll use to generate pathways   FIXME
    method: "direct" or "cot"
    gen_kwargs: parameters to be passed into model.generate(); for example:
      - max_length
      - temperature
      - top_p
    """
    if method == "cot":
      assert self.cot_prompts != None, "get pathways with 'cot' requires class attribute cot_prompts, but cot_prompts is None."

    print("-"*100)
    print(f'generating {len(dataset)} {self.name} samples...')
    prompts = [self.create_prompt(exp, method) for exp in dataset]
    if device == "cpu":
      batches = [tokenizer(batch, return_tensors="pt", padding=True) for batch in self.get_batches(prompts, batch_size)]
    else: 
      batches = [tokenizer(batch, return_tensors="pt", padding=True).to("cuda") for batch in self.get_batches(prompts, batch_size)]
    result = []
    for i, batch in enumerate(batches):
      result.extend(list(self.get_batches(self.generate_batched(model, tokenizer, batch, num_pathways, device=device, **gen_kwargs), num_pathways)))
    return result


  def filter_generated_paths(self, exp, paths: List, filter: Optional[str] = "correct", **filter_args):
    """
    This is used by train datasets. Takes in a singular example and a list of generated paths and return a list of selected paths based on the filter method.
    
    exp: an entry from the dataset
    paths: a list of paths generated for that example
    filter: "correct" or "majority"
    filter_args: 
      - confidence_cutoff
      - other filter config values # FIXME
    """
    if filter == "correct":
      filtered_pred = self.correct_answer(exp)
      filtered_paths = [p for p in paths if self.extract_answer(p) == filtered_pred]
    elif filter == "majority":
      confidence_cutoff = filter_args.pop("confidence_cutoff", 0.8)
      predictions = {}
      for i in range(len(paths)):
        predictions[i] = self.extract_answer(paths[i])
      if len(predictions) == 0: 
        filtered_pred = ''
      else: 
        counter = Counter(predictions.values())
        filtered_pred = counter.most_common(1)[0][0]
      voted_keys = [k for k, v in predictions.items() if v == filtered_pred]
      if len(voted_keys)/len(paths) >= confidence_cutoff:
        filtered_paths = [paths[k] for k in voted_keys]
      else:
        filtered_paths = []
    return filtered_paths, filtered_pred
    
  def calculate_dataset_accuracy(self, dataset, pathways):
    """
    Return the accuracy of one class in EvalDataset

    dataset: List. The dataset of the class that we're evaluating
    pathways: a list of answers generated for each example in the dataset
    """
    predictions = [self.extract_answer(path) for path in pathways]
    truths = [self.correct_answer(exp) for exp in dataset]
    correct_count = sum([p == t for p, t in zip(predictions, truths)])
    return correct_count/len(truths)

  def eval(self, model, tokenizer, dataset, batch_size, class_name = None, method = "direct", save_every = 1000, resume_from_checkpoint = False, checkpoint_dir: Optional[str] = None):
    """
    Returns the overall accuracy for a dataset

    model, tokenizer: huggingface model, tokenizer
    dataset: a dataset (List, after randomization in __init__)
    batch_size: batch size for generating inferences
    method: "direct" or "cot"
    save_every: number of inferences to run before saving to file
    resume_from_checkpoint: flag for whether to resume from previous checkpoint
    """
    if method == "cot":
      assert self.cot_prompts != None, "'cot' eval requires class attribute cot_prompts, but cot_prompts is None."
    print(f'--- Evaluating {len(dataset)} examples from {self.name} ---')
    prompts = [self.create_prompt(exp, class_name, method) for exp in dataset] if class_name else [self.create_prompt(exp, method) for exp in dataset]
    # FIXME: pass in device arg to choose if we want to use cuda
    batches = [tokenizer(batch, return_tensors="pt", padding=True).to("cuda") for batch in self.get_batches(prompts, batch_size)]

    # # create checkpoint_dir
    # if not checkpoint_dir:
    #   checkpoint_dir = f'eval-checkpoint'
    # if not os.path.exists(checkpoint_dir):
    #   os.makedirs(checkpoint_dir)
    # if class_name:
    #   save_to = f'{checkpoint_dir}/{self.name}-{class_name}-predictions.json'
    # else:
    #   save_to = f'{checkpoint_dir}/{self.name}-predictions.json'

    # if resume_from_checkpoint:
    #   if not os.path.exists(save_to):
    #     print(f'[ERR] Failed to resume from checkpoint. {save_to} not found.')
    #     return
    #   with open(save_to, "r") as f:
    #     predictions = json.load(f)
    #   batches = batches[len(predictions)//batch_size: ]
    #   print(f'Loaded {len(predictions)} predictions. Resume from batch #{len(predictions)//batch_size}')
    # else:
    #   predictions = []

    predictions = []

    for i in range(len(batches)):
      predictions.extend(self.generate_batched(model, tokenizer, batches[i], num_pathways=1))
      # if (i != 0 and i % save_every == 0) or (i == len(batches)-1):
      #   print(f'saving {i}th checkpoint ...')
        # with open(save_to, "w") as f:
        #   json.dump(predictions, f)
    return self.calculate_dataset_accuracy(dataset, predictions)