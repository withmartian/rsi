import os
import json
from .Dataset import Dataset
from typing import Optional, List, Dict, Tuple

class EvalDataset(Dataset):

  def check_required_attributes(self):
    """
    Return true if the Dataset class has all required attributes:
      - name
      - cot_prompts
    """
    assert self.name != None, f'Class attribute name cannot be None. Please define the attribute in your custom Dataset class.'
    if self.cot_prompts == None:
      print("[Warning] Class attribute cot_prompts is None.")

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
    
  @staticmethod
  def calculate_overall_accuracy(class_accuracy):
    """
    Take the average of all class/subset accuracies and compute the overall accuracy of the dataset
    
    class_accuracy: a dictionary of class_name and class_accuracy
    """
    total = sum([class_accuracy[c] for c in class_accuracy])
    return total/len(class_accuracy)


  def eval(self, model, tokenizer, dataset, batch_size, class_name = None, method = "direct", save_every = 1000, resume_from_checkpoint = False, checkpoint_dir: Optional[str] = None):
    """
    Returns the overall accuracy for a MMLU class

    model, tokenizer: huggingface model, tokenizer
    dataset: a dataset (List, after randomization in __init__) for the class that we are evaluating
    batch_size: gpu batch size for inferences
    method: "direct" or "cot"
    save_every: number of inferences to run before saving to file
    resume_from_checkpoint: flag for whether to resume from previous checkpoint
    """
    print(f'--- Evaluating {len(dataset)} examples from {self.name} ---')
    prompts = [self.create_prompt(exp, class_name, method) for exp in dataset] if class_name else [self.create_prompt(exp, method) for exp in dataset]
    batches = [tokenizer(batch, return_tensors="pt", padding=True) for batch in self.get_batches(prompts, batch_size)]

    # create checkpoint_dir
    if not checkpoint_dir:
      checkpoint_dir = f'{self.name}-eval-checkpoint'
    if not os.path.exists(checkpoint_dir):
      os.makedirs(checkpoint_dir)
    save_to = f'{checkpoint_dir}/predictions.json'

    if resume_from_checkpoint:
      if not os.path.exists(save_to):
        print(f'[ERR] Failed to resume from checkpoint. {save_to} not found.')
        return
      with open(save_to, "r") as f:
        predictions = json.load(f)
      batches = batches[len(predictions)//batch_size: ]
      print(f'Loaded {len(predictions)} predictions. Resume from batch #{len(predictions)//batch_size}')
    else:
      predictions = []

    for i in range(len(batches)):
      predictions.extend(self.generate_batched(model, tokenizer, batches[i], num_pathways=1))
      if i != 0 and i % save_every == 0:
        print(f'saving {i}th checkpoint ...')
        with open(save_to, "w") as f:
          json.dump(predictions, f)
    return self.calculate_dataset_accuracy(dataset, predictions)