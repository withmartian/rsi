import collections
from collections import Counter
from typing import Optional, List, Dict, Tuple
from Dataset import Dataset

class TrainDataset(Dataset):

  def __init__(self, generate_finetune_mixture):
    """
    This function takes in attributes and functions a TrainDataset instance must have
  
    generate_finetune_mixture: the dataset's generate_finetune_mixture function
    """
    self.generate_finetune_mixture = generate_finetune_mixture


  def check_required_attributes(self):
    """
    Return true if the Dataset class has all required attributes:
      - name
      - instruction
      - cot_prompts
      - direct_prompts
    """
    attributes = [self.name, self.instruction, self.cot_prompts, self.direct_prompts]
    attribute_names = ["name", "instruction", "cot_prompts", "direct_prompts"]
    for attribute, attribute_name in zip(attributes, attribute_names):
      assert attribute != None, f'Class attribute {attribute_name} cannot be None. Please define the attribute in your custom Dataset class.'
      
  
  def filter_generated_paths(self, exp, paths: List, filter: Optional[str] = "correct", **filter_args):
    """
    Takes in a singular example and a list of generated paths and return a list of selected paths based on the filter method
    
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
      return filtered_paths