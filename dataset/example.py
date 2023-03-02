from typing import Optional, List, Dict, Tuple

def generate_5way_finetune_mixture(self, exp, filtered_paths: List, filtered_pred):
  """
  An example function to pass into the TrainDataset class. This function takes in a list of filtered paths and returns a list of diversified mixtures for fine tuning. 
  exp: an entry from the dataset
  filtered_paths: a list of filtered paths returned by the filter_generated_paths function
  filtered_pred: the prediction/answer to the entry that all filtered paths agree upon
  """

  mixture = []
  question = exp["question"]
  direct_examplars = ''
  for exp in self.direct_prompts.split("\n\n"):
    direct_examplars += f'{self.instruction}\n{exp}\n\n'

  # instruction + no examplar + direct answer
  input = f'{self.instruction}\nQ: {question}'
  target = f'A: The answer is {filtered_pred}.'
  mixture.append({"input": input, "target": target})

  # instruction + direct examplars + direct answer
  input = direct_examplars + f'{self.instruction}\nQ: {question}'
  target = f'A: The answer is {filtered_pred}.'
  mixture.append({"input": input, "target": target})

  for path in filtered_paths:
    target = f'A: {path}'

    # instruction + no examplar + CoT answer
    input = f'{self.instruction[:len(self.instruction)-1]} by reasoning step-by-step.\nQ: {question}'
    mixture.append({"input": input, "target": target})

    # instruction + CoT examplars + CoT answer
    input = f'{self.cot_prompts}\n\n{self.instruction}.\nQ: {question}'
    mixture.append({"input": input, "target": target})

    # no instruction + CoT examplars + CoT answer
    input = f'{self.cot_prompts}\n\nQ: {question}'
    mixture.append({"input": input, "target": target})
    
  return mixture