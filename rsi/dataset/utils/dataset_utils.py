import re

def extract_numerical_ans(output):
  """
  Given an output string, output the first number that appears in the last sentence.
  """
  if output: 
    pred = output.strip().split(". ")[-1] # last sentence of the output
    pred = [s for s in re.findall(r'-?\d+\.?\d*', pred)] # extract all numbers from the sentence
    if pred:
      try:
        return float(pred[0])
      except:
        return 'invalid'
  return ''

def extract_last_word(output):
  """
  Given an output string, extract the last word of the sentence.
  """
  if output:
    pred = output.strip().split(". ")[-1] # last sentence 
    pred = pred.split(" ")[-1].split(".")[0] # last word of last sentence without period
    return pred  # (a) or (A) or (1)

def extract_tf_ans(output):
  """
  Given an output string, turn true/false or yes/no strings into boolean True or False
  """
  if output.lower() == 'true' or output == 'yes':
    return True
  elif output.lower() == 'false' or output == 'no':
    return False
  else:
    return 'invalid'

def generate_5way_finetune_mixture(instruction: str, direct_prompts: str, cot_prompts: str, exp, filtered_paths: List, filtered_pred):
  """
  An example function for the generate_finetune_mixture arg in Dataset class initialization. 
  This function takes in a list of filtered paths and augments the paths with instructions, cot promts, and examplars to return a list of diversified mixtures for fine tuning. 
  instruction: Dataset.instruction
  direct_prompts: Dataset.direct_prompts
  cot_prompts: Dataset.cot_prompts
  exp: an entry from the dataset
  filtered_paths: a list of filtered paths returned by the filter_generated_paths function
  filtered_pred: the prediction/answer to the entry that all filtered paths agree upon
  """
  assert all([instruction, direct_prompts, cot_prompts]), "One or more of required class attributes `instruction`, `direct_prompts`, `cot_prompts` is None."
  mixture = []
  question = exp["question"]
  direct_examplars = ''
  for exp in direct_prompts.split("\n\n"):
    direct_examplars += f'{instruction}\n{exp}\n\n'

  # instruction + no examplar + direct answer
  input = f'{instruction}\nQ: {question}'
  target = f'A: The answer is {filtered_pred}.'
  mixture.append({"input": input, "target": target})

  # instruction + direct examplars + direct answer
  input = direct_examplars + f'{instruction}\nQ: {question}'
  target = f'A: The answer is {filtered_pred}.'
  mixture.append({"input": input, "target": target})

  for path in filtered_paths:
    target = f'A: {path}'

    # instruction + no examplar + CoT answer
    input = f'{instruction[:len(instruction)-1]} by reasoning step-by-step.\nQ: {question}'
    mixture.append({"input": input, "target": target})

    # instruction + CoT examplars + CoT answer
    input = f'{cot_prompts}\n\n{instruction}.\nQ: {question}'
    mixture.append({"input": input, "target": target})

    # no instruction + CoT examplars + CoT answer
    input = f'{cot_prompts}\n\nQ: {question}'
    mixture.append({"input": input, "target": target})
    
  return mixture


def calculate_overall_accuracy(class_accuracy):
  """
  Take the average of all class/subset accuracies and compute the overall accuracy of the dataset
  
  class_accuracy: a dictionary of class_name and class_accuracy
  """
  total = sum([class_accuracy[c] for c in class_accuracy])
  return total/len(class_accuracy)