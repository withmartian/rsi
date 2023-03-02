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

# applies to AQUA, ECQA, QASC, QED, Creak
def extract_last_word(output):
  """
  Given an output string, extract the last word of the sentence.
  """
  if output:
    pred = output.strip().split(". ")[-1] # last sentence 
    pred = pred.split(" ")[-1].split(".")[0] # last word of last sentence without period
    return pred  # (a) or (A) or (1)

# applies to Creak
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