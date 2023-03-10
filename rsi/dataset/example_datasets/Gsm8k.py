from typing import Optional, List, Dict, Tuple
from datasets import load_dataset
import random, sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.dataset_utils import extract_numerical_ans
from Dataset import Dataset


class Gsm8k(Dataset):
  name = "gsm8k"
  instruction = "Answer the following math question with an integer answer."
  cot_prompts = """Q: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?
A: He has 5 toys. He got 2 from mom, so after that he has 5 + 2 = 7 toys. Then he got 2 more from dad, so in total he has 7 + 2 = 9 toys. The answer is 9.

Q: There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?
A: There are 4 days from monday to thursday. 5 computers were added each day. That means in total 4 * 5 = 20 computers were added. There were 9 computers in the beginning, so now there are 9 + 20 = 29 computers. The answer is 29.

Q: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?
A: Michael initially had 58 balls. He lost 23 on Tuesday, so after that he has 58 - 23 = 35 balls. On Wednesday he lost 2 more so now he has 35 - 2 = 33 balls. The answer is 33.

Q: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?
A: She bought 5 bagels for $3 each. This means she spent 5 * $3 = $15 on the bagels. She had $23 in beginning, so now she has $23 - $15 = $8. The answer is 8."""
  direct_prompts = """Q: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?
A: The answer is 9.

Q: There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?
A: The answer is 29.

Q: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?
A: The answer is 33.

Q: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?
A: The answer is 8."""

  def __init__(self, subset: Optional[str] = "main", random_seed = 0):
    """
    Initializes attributes of the dataset
    subset: "main" or "socratic"
    """
    self.check_required_attributes()
    dataset = load_dataset("gsm8k", subset)
    random.seed(random_seed)
    self.train = random.sample([exp for exp in dataset["train"]], dataset.num_rows["train"])
    self.test = random.sample([exp for exp in dataset["test"]], dataset.num_rows["test"])
    self.last_sampled = 0

  def create_prompt(self, exp, method: str = "direct"):
    """
    exp: a singular example
    method: "cot" or "direct"
    """
    question = exp["question"]
    if method == "cot":
      return self.cot_prompts + "\n\nQ: " +  question + "\n" + "A:"
    elif method == "direct":
      return "Q: " +  question + "\n" + "A:"

  def correct_answer(self, exp):
    """
    exp: a singular example
    """
    return float(exp['answer'].split()[-1].replace(',', ''))

  def extract_answer(self, output):
    """
    output: a singular output
    """
    return extract_numerical_ans(output)
