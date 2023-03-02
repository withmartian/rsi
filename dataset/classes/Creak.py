from .TrainDataset import TrainDataset
from datasets import load_dataset
import random
from dataset_utils import extract_last_word, extract_tf_ans
from example import generate_5way_finetune_mixture

class Creak(TrainDataset):
  name = "creak"
  instruction = "Answer the following true/false question."
  cot_prompts = """Q: Is the following sentence plausible? “Only people named Floyd wearing pink are allowed to attend Pink Floyd concerts.”
A: Pink Floyd is a popular rock group. The rock group would not be as popular is they had such requirements for their concerts. So the answer is false. 

Q: Is the following sentence plausible? “Fax works without any internet connection.”
A: Internet connection is required for a fax to function well. So the answer is false.

Q: Is the following sentence plausible? “A popular RCA Records artist who has many hit songs is Kesha.”
A: Kesha is a musician from Nashville, Tennessee. So the answer is true. 

Q: Is the following sentence plausible? “Larry King served tea during his show.”
A: He had a set format that did not involve tea. So the answer is false. """
  direct_prompts = """Q: Is the following sentence plausible? “Only people named Floyd wearing pink are allowed to attend Pink Floyd concerts.”
A: So the answer is false. 

Q: Is the following sentence plausible? “Fax works without any internet connection.”
A: So the answer is false.

Q: Is the following sentence plausible? “A popular RCA Records artist who has many hit songs is Kesha.”
A: So the answer is true. 

Q: Is the following sentence plausible? “Larry King served tea during his show.”
A: So the answer is false. """

  def __init__(self, generate_finetune_mixture = generate_5way_finetune_mixture, random_seed = 0):
    """
    Initializes attributes of the dataset.
    generate_finetune_mixture: a function that takes in a list of filtered inferences and return a list of fine-tune entries
    """
    self.check_required_attributes()
    super().__init__(generate_finetune_mixture)
    dataset = load_dataset("amydeng2000/CREAK")
    random.seed(random_seed)
    self.train = random.sample([exp for exp in dataset["train"]], dataset.num_rows["train"])
    self.valid = random.sample([exp for exp in dataset["validation"]], dataset.num_rows["validation"])
    self.last_sampled = 0

  def create_prompt(self, exp, method: str = "direct"):
    """
    exp: a singular example
    method: "cot" or "direct"
    """
    question = f'Is the following sentence plausible? "{exp["sentence"]}"'
    if method == "cot":
      return self.cot_prompts + "\n\nQ: " +  question + "\n" + "A:"
    elif method == "direct":
      return "Q: " +  question + "\n" + "A:"

  def correct_answer(self, exp):
    """
    exp: a singular example
    """
    return exp['label'] == 'true'

  def extract_answer(self, output):
    """
    output: a singular output
    """
    return extract_tf_ans(extract_last_word(output))
