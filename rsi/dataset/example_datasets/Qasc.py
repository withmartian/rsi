from .TrainDataset import TrainDataset
from datasets import load_dataset
import random
from dataset_utils import extract_last_word
from example import generate_5way_finetune_mixture

class Qasc(TrainDataset):
  name = "qasc"
  instruction = "Answer the following multiple choice question with options (A), (B), (C), (D), (E), (F), (G), (H)."
  cot_prompts = """Q: What forms beads of water?  (A) Necklaces. (B) Steam. (C) Glass beads . (D) a wave (E) tiny (F) a solute (G) rain (H) Bracelets.
A: beads of water are formed by water vapor condensing. An example of water vapor is steam. The answer is (B).

Q: what kind of beads are formed from vapor condensing? (A) tiny (B) H20 (C) h2o (D) carbon (E) hydrogen (F) rain (G) oxygen (H) Dew
A: beads of water are formed by water vapor condensing. Water is made up of H2O molecules. The answer is (C).

Q: what kind of beads are formed by their vapor condensing? (A) h2o (B) rain (C) tiny (D) H20 (E) CO 2 (F) blue (G) Aves (H) Dew
A: beads of water are formed by water vapor condensing. Water is made up of H2O molecules. The answer is (A).

Q: What happens to the heat energy during condensation. (A) It goes to the remaining air molecules (B) Temperature changing (C) they travel great distances (D) raising their temperature (E) liquid precipitation (F) changing phenomenon (G) Movement of an air mass (H) electrons in motion
A: beads of water are formed by water vapor condensing. When water vapor condenses, energy in the form of heat is given to the remaining air molecules. The answer is (A)."""
  direct_prompts = """Q: What forms beads of water?  (A) Necklaces. (B) Steam. (C) Glass beads . (D) a wave (E) tiny (F) a solute (G) rain (H) Bracelets.
A: The answer is (B).

Q: what kind of beads are formed from vapor condensing? (A) tiny (B) H20 (C) h2o (D) carbon (E) hydrogen (F) rain (G) oxygen (H) Dew
A: The answer is (C).

Q: what kind of beads are formed by their vapor condensing? (A) h2o (B) rain (C) tiny (D) H20 (E) CO 2 (F) blue (G) Aves (H) Dew
A: The answer is (A).

Q: What happens to the heat energy during condensation. (A) It goes to the remaining air molecules (B) Temperature changing (C) they travel great distances (D) raising their temperature (E) liquid precipitation (F) changing phenomenon (G) Movement of an air mass (H) electrons in motion
A: The answer is (A)."""

  def __init__(self, generate_finetune_mixture = generate_5way_finetune_mixture, random_seed = 0):
    """
    This function initializes the instruction, cot and direct prompts, 
    randomized training and testing set, and dataset sample counter
    """
    self.check_required_attributes()
    super().__init__(generate_finetune_mixture)
    dataset = load_dataset("qasc")
    random.seed(random_seed)
    self.train = random.sample([exp for exp in dataset["train"]], dataset.num_rows["train"])
    self.test = random.sample([exp for exp in dataset["test"]], dataset.num_rows["test"])
    self.last_sampled = 0

  def create_prompt(self, exp, method: str = "direct"):
    """
    exp: a singular example
    method: "cot" or "direct"
    """
    question = exp["formatted_question"]
    if method == "cot":
      return self.cot_prompts + "\n\nQ: " +  question + "\n" + "A:"
    elif method == "direct":
      return "Q: " +  question + "\n" + "A:"

  def correct_answer(self, exp):
    """
    exp: a singular example
    """
    return f'({exp["answerKey"]})'

  def extract_answer(self, output):
    """
    output: a singular output
    """
    return extract_last_word(output)
