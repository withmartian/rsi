from datasets import load_dataset
import random, sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.dataset_utils import extract_last_word

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Dataset import Dataset
PATH = os.path.dirname(os.path.abspath(__file__))

class Bbh(Dataset):
    name = "bbh"
    classes = ['boolean_expressions', 'causal_judgement', 'date_understanding', 'disambiguation_qa', 'dyck_languages', 'formal_fallacies', 'geometric_shapes', 'hyperbaton', 'logical_deduction_five_objects', 'logical_deduction_seven_objects', 'logical_deduction_three_objects', 'movie_recommendation', 'multistep_arithmetic_two', 'navigate', 'object_counting', 'penguins_in_a_table', 'reasoning_about_colored_objects', 'ruin_names', 'salient_translation_error_detection', 'snarks', 'sports_understanding', 'temporal_sequences', 'tracking_shuffled_objects_five_objects', 'tracking_shuffled_objects_seven_objects', 'tracking_shuffled_objects_three_objects', 'web_of_lies', 'word_sorting']
    cot_prompts = {}
    for filename in os.listdir(f'{PATH}/cot_prompts/bbh'):
        f = open(f'{PATH}/cot_prompts/bbh/{filename}', 'r')
        cot_prompts[filename.split(".")[0]] = f.read().split('\n-----\n')[1]

    def __init__(self, random_seed = 0):
        """
        Initializes the dataset and check required attributes
        """
        self.check_required_attributes()
        self.train = {}  # Big-Bench-Hard only has no train split
        random.seed(random_seed)
        for c in self.classes:
            dataset = load_dataset('lukaemon/bbh', c, split="test")
            self.train[c] = random.sample([exp for exp in dataset], dataset.num_rows)

    def create_prompt(self, exp, class_name: str = None, method: str = "direct"):
        """
        exp: a singular example
        method: for Tydiqa, we only use the "direct" method because baseline accuracy is high. 
        """
        question = exp['input']
        if method == "cot":
            assert class_name != None, "Must provide `class_name` to create a few shot cot prompt."
            return self.cot_prompts[class_name] + "\n\nQ: " +  question + "\n" + "A:"
        elif method == "direct":
            return "Q: " +  question + "\n" + "A:"

    def correct_answer(self, exp):
        """
        exp: a singular example
        """
        return exp["target"]

    def extract_answer(self, output):
        """
        output: a singular output
        """
        try:
            # try getting the string after "So the answer is" without period at the end
            return output.split("So the answer is ")[1].split(".")[0] 
        except:
            # if the inference did not follow the CoT format, extract the last word of the inference output
            return extract_last_word(output)