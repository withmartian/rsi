import random, json, os, torch
from typing import Tuple, List, Dict
from Files.generate import generate_training_dataset
from Files.fineTune import fine_tune
from Files.evaluate import evaluate
from dataset.Dataset import Dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration
from dataset.example_datasets.Creak import Creak
from dataset.example_datasets.Ecqa import Ecqa
from dataset.example_datasets.Tydiqa import Tydiqa
from transformers.optimization import Adafactor
from transformers import TrainingArguments, Trainer

def select_eval_iterations(num_evals: int, total_iter: int):
    """
    Given the total number of RSI iterations and the number of evaluations, 
    return a list of indicies (iterations) at which we evaluate the model.
    """
    assert total_iter >= num_evals
    r = total_iter//num_evals
    mod = total_iter % num_evals
    counts = [1+r]*mod + [r]*(num_evals-mod)
    random.shuffle(counts)
    counts = [sum(counts[:i])-1 for i in range(1, len(counts))]
    counts.append(total_iter-1)
    return counts

def update_rsi_states(iteration, current_state):
    if not os.path.exists("rsi-states.json"):
        with open("rsi-states.json", "w") as f:
            json.dump({}, f)
    with open("rsi-states.json", "r") as f:
        states = json.load(f)
    states["iteration"] = iteration
    states["current_state"] = current_state
    with open("rsi-states.json", "w") as f:
        json.dump(states, f)

def rsi(N, iterations, num_evals, model, tokenizer, train_datasets: List[Tuple[Dataset, List]], eval_datasets, generate_args: Dict, train_args: Dict, eval_args: Dict):
    """
    datasets_dics: a dictionary. Key: data objects. Value: dataset of the corresponding data object. 
    generate_args: Dict
        - batch_size: default to 16
        - num_pathways: defualt to 32
        - method: default to "cot"
        - checkpoint_dir: default to None
    train_args: Dict
        - training_args: A Huggingface TrainingArguments object.
        - optimizer: default adafactor
        - lr_scheduler: default None
        - model_output_dir: Optional[str] = None
        - resume_trainer_states: default True
        - recover_from_checkpoint: default False
    eval_args: Dict
        - batch_size
        - save_every: Optional[int] = 50
        - resume_from_checkpoint: Optional[bool] = False
        - checkpoint_dir: Optional[str] = None
    """
    if not os.path.exists("mixture"):
        os.mkdir("mixture")
    performance = []
    if iterations >= num_evals:
        eval_iters = select_eval_iterations(num_evals, iterations)
        for iter in range(iterations):
            # generate
            update_rsi_states(iter, "generate")
            mixture = generate_training_dataset(N, model, tokenizer, train_datasets, **generate_args)
            with open(f'mixture/iter-{iter}.json', "w") as f:
                json.dump(mixture, f)
            # fine tune
            update_rsi_states(iter, "fine-tune")
            fine_tune(f'mixture/iter-{iter}.json', model, tokenizer, **train_args)
            # eval
            update_rsi_states(iter, "eval")
            if iter in eval_iters:
                metrics = evaluate(eval_datasets, model, tokenizer, **eval_args)
                performance.append((iter*N*len(train_datasets), metrics))

    return performance



if __name__ == "__main__":
    N = 30
    iterations = 2
    num_evals = 1
    tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-small")
    model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-small", torch_dtype=torch.bfloat16, device_map="auto") #, cache_dir="drive/MyDrive/FLAN-T5-XXL"
    creak = Creak()
    ecqa = Ecqa()
    train_datasets = [(creak, creak.train), (ecqa, ecqa.train)]
    eval_datasets = [(Tydiqa(), "direct")]
    generate_args = {
        "batch_size": 8,
        "num_pathways": 32,
        "method": "cot"
    }
    train_args = {
        "optimizer": Adafactor(model.parameters(), relative_step=False, warmup_init=False, scale_parameter=False, lr=1e-3),
        "lr_scheduler": None,
        "training_args": TrainingArguments(         
            output_dir="./fine_tune_checkpoints", 
            logging_steps=50,
            num_train_epochs=3,         
            per_device_train_batch_size=16,
            save_strategy="epoch",
            save_total_limit=1,
            ),
        "resume_trainer_states": True,
        "recover_from_checkpoint": False
    }
    eval_args = {
        "batch_size": 32,
        "save_every": 100,
        "resume_from_checkpoint": False,
        "checkpoint_dir": None
    }

    rsi(N, iterations, num_evals, model, tokenizer, train_datasets, eval_datasets, generate_args, train_args, eval_args)
