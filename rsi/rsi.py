import random, json, os
from typing import Tuple, List, Dict
from Files.generate import generate_training_dataset
from Files.fineTune import fine_tune
from Files.evaluate import evaluate
from dataset.Dataset import Dataset

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

def update_rsi_states(iteration, key, value):
    if not os.path.exists("rsi-states.json"):
        with open("rsi-states.json", "w") as f:
            json.dump([], f)
    else:
        with open("rsi-states.json", "r") as f:
            states = json.load(f)
        if not states or states[-1]["iteration"] != iteration:
            states.append({"iteration": iteration, "generate": "incomplete", "fine-tune": "incomplete", "eval": "incomplete"})
        if key and value:
            states[-1][key] = value
        with open("rsi-states.json", "w") as f:
            json.dump(states, f)

def rsi(N, iterations, num_evals, model, tokenizer, train_datasets: List[Tuple(Dataset, List)], eval_datasets, generate_args: Dict, train_args: Dict, eval_args: Dict):
    """
    datasets_dics: a dictionary. Key: data objects. Value: dataset of the corresponding data object. 
    generate_args: Dict
        - batch_size: default to 16
        - num_pathways: defualt to 32
        - method: default to "cot"
    train_args: Dict
        - training_args: default batch_size=16, num_train_epochs=3
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
            mixture = generate_training_dataset(N, model, tokenizer, train_datasets, **generate_args)
            with open(f'mixture/iter-{iter}.json', "w") as f:
                json.dump(mixture, f)
            update_rsi_states(iter, "generate", "complete")
            # fine tune
            fine_tune(f'mixture/iter-{iter}.json', model, tokenizer, **train_args)
            update_rsi_states("fine-tune", "complete")
            # eval
            if iter in eval_iters:
                metrics = evaluate(eval_datasets, model, tokenizer, **eval_args)
                performance.append((iter*N*len(train_datasets), metrics))
            update_rsi_states("eval", "complete")

    return performance

