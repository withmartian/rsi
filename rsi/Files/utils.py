import argparse, os, json

def str_to_bool(s):
    if s.lower() in ['true', 't', 'yes', 'y', '1']:
        return True
    elif s.lower() in ['false', 'f', 'no', 'n', '0']:
        return False
    else:
        raise argparse.ArgumentTypeError(f'Invalid Boolean value: {s}')
    
def get_checkpoint_states(checkpoint_dir, resume_from_checkpoint, iteration):
  states = {"iteration": iteration, "completed_datasets": []}
  if not os.path.exists(f'{checkpoint_dir}/states.json'):
    if not os.path.exists(f'{checkpoint_dir}'):
      os.mkdir(f'{checkpoint_dir}')
    with open(f'{checkpoint_dir}/states.json', "w") as f:
      json.dump(states, f)
  if resume_from_checkpoint:
    with open(f'{checkpoint_dir}/states.json', "r") as f:
        states = json.load(f)
  return states