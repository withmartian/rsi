def generate_N_training_entries(model, tokenizer, N, datasets, batch_size=8, num_pathways=32, checkpoint_path='', save_every=500, resume_from_checkpoint=False):
    for dataset in datasets:
        dataset.



def generate_mixture_inner_loop(curr_mixture, N, dset, bs, num_pathways, last_saved, save_every):
    while len(curr_mixture) < N:
    #   batch_samples = get_batch_samples(dset, bs)
      result = get_pathways(dset, batch_samples, batch_size=bs, num_examples=len(batch_samples), num_pathways=num_pathways)
      curr_mixture.extend(generate_mixture(dset, batch_samples, result))
      if len(curr_mixture) // save_every > last_saved:
        last_saved += 1
        with open(f'mixture-checkpoint/{dset}-checkpoint.json', "w") as f:
          json.dump(curr_mixture, f)
    return curr_mixture
