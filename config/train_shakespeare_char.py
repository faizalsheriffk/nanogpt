# train a miniature character-level shakespeare model
# good for debugging and playing on macbooks and such

out_dir = 'out-shakespeare-char'
eval_interval = 250 # keep frequent because we'll overfit
eval_iters = 20 # adjusted it to work on CPU
log_interval = 1 # don't print too too often

# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = False

wandb_log = False # override via command line if you like
wandb_project = 'shakespeare-char'
wandb_run_name = 'mini-gpt'

dataset = 'shakespeare_char'
gradient_accumulation_steps = 1
batch_size = 12 # 12 examples per iterations
block_size = 64 # context of up to 64 previous characters

# baby GPT model that can run on mac:)
n_layer = 4
n_head = 4
n_embd = 128
dropout = 0.0  # go easy on regularization

learning_rate = 1e-3 # with baby networks can afford to go a bit higher
max_iters = 2000 # reduced iterations
lr_decay_iters = 2000 # make equal to max_iters usually
min_lr = 1e-4 # learning_rate / 10 usually
beta2 = 0.99 # make a bit bigger because number of tokens per iter is small

warmup_iters = 100 # not super necessary potentially

# on macbook also add
device = 'cpu'  # run on cpu only
compile = False # do not torch compile the model
