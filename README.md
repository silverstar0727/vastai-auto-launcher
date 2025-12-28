# Intro
This is to-go pytorch template utilizing [lightning](https://github.com/Lightning-AI/lightning) and [wandb](https://github.com/wandb/wandb). 
This template uses `Lightning CLI` for config management. 
It follows most of [Lightning CLI docs](https://lightning.ai/docs/pytorch/latest/api_references.html#cli) but, integrated with `wandb`.
Since `Lightning CLI` instantiate classes on-the-go, there were some work-around while integrating `WandbLogger` to the template.
This might **not** be the best practice, but still it works and quite convinient.

# How To Use
It uses `Lightning CLI`, so most of its usage can be found at its [official docs](https://lightning.ai/docs/pytorch/latest/api_references.html#cli).  
There are some added arguments related to `wandb`.

* `--name` or `-n`: Name of the run, displayed in `wandb`
* `--version` or `-v`: Version of the run, displayed in `wandb` as tags

Basic cmdline usage is as follows.  
We assume cwd is project root dir.

### `fit` stage 
```bash
python src/main.py fit -c configs/config.yaml -n debug-fit-run -v debug-version
```
If using `wandb` for logging, change `"project"` key in `cli_module/rich_wandb.py`
If you want to access log directory in your `LightningModule`, you can access as follows.
```python
log_root_dir = self.logger.log_dir or self.logger.save_dir
```

### Clean Up Wandb Artifacts
If using `wandb` for logging, model ckpt files are uploaded to `wandb`.  
Since the size of ckpt files are too large, clean-up process needed.  
Clean-up process delete all model ckpt artifacts without any aliases (e.g. `best`, `lastest`)
To toggle off the clean-up process, add the following to `config.yaml`. Then every version of model ckpt files will be saved to `wandb`.
```yaml
trainer:
  logger:
    init_args:
      clean: false
```

### Model Checkpoint
One can save model checkpoints using `Lightning Callbacks`. 
It contains model weight, and other state_dict for resuming train.  
There are several ways to save ckpt files at either local or cloud.

1. Just leave everything in default, ckpt files will be saved locally. (at `logs/${name}/${version}/fit/checkpoints`)

2. If you want to save ckpt files as `wandb` Artifacts, add the following config. (The ckpt files will be saved locally too.)
```yaml
trainer:
  logger:
    init_args:
      log_model: all
```
3. If you want to save ckpt files in cloud rather than local, you can change the save path by adding the config. (The ckpt files will **NOT** be saved locally.)
```yaml
model_ckpt:
  dirpath: gs://bucket_name/path/for/checkpoints
```

#### `AsyncCheckpointIO` Plugins
You can set async checkpoint saving by providing config as follows.  
```yaml
trainer:
  plugins:
    - AsyncCheckpointIO
```



#### Automatic Batch Size Finder
Just add `BatchSizeFinder` callbacks in the config
```yaml
trainer:
  callbacks:
    - class_path: BatchSizeFinder
```
Or add them in the cmdline.
```bash
python src/main.py fit -c configs/config.yaml --trainer.callbacks+=BatchSizeFinder
```

##### NEW! `tune.py` for lr_find and batch size find
```bash
python src/tune.py -c configs/config.yaml
```
NOTE: No subcommand in cmdline

#### Resume
Basically all logs are stored in `logs/${name}/${version}/${job_type}` where `${name}` and `${version}` are configured in yaml file or cmdline. 
`{job_type}` can be one of `fit`, `test`, `validate`, etc.
  

### `test` stage
```bash
python src/main.py test -c configs/config.yaml -n debug-test-run -v debug-version --ckpt_path YOUR_CKPT_PATH
```




## Vast.ai Auto Training

Automatically select and launch GPU instances on Vast.ai for training.

### Features
- Auto-select optimal GPU instance based on price, reliability, and requirements
- Trigger training via PR comment (`/train`) in GitHub Actions
- Auto-terminate instance on training completion or failure
- Auto-download data from S3
- Integrated W&B logging

### Required Secrets (GitHub Actions)
Configure the following secrets in your GitHub repository settings:

| Secret | Required | Description |
|--------|----------|-------------|
| `VAST_API_KEY` | Yes | Vast.ai API key |
| `WANDB_API_KEY` | Yes | Weights & Biases API key |
| `AWS_ACCESS_KEY_ID` | No | AWS access key (for S3) |
| `AWS_SECRET_ACCESS_KEY` | No | AWS secret key (for S3) |
| `S3_DATA_PATH` | No | S3 data path (e.g., `s3://bucket/data`) |

### Usage

#### 1. GitHub Actions (PR Comment)
Comment `/train` on a PR to automatically start training.

```bash
# Basic execution
/train

# Specify config file
/train --config configs/my_config.yaml

# Pass extra arguments
/train --config configs/config.yaml --extra-args "--trainer.max_epochs 100"
```

#### 2. Local Execution
Build Docker image locally and run on Vast.ai.

```bash
# Set environment variables
export VAST_API_KEY=your_vastai_api_key
export WANDB_API_KEY=your_wandb_api_key

# Basic execution
./scripts/local_vastai.sh -u your_dockerhub_username

# Specify config file
./scripts/local_vastai.sh -u your_dockerhub_username -c configs/my_config.yaml

# Specify GPU type and max price
./scripts/local_vastai.sh -u your_dockerhub_username --gpu-type "RTX 3090" --max-price 1.5

# Dry run (search only, no instance creation)
./scripts/local_vastai.sh -u your_dockerhub_username --dry-run
```

#### 3. Direct Python Script
```bash
python scripts/vastai_launcher.py \
    --docker-image your_image:tag \
    --config configs/config.yaml \
    --gpu-type "RTX 4090" \
    --max-price 2.0
```

### Instance Auto-Termination
Vast.ai instances are automatically destroyed when:
- `WANDB_API_KEY` is not set (immediate termination)
- Training completes (success or failure)

### File Structure
```
scripts/
├── vastai_launcher.py  # Vast.ai instance selection and launch
├── local_vastai.sh     # Local build + Vast.ai launch
├── entrypoint.sh       # Docker container entrypoint
└── fit.sh              # Training execution script
```

## TODO
* Check pretrained weight loading
* Consider multiple optimizer using cases (i.e. GAN)
* Add instructions in README (on-going)
* Clean code
 