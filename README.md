# Fine-tuning Vision Language Models (VLMs)

## Dependencies

Dependencies are managed by [uv][uv], a package/project manager for Python that is meant to bring a Cargo-style workflow
from Rust to Python. This includes managing the dependencies with a lockfile to ensure that all builds are reproducible.
While all of this was possible with virtual environments and specific tools, they always needed to be managed manually.
`uv` will automatically create a virtual environment for the project (in `.venv/`) and can then be used to run any
Python script in this environment without needing to manually activate the environment (but you can still do this if
you'd like to have everything available for your whole system).

You can install uv with a standalone installer (recommended), as described in the [Docs - Installing uv][uv-install],
or with pip.

```sh
# Prefer installing the standalone installer instead of installing it from pip
pip install --upgrade uv
```

Once installed, you can manage the dependencies through uv as follows:

```sh
# Installs all the dependencies and automatically creates the venv
uv sync
```

In order to run a binary from an installed package in the venv without activating it, you can use `uv run`. The same
applies to running Python from the venv with `uv run python` (just prefixing it with `uv run`). A Python file can be run
directly with `uv run some_script.py` (shorthand version of `uv run python some_script.py`).

```sh
# Run the train script from the venv
uv run train.py --help

# Run the ruff binary (installed as a dev dependency) to lint files
uv run ruff check
```

### ARM + CUDA Builds (GH200)

Unfortunately, most ML Python packages are not published for ARM + CUDA, so that means they mostly need to be built from
source. An odd exception is PyTorch, which does not publish the ARM + CUDA package to PyPI, but have it on their own
index. If you install PyTorch on ARM from PyPI, you'll always get the CPU only version.

While uv allows specifying dependencies based on the architecture, there are some annoyances for the compilation from
source for certain packages. Notably, they require certain build dependencies, which are also dependencies of the
project itself, and as uv does not prioritise certain dependencies, it would result in a *module not found* error.
Everything is taken care of by uv, as that was strictly defined in the list of dependencies
(even the `--no-build-isolation` is configured there), but it just means that you need to do it in multiple steps

To achieve this, there are dependency groups with the name `triton` and `full`, that need to be installed
in order and after the regular dependencies.
You need to compile triton first, because it requires torch during the build process, however when torch is also
installed with GPU support (which is configured in `pyproject.toml`), it will not work. Luckily, the custom registry
only applies to direct dependencies and not their dependencies, which means that triton will use the default torch
version for arm64, namely the CPU version that doesn't need triton.

```sh
# First install all pre-compiled packages (including build dependencies)
# By default it would install the `full` dependencies, so we need to disable the default groups.
uv sync --no-default-groups

# Then install the compile group triton to compile triton
# Use -v to see the compilation progress, otherwise it's just a spinner.
uv sync --no-default-groups --group triton -v

# Finally install the full version, to install the remaining dependencies.
# Use -v to see the compilation progress, otherwise it's just a spinner.
# The TORCH_CUDA_ARCH_LIST is needed for xformers who fails to infer the arch correctly.
TORCH_CUDA_ARCH_LIST="9.0;9.0a;9.0+PTX;9.0a+PTX" uv sync -v
```

Note: There are two issues with `xformers`, firstly, it tries to infer `TORCH_CUDA_ARCH_LIST` if it was not set, but
expects it to only contain numbers, whereas the Hopper GPUs also have `9.0a`, so converting that to a number fails. This
can be avoided, by setting the environment variable for the build. Secondly, some of the compilations time out for some
reason, which can be fixed by running it again.

Hence, `TORCH_CUDA_ARCH_LIST="9.0;9.0a;9.0+PTX;9.0a+PTX" uv sync -v` is used to compile it correctly. It might be
necessary to run multiple times to compile correctly.

## Faster Model Downloads

The *experimental* library [`hf-transfer`][hf-transfer] allows much greater download speeds for models (beyond 500MB/s) from the
HuggingFace Hub, but may be less stable. It is included in the dependencies, but you need to enable it by setting the
environment variable `HF_HUB_ENABLE_HF_TRANSFER`

For example, if you run any script that downloads the model, you could just prepend it with the environment variable
just for that command:

```sh
HF_HUB_ENABLE_HF_TRANSFER=true uv run [...]
```

## Training

Training is done with the `train.py` script:

```sh
uv run train.py --train-data path/to/train.tsv --validation-data path/to/validation.tsv --prompts path/to/prompts.json -m unsloth/Qwen2-VL-7B-Instruct --name rvl-cdip-qwen2-r16 -b 2
```

- `--train-data` and `--validation-data` can either be a TSV file listing all JSON files that should be used for
  training / validation, respectively, or a directory, in which case all JSON files inside that directory are included.
- `-p` / `--prompts` is a JSON file containing system and/or question prompts that will be selected randomly during
  training to get some variations. This is necessary if individual samples don't provide a system prompt or question.
  When the system prompt or question would always be the same across all samples, it only needs to be specified once in
  the `prompt.json` rather than for each sample.
- `-m` / `--model` specifies which model to use. Any HuggingFace compatible model can be loaded, as long as unsloth
  supports them (all derivatives of popular main models). Prefer to use the instruct models.
- `--name` is used to give the experiment a name, which is used to store the checkpoints in `log/<name>` as well as the name of the experiment in [Weights & Biases][wandb].
- `-b` / `--batch-size` determines the number of documents to use per batch.
- `-r` / `--rank` defines the rank of LoRA (default is 16).

### Reinforcement Learning (RL) with GRPO

> [!NOTE]
> Enable it with the option `--trainer grpo`
>
> You may need to reduce the learning compared to the one used for SFT. For reference, for the RVL-CDIP classification
> a learning rate of *5e-4* worked well for SFT, whereas for GRPO it needed to be reduced to *2e-5*.

In addition to the regular supervised fine-tuning (SFT), you can also train a model with reinforcement learning (RL),
specifically with Group Relative Policy Optimisation (GRPO), which was popularised by Deepseek-R1. In this mode, at each
iteration you sample a given number of outputs from the model (default 8, can be changed with `--num-generations`) and
calculates the rewards based on certain verifiable reward functions, such as structure of containing
a `<reasoning>...</reasoning>` tag, as well as a simple check for the correct answer of the classification task.

Due to generating multiple responses from the model for each batch, as well as having to back propagate through all of
them, this takes much longer to train than SFT. However, you gain possible reasoning traces that can help for
explainability as well as often being better a generalising to out of distribution data, including unseen classes.

While it may not always be superior, it is certainly worth trying once you have established a solid baseline with SFT.
As it takes much longer to train, it is recommended to start with SFT in order to ensure the training works as expected.

If you want to continue training a model with GRPO from an SFT checkpoint, you may notice that the model never produces
the expected reasoning tags. Firstly, make sure to include that in the prompt, but if the model fails to follow that,
because it was trained to give a straight answer from the SFT, you can force the model to begin with a reasoning tag by
prefilling the answer. This can be achieved with the option `--prefill <reasoning>`, or whatever you want to prefill the
answer with.

### Multi-GPU

Training with multiple GPUs can be achieved by using [`torchrun`][torchrun] when launching the script. Instead of using
`uv run train.py` you can run `uv run torchrun [torchrun-options] train.py` instead, where the torchrun flags need to
come before the `train.py`, otherwise it is considered to be an option to the train script.

```sh
uv run torchrun --standalone --nproc-per-node=2 train.py --train-data path/to/train.tsv --validation-data path/to/validation.tsv --prompts path/to/prompts.json -m unsloth/Qwen2-VL-7B-Instruct --name rvl-cdip-qwen2-r16 -b 2
```

You need to specify how many GPUs you are using with `--nproc-per-node`, which in this example is 2.

`torchrun` is a binary shipped with torch, that should be available in the virtual environment, so uv should
automaticaly discover it when using `uv run`, if there is any issue with it, it is also possible to run it with `uv run
python -m torch.distributed.run` instead.

*Important: This loads the model on every GPU and runs them in parallel (Data parallelism), so the model needs to fit
into a single GPU.*


### Data Format

The data is given as JSON files, which contains at the very least the path to the image and the answer that is expected
from the LLM. Additionally, there can be a system prompt (fully optional) and a question, which is used as the user
input that is provided alongside the image. A question is *required*, but both of these can be specified with the
`prompts.json` (as shown below).

Minimal example for an image classification:

```json
{
  "image": "images/train_0834_6.png",
  "answer": "scientific publication"
}
```

and with question and system prompt:

```json
{
  "image": "images/train_0834_6.png",
  "answer": "scientific publication",
  "system": "You are an expert in document classification.",
  "question": "What type of document is depicted in the image?"
}
```

*Note: Not all models support system prompts, for example Llama3.2-Vision does not accept a system prompt in combination
with images (for some reason). In that case, you need to merge it into the question (as this is allowed).*

#### Prompts

For tasks that always use the same system prompt / question, such as in image classification, it would be unnecessary to
add these prompts to each individual sample, hence it is possible to provide a JSON file to `-p` / `--prompts`, which
will be used for all samples.

Additionally, it may be beneficial to have variation during the training, rather than being stuck with the same prompt.
Both `system` and `question` must be a list of prompts, from which one is sampled randomly for each data point during
training, and during validation the first one will always be used.

Example prompts with a single choice (add additional prompts to get the variations):

```json
{
  "system": [
    "You are a helpful assistant that classifies various types of documents based on their images. Respond with only the chosen type of document from the following list of document types:\n\n- letter\n- form\n- email\n- handwritten\n- advertisement\n- scientific report\n- scientific publication\n- specification\n- file folder\n- news article\n- budget\n- invoice\n- presentation\n- questionnaire\n- resume\n- memo"
  ],
  "question": [
    "What type of document is depicted in the image?"
  ]
}
```

[hf-transfer]: https://github.com/huggingface/hf_transfer
[torchrun]: https://pytorch.org/docs/stable/elastic/run.html
[uv]: https://github.com/astral-sh/uv
[uv-install]: https://docs.astral.sh/uv/getting-started/installation/
