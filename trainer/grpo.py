import copy
import time

import torch
from progrich import ProgressBar, Spinner
from torch.utils.data import DataLoader
from transformers import BatchEncoding
from trl.trainer.utils import selective_log_softmax
from unsloth import FastModel

from config.grpo import GrpoImportanceWeight, GrpoScale
from dataset.batch import Batch, GroupedBatch
from dataset.prefill import prefix_completions_with_prefill
from dist import sync_dict_values
from metric.metrics import CLASS_ACCURACY, CLASS_ACCURACY_UNCASED, TRAIN_LOSS, Metric
from metric.tracker import MetricTracker
from reward import ClassificationReward, StructureReward
from trainer.utils import set_sampler_epoch

from .base import BaseTrainer
from .result import TrainOutput, TrainResult


class GrpoTrainer(BaseTrainer):
    """
    A Trainer for Group Relative Policy Optimisation (GRPO), an RL for any instruct
    model with simple reward functions instead of having a value model.

    The reward functions can be as simple as checking for the structure to be correct,
    assigning it an value of 1.0, or more complex, for example, how many paragraphs
    contain exactly four lines.

    As introduced by Deepseek-R1.
    """

    scale_rewards: GrpoScale
    importance_weight: GrpoImportanceWeight

    def __init__(
        self,
        *args,
        num_generations: int = 8,
        top_p: float = 0.92,
        temperature: float = 0.6,
        metrics: list[Metric] = [CLASS_ACCURACY, CLASS_ACCURACY_UNCASED, TRAIN_LOSS],
        reward_fns: list = [
            StructureReward(
                r"<reasoning>\n.*?\n</reasoning>",
                value=1.0,
                name="<reasoning>",
                max_count=1,
            ),
            StructureReward(
                r"<answer>\n.*?\n</answer>", value=1.0, name="<answer>", max_count=1
            ),
            ClassificationReward(),
        ],
        # Deepseek-R1 used 0.04, but that seems to be too high.
        kl_weight: float = 0.01,
        clip_range: float = 0.2,
        scale_rewards: GrpoScale = "std",
        importance_weight: GrpoImportanceWeight = "token",
        **kwargs,
    ):
        super().__init__(metrics=metrics, *args, **kwargs)
        self.num_generations = num_generations
        self.top_p = top_p
        self.temperature = temperature
        self.reward_fns = reward_fns
        self.kl_weight = kl_weight
        self.clip_range = clip_range
        self.scale_rewards = scale_rewards
        self.importance_weight = importance_weight

    @torch.no_grad()
    def generate_completions(self, batch: Batch, num: int = 1) -> GroupedBatch:
        """
        Generate completions for a batch to be used as training inputs.

        The same inputs are used for multiple generations.
        While some implementations try to do some elaborate batching, it is much
        simpler to just use the same as the training batch, but execute the whole
        batch multiple times.

        Inputs must be left padded!

        Note: This keeps the tokenised output, which can be used directly for the
        actual forward pass to the model containing the loss calculations.
        However, since the inputs might have padding and the output might have different
        amounts of padding, it could save memory when batching them after the fact.

        For example the following could occur:
            <pad><pad><pad>What is 2+2?<assistant>The result of the equation is 4.
            What is 2+2*10?<assistant>22<pad><pad><pad><pad>

        In this case, the first sample has 3 padding tokens in the beginning, because
        the original input was shorter, and the second one has 4 padding tokens at the
        end because the output is shorter. Instead of having to add padding to the
        second, the first one could be shifted to the left, which then only requires
        1 padding token at the end of the second sample. In essence, there should at
        least one sample with zero padding (the longest one).

        While this is clearly not ideal, it would be much more effort to retokenise
        everything, as the images would also need to be included again.

        Args:
            batch (Batch): Batch of inputs to generate completions. This is used like
                during inference, but allows to generate multiple completions for the
                group rewards.
            num (int): Number of completions to generate for each sample in the batch.
                [Default: 1]

        Returns:
            generated (GroupedBatch): Grouped batches from the completions.
        """
        # This is a CPU version of the batch data, which will be used for all the
        # newly generated batches, as they need to be stored on CPU until all of them
        # are generated. This is incredibly wasteful, but with the weird HF batch,
        # I can't think of any clean way to do this.
        # But also, the inputs need to stay on GPU for all generations.
        data_cpu = copy.deepcopy(batch.data)
        inputs = batch.data.to(self.hardware.device)
        prompt_len = inputs.input_ids.size(1)
        unwrapped_model = self.unwrap_model()
        tokeniser = self.unwrap_tokeniser()
        generated_batches: list[BatchEncoding] = []
        generated_completion_strs: list[list[str]] = []
        spinner = Spinner("Waiting for generation...")
        spinner.start()
        for i in range(num):
            outputs = unwrapped_model.generate(  # pyright: ignore[reportCallIssue]
                **inputs,
                max_new_tokens=self.max_new_tokens,
                pad_token_id=tokeniser.pad_token_id,
                do_sample=True,
                top_p=self.top_p,
                temperature=self.temperature,
            )
            # Putting onto CPU as the tokens will be used as a batch only after all the
            # generations have been completed.
            outputs = outputs.cpu()
            completion_ids = outputs[:, prompt_len:]
            completion_len = completion_ids.size(1)
            completion_padding = (completion_ids != tokeniser.pad_token_id,)
            # Extend the attention mask to ignore any padding create during the
            # generation.
            attn_mask = torch.cat(
                [
                    data_cpu.attention_mask,
                    completion_ids != tokeniser.pad_token_id,
                ],
                dim=-1,
            )
            # Now need the input batch (also on CPU) but modify its content, as there
            # are other parts that will be transferred, e.g. the pixel_values.
            # Need to copy the original batch as it will be modified in-place.
            # Very hacky, but it's easier than having to reload the image.
            new_data = data_cpu if i == num - 1 else copy.deepcopy(data_cpu)
            new_data["input_ids"] = outputs
            new_data["attention_mask"] = attn_mask
            if "cross_attention_mask" in new_data:
                cross_attn_mask = new_data["cross_attention_mask"]
                # Just to make pyright happy. HF types are just a mess.
                assert isinstance(cross_attn_mask, torch.Tensor)
                # Repeat the last values for the cross attention mask, as that should
                # continue for the remaining tokens.
                # See https://github.com/huggingface/transformers/blob/2e24ee4dfa39cc0bc264b89edbccc373c8337086/src/transformers/models/mllama/processing_mllama.py#L75-L78
                last_values = cross_attn_mask[:, -1:]
                added_mask = last_values.repeat(1, completion_len, 1, 1)
                # The padding should always be zero.
                added_mask[completion_padding] = 0
                new_data["cross_attention_mask"] = torch.cat(
                    [cross_attn_mask, added_mask],
                    dim=1,
                )
            # Remove the labels, as it makes no sense to want to predict what all the
            # generated answer produced, as not all of them will be correct.
            # TODO: Experiment with using labels, either for all, or to include the
            # correct answer in there and only include the labels for that.
            # i.e. combine normal fine-tuning with GRPO.
            new_data["labels"] = None
            # Convert the output to text, as most rewards will be calculated from the
            # text.
            completion_strs = [
                tokeniser.decode(comp, skip_special_tokens=True)
                for comp in completion_ids
            ]
            completion_strs = prefix_completions_with_prefill(
                completion_strs, prefill=self.prefill
            )
            generated_batches.append(new_data)
            generated_completion_strs.append(completion_strs)
            spinner.update(
                f"Generated completions [{i + 1} / {num}] • {completion_strs[0]}"
            )
        spinner.stop()
        return GroupedBatch.from_generations(
            batches=generated_batches,
            completions=generated_completion_strs,
            reference=batch,
        )

    # A bit annoying that this method needs to be redefined, just to add a loop across
    # the generations. But this is the cleanest way to integrate it otherwise with the
    # rest of the Trainer API.
    def train_epoch(self, data_loader: DataLoader, epoch: int) -> TrainResult:
        torch.set_grad_enabled(True)
        self.model.train()
        # Needed to revert the inference mode.
        FastModel.for_training(self.unwrap_model())
        num_replicas = set_sampler_epoch(data_loader, epoch=epoch)
        # Zeroing out the gradients here, because during the backward pass the zeroing
        # happens at the end, which saves the memory from it since the
        # zero_grad(set_to_none=True) (default) will eliminate the need to have the
        # gradients in memory, hence resetting them afterwards is beneficial.
        # But for the first step it needs to be done manually.
        self.optimiser.zero_grad()

        start_time = time.time()
        match self.scale_rewards:
            case "std":
                reward_scale = True
            case "max-len":
                reward_scale = self.max_new_tokens
            case "none":
                reward_scale = None
        metrics = MetricTracker(self.metrics, when="train")
        pbar = ProgressBar(
            "Train",
            total=len(data_loader.dataset),  # pyright: ignore[reportArgumentType]
            prefix=f"Epoch {epoch + 1} -",
            # Attach it to the total progress bar.
            progress=self.progress,
        )
        pbar.start()
        losses = []
        i = 0
        for batch in data_loader:
            i += 1
            # The last batch may not be a full batch
            curr_batch_size = batch.data["input_ids"].size(0)
            with self.hardware.autocast():
                generated = self.generate_completions(batch, num=self.num_generations)
            gen_metrics = MetricTracker(self.metrics, when="train")
            j = 0
            spinner = Spinner("Waiting for results of first batch of generations...")
            spinner.start()
            for gen_batch in generated.iter_batches(
                self.reward_fns, scale=reward_scale
            ):
                j += 1
                with self.hardware.autocast():
                    output = self.forward(gen_batch)
                self.backward(output.loss)
                gen_metrics.append(output.metrics)
                losses.append(output.loss.item())
                spinner.update(
                    f"Current Batch {i} [Generation: {j} / {self.num_generations}]: "
                    f"{gen_batch.data['input_ids'].size()} • "  # pyright: ignore[reportAttributeAccessIssue]
                    f"Loss {losses[-1]} • "
                    f"Avg loss {torch.mean(torch.tensor(losses, dtype=torch.float))}"
                )
            metrics.append(gen_metrics.mean())
            pbar.advance(curr_batch_size * num_replicas)
            spinner.stop()
        pbar.stop()

        mean_metrics = metrics.mean()
        # Gather the metrics onto the primary process
        mean_metrics = sync_dict_values(mean_metrics, device=self.hardware.device)
        return TrainResult(
            lr=self.get_lr(),
            metrics=mean_metrics,
            time_elapsed=time.time() - start_time,
        )

    # Per token log probs
    def _forward_log_probs(
        self, inputs: BatchEncoding, prompt_len: int | None = None
    ) -> torch.Tensor:
        outputs = self.model(**inputs)
        # Ignore the last logits, since that would be the next token (after the end).
        logits = outputs.logits[:, :-1]
        # The first token is not predicted.
        input_ids = inputs.input_ids[:, 1:]
        if prompt_len:
            logits = logits[:, prompt_len:]
            input_ids = input_ids[:, prompt_len:]
        # Log probs of each of the expected tokens.
        log_probs = selective_log_softmax(logits, input_ids)
        return log_probs

    def forward(self, batch: Batch) -> TrainOutput:
        unwrapped_model = self.unwrap_model()
        inputs = batch.data.to(self.hardware.device)
        advantages = torch.tensor(batch.info["advantages"], device=self.hardware.device)
        # All samples have the same prompt_len, so just use the first one.
        prompt_len = batch.info["prompt_len"][0]
        with torch.no_grad(), unwrapped_model.disable_adapter():  # pyright: ignore[reportCallIssue]
            # Log probs of the reference model, i.e. the base model, which can be
            # achieved by simply disabling the LoRA adapters.
            #
            # This is used to ensure that the model doesn't drift away too heavily
            # from the original model.
            ref_log_probs = self._forward_log_probs(inputs, prompt_len=prompt_len)

        # Log probs for the current model (includes the LoRA adapters) that is being
        # fine-tuned. As usual, this includes the gradients, so this is the regular
        # forward pass of the model.
        log_probs = self._forward_log_probs(inputs, prompt_len=prompt_len)

        # Normally in RL, the old model is kept for a few iterations before it is
        # updated. But since this updates it at every iteration, the same log probs
        # are used. Important: .detach() to avoid backpropagating through this, as
        # it is only supposed to be a static reference.
        # TODO: Implement the multiple iterations
        old_log_probs = log_probs.detach()

        # Unbiased estimator of KL-Divergence
        # http://joschu.net/blog/kl-approx.html
        log_diff = ref_log_probs - log_probs
        kl_divergence = torch.exp(log_diff) - log_diff - 1

        # Only keep the values for actual completion tokens (i.e. remove padding).
        # The +1 is due to the outputs being shifted by 1.
        completion_mask = inputs.attention_mask[:, prompt_len + 1 :]
        num_completion_tokens = torch.sum(completion_mask, dim=-1, keepdim=True)

        log_ratio = log_probs - old_log_probs
        match self.importance_weight:
            case "token":
                log_importance_weight = log_ratio
            case "sequence":
                # In order for the remaining calculations to work with the token
                # version, the reduced (token) dimension is kept as a singular
                # dimension, so that it is broadcast across all tokens.
                # NOTE: This uses a multiplication with the mask and the sum instead of
                # filterting with log_ratio[completion_mask] for the fact that this
                # acess would flatten the values into a single vector.
                # Also, this is faster than materialising the subset.
                log_importance_weight = (
                    torch.sum(log_ratio * completion_mask, dim=-1, keepdim=True)
                    / num_completion_tokens
                )

        advantage_coeff1 = torch.exp(log_importance_weight)
        advantage_coeff2 = torch.clamp(
            advantage_coeff1, 1 - self.clip_range, 1 + self.clip_range
        )
        advantage_term1 = advantage_coeff1 * advantages.unsqueeze(-1)
        advantage_term2 = advantage_coeff2 * advantages.unsqueeze(-1)

        # The loss is negation of the advantage objective
        loss = self.kl_weight * kl_divergence - torch.min(
            advantage_term1, advantage_term2
        )
        # The loss is only calculated for the completion tokens.
        # NOTE: This multiplication is slightly faster than accessing the masked
        # values with loss[completion_mask.to(torch.bool)].
        loss = torch.mean(
            torch.sum(loss * completion_mask, dim=-1, keepdim=True)
            / num_completion_tokens
        )

        return TrainOutput(
            loss=loss,
            metrics=dict(
                loss=loss.item(),
            ),
            info=batch.info,
        )
