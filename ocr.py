import os
from pathlib import Path

import torch
from progrich import ProgressBar
from torch.utils.data import DataLoader
from unsloth import FastModel

from config.ocr import OcrConfig
from dataset import InstructDataset
from dataset.collate import InstructCollator
from model.utils import unwrap_tokeniser


@torch.inference_mode()
def main():
    cfg = OcrConfig.parse_config()
    torch.manual_seed(cfg.hardware.seed)

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    hardware_manager = cfg.hardware.create_manager()
    model, processor = FastModel.from_pretrained(
        cfg.model,
        load_in_4bit=False,
        # Set this to get the full precision model, otherwise unsloth just decides to
        # use a 4bit version of QLoRA.
        full_finetuning=True,
    )
    tokeniser = unwrap_tokeniser(processor)
    tokeniser.padding_side = "left"
    # model = model.to(hardware_manager.device)
    model = model.eval()
    FastModel.for_inference(model)
    dataset = InstructDataset(
        cfg.data,
        processor=processor,
        prompts=cfg.prompts,
        random_prompt_probability=0.0,
        first_prompt_only=True,
        image_resizer=cfg.image.create_resizer(),
    )
    collator = InstructCollator(processor=processor, include_answer=False)
    data_loader = DataLoader(
        dataset,
        batch_size=cfg.hardware.batch_size,
        num_workers=cfg.hardware.calculate_num_workers(),
        shuffle=False,
        pin_memory=not cfg.hardware.no_pin_memory and hardware_manager.is_cuda(),
        collate_fn=collator,
    )
    model_path = Path(cfg.model)
    cp_name = model_path.name
    out_dir = cfg.out_dir / cp_name
    out_dir.mkdir(parents=True, exist_ok=True)

    with ProgressBar("Extracting Text (OCR)", total=len(dataset), persist=True) as pbar:
        for batch in data_loader:
            # The last batch may not be a full batch
            curr_batch_size = batch.data["input_ids"].size(0)
            inputs = batch.data.to(hardware_manager.device)
            with hardware_manager.autocast():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=cfg.max_new_tokens,
                    pad_token_id=tokeniser.pad_token_id,
                )

            preds = [
                tokeniser.decode(out[input_ids.size(0) :], skip_special_tokens=True)
                for input_ids, out in zip(inputs.input_ids, outputs)
            ]
            for pred, path in zip(preds, batch.info["path"]):
                with open(
                    out_dir / path.with_suffix(".md").name, "w", encoding="utf-8"
                ) as fd:
                    fd.write(pred)
            pbar.advance(curr_batch_size)


if __name__ == "__main__":
    main()
