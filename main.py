from pathlib import Path

import evaluate
import numpy as np
import torch
import transformers
from datasets import load_dataset
from peft import LoraConfig
from peft import get_peft_model
from peft import prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from transformers import BitsAndBytesConfig
from trl import DataCollatorForCompletionOnlyLM
from trl import SFTTrainer

from argument_parsing import function_calling_argument_parsing
from dataset.prompt_procession import formatting_prompts_func


def latest_file(file_list):
    return max(file_list, key=lambda x: x.stat().st_ctime)


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || "
        f"trainable%: {100 * trainable_params / all_param}"
    )


def main(total_train_steps, model_path, checkpoint_name, neftune_alpha, restore):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    repo_id = "mistralai/Mistral-7B-v0.1"
    tokenizer = AutoTokenizer.from_pretrained(repo_id)
    # Create a new token and add it to the tokenizer
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'

    model = AutoModelForCausalLM.from_pretrained(
        repo_id,
        quantization_config=bnb_config,
        device_map={"": 0},
    )
    model = prepare_model_for_kbit_training(model)

    config = LoraConfig(
        r=8,
        lora_alpha=8,
        target_modules=["q_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM"
    )

    peft_model = get_peft_model(model, config)
    print_trainable_parameters(peft_model)

    dataset_fc_v2_train = load_dataset("glaiveai/glaive-function-calling-v2", split='train[:-3%]')
    dataset_fc_v2_valid = load_dataset("glaiveai/glaive-function-calling-v2", split='train[-3%:]')

    ASSISTANT = 'ASSISTANT'
    response_template_ids = tokenizer.encode('\n' + ASSISTANT)[2:]

    em = evaluate.load("exact_match")

    def compute_metrics(eval_preds):
        predictions, labels = eval_preds
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)

        decoded_labels = []
        decoded_predictions = []

        for pred, sentence_label in zip(predictions, labels):
            decoded_labels.append(tokenizer.decode(sentence_label, skip_special_tokens=True))
            decoded_predictions.append(tokenizer.decode(pred, skip_special_tokens=True))

        return em.compute(predictions=decoded_predictions, references=decoded_labels)

    def preprocess_logits_for_metrics(logits, labels):
        prediction = logits.argmax(-1)
        return prediction

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'  # - No Flash Attention-2

    output_dir = Path(model_path) / checkpoint_name

    train_args = transformers.TrainingArguments(
        do_train=True,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=32,
        warmup_steps=total_train_steps // 2,
        max_steps=total_train_steps,
        learning_rate=2e-4,
        fp16=True,
        weight_decay=5e-5,
        logging_steps=2,
        save_steps=50,
        output_dir=output_dir,
        optim="paged_adamw_8bit",
        do_eval=True,
        do_predict=True,
        evaluation_strategy="steps",
        eval_steps=100,
        lr_scheduler_type='linear',
        report_to='tensorboard',
        gradient_checkpointing=True,
        #     group_by_length=True,
    )

    trainer = SFTTrainer(
        model=peft_model,
        max_seq_length=4096,
        train_dataset=dataset_fc_v2_train,
        eval_dataset=dataset_fc_v2_valid,
        tokenizer=tokenizer,
        formatting_func=formatting_prompts_func,
        args=train_args,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        #neftune_noise_alpha=neftune_alpha,
        data_collator=DataCollatorForCompletionOnlyLM(
            tokenizer=tokenizer,
            response_template=response_template_ids,
        )
    )

    #trainer.train(resume_from_checkpoint=restore)
    trainer.train()
    return


if __name__ == "__main__":
    args = function_calling_argument_parsing()

    main(
        total_train_steps=args.total_train_steps,
        model_path=args.model_path,
        checkpoint_name=args.checkpoint_name,
        neftune_alpha=args.neftune_alpha,
        restore=args.restore,
    )
