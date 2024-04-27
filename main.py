import argparse
import datasets
import os
import torch

from transformers import AutoImageProcessor, AutoTokenizer, AutoModel, AutoConfig
from transformers import VisionEncoderDecoderConfig, VisionEncoderDecoderModel
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from transformers import default_data_collator
from nltk.translate.bleu_score import corpus_bleu

def parse_args():
    parser = argparse.ArgumentParser(description="Script to parse command-line arguments")
    parser.add_argument("--data_dir", type=str, help="Path to the data directory")
    parser.add_argument("--output_directory", type=str, help="Output directory for saving model runs")
    parser.add_argument("--experiment_name", type=str, help="Name of the experiment")
    parser.add_argument("--encoder", type=str, help="Encoder model name or path")
    parser.add_argument("--decoder", type=str, help="Decoder model name or path")
    parser.add_argument("--device", type=str, help="GPU or CPU", default="gpu")
    parser.add_argument("--lr", type=float, help="Learning rate for training")
    parser.add_argument("--lr_scheduler_step_size", type=int, help="Step size for learning rate scheduler")
    parser.add_argument("--lr_scheduler_gamma", type=float, help="Gamma for learning rate scheduler")
    parser.add_argument("--per_device_train_batch_size", type=int, default=32, help="Batch size per device for training")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=32, help="Batch size per device for evaluation")
    parser.add_argument("--dataloader_num_workers", type=int, default=32, help="Number of subprocesses to use for data loading")
    parser.add_argument("--num_train_epochs", type=int, default=10, help="Total number of training epochs to perform")
    parser.add_argument("--logging_steps", type=int, default=100, help="Log every N updates steps")
    parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every N updates steps")
    parser.add_argument("--save_total_limit", type=int, default=2, help="Limit the total amount of checkpoints")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of updates steps to accumulate before performing a backward/update pass")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay to apply")


    args = parser.parse_args()

    args.experiment_path = os.path.join(
        args.output_directory, args.encoder.split("/")[-1]+"@"+args.decoder.split("/")[-1], args.experiment_name)
    if not os.path.exists(args.experiment_path):
        os.makedirs(args.experiment_path)
    args.log_directory = os.path.join(args.experiment_path, "logs")
    if not os.path.exists(args.log_directory):
        os.makedirs(args.log_directory)
    return args

def get_device(device_type):
    if device_type == "gpu" and torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")
    

def get_model(args):
    # Load pre-trained encoder and decoder models
    # encoder_model_config = AutoConfig.from_pretrained(args.encoder)
    # encoder_model = AutoModel.from_pretrained(args.encoder, config=encoder_model_config)

    # decoder_model_config = AutoConfig.from_pretrained(args.decoder)
    # decoder_model_config.add_cross_attention=True
    # decoder_model = AutoModel.from_pretrained(args.decoder, config=decoder_model_config)
    
    
    # config = VisionEncoderDecoderConfig.from_encoder_decoder_configs(encoder_model_config, decoder_model_config)
    # config.add_cross_attention=True
    # model = VisionEncoderDecoderModel(config=config, encoder=encoder_model, decoder=decoder_model)
    
    model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(
        args.encoder, args.decoder, return_dict=True
    )
    
    
    print(model.decoder.config.add_cross_attention)
    print(model.config.add_cross_attention)
    
    return model

def get_pre_processor(args):
    image_processsor = AutoImageProcessor.from_pretrained(args.encoder)
    return image_processsor

tokenizer = None
def get_tokenizer(args):
    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.decoder)
    return tokenizer

def update_tokenizer(tokenizer, args):
    if "gpt" in args.decoder:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    elif "bert" in args.decoder:
        pass
    return tokenizer

def update_tokens(model):
    model.config.decoder_start_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.eos_token_id

    model.generation_config.pad_token_id = model.generation_config.eos_token_id
    
    model.decoder.resize_token_embeddings(len(tokenizer))
    return model

def check_tokens(model):
    print("tokenizer_bos_token_id:", tokenizer.bos_token_id)
    print("tokenizer_pad_token_id:", tokenizer.pad_token_id)
    print("tokenizer_eos_token_id:", tokenizer.eos_token_id)
    
    print("decoder_start_token_id:", model.config.decoder_start_token_id)
    print("pad_token_id:", model.config.pad_token_id)
    print("eos_token_id:", model.config.eos_token_id)
    print("model.generation_config.eos_token_id", model.generation_config.eos_token_id)
    

def get_datasets(args):
    image_processor = get_pre_processor(args)
    global tokenizer
    tokenizer = get_tokenizer(args)
    tokenizer = update_tokenizer(tokenizer, args)

    train_dataset = datasets.FlickrDataset(args.data_dir, split="train",
                                          image_processor=image_processor, tokenizer=tokenizer)
    val_dataset = datasets.FlickrDataset(args.data_dir, split="val",
                                        image_processor=image_processor, tokenizer=tokenizer)
    return train_dataset, val_dataset


def compute_bleu(predictions, references):
    pred_texts = [tokenizer.decode(pred, skip_special_tokens=True, clean_up_tokenization_spaces=True) for pred in predictions]
    ref_texts = [[tokenizer.decode(ref, skip_special_tokens=True, clean_up_tokenization_spaces=True)] for ref in references]

    bleu_score = corpus_bleu(ref_texts, pred_texts)
    return bleu_score


def train(args):

    model = get_model(args)
    model = model.to(device)
    train_dataset, val_dataset = get_datasets(args)
    model = update_tokens(model)
    check_tokens(model)

    # Define training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_directory,
        do_train=True,
        do_eval=True,
        evaluation_strategy="steps",
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        dataloader_num_workers=args.dataloader_num_workers,
        num_train_epochs=args.num_train_epochs,
        logging_dir=args.log_directory,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.lr,
        lr_scheduler_type='linear',
        warmup_steps=0,
        # lr_scheduler_step_size=args.lr_scheduler_step_size,
        # lr_scheduler_decay_power=args.lr_scheduler_gamma,
        weight_decay=args.weight_decay,
        predict_with_generate=True,
        overwrite_output_dir=True,
    )

    # Define trainer object
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        # tokenizer=tokenizer,
        data_collator=default_data_collator,  # You can define your own data collator if needed
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=None,  # You can define your own evaluation metrics function if needed
    )

    # Train the model
    trainer.train()

if __name__ == "__main__":
    args = parse_args()
    global device
    device = get_device(args.device)
    print(device)
    print(args)
    train(args)
