import argparse
import torch
import datasets
import matplotlib.pyplot as plt
import random
import os

from transformers import AutoImageProcessor, AutoTokenizer, VisionEncoderDecoderModel

def parse_args():
    parser = argparse.ArgumentParser(description="Script to parse command-line arguments")
    parser.add_argument("--data_dir", type=str, help="Path to the data directory")
    parser.add_argument("--output_directory", type=str, help="Output directory for saving model runs")
    parser.add_argument("--experiment_name", type=str, help="Name of the experiment")
    parser.add_argument("--encoder", type=str, help="Encoder model name or path")
    parser.add_argument("--decoder", type=str, help="Decoder model name or path")
    parser.add_argument("--device", type=str, help="GPU or CPU", default="gpu")
    parser.add_argument("--dataloader_num_workers", type=int, default=32, help="Number of subprocesses to use for data loading")

    args = parser.parse_args()

    args.experiment_path = os.path.join(
        args.output_directory, args.encoder.split("/")[-1]+"@"+args.decoder.split("/")[-1], args.experiment_name)
    if not os.path.exists(args.experiment_path):
        raise(f"Experiment path ({args.experiment_path}) does not exist")
    
    return args


def get_device(device_type):
    if device_type == "gpu" and torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

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

def load_test_set(args):
    image_processor = get_pre_processor(args)
    global tokenizer
    tokenizer = get_tokenizer(args)
    tokenizer = update_tokenizer(tokenizer, args)

    test_dataset = datasets.FlickrDataset(args.data_dir, split="test", image_processor=image_processor)
    return test_dataset

def load_model(args):
    # Load the trained model
    model = VisionEncoderDecoderModel.from_pretrained(args.experiment_path)
    return model

def generate_predictions(model, test_dataset, num_samples=5):
    # Generate captions for a random sample of test set images
    samples = random.sample(range(len(test_dataset)), num_samples)
    captions = []

    for idx in samples:
        example = test_dataset[idx]
        image = example['pixel_values'].unsqueeze(0)
        org_image = example['org_img']
        caption = example['labels']

        # Forward pass through the model to generate captions
        outputs = model.generate(image)

        # Decode the generated caption
        predicted_caption = tokenizer.decode(outputs[0], skip_special_tokens=True)

        captions.append({'image': org_image, 'actual_caption': caption, 'predicted_caption': predicted_caption})

    return captions

def visualize_samples(samples, args):
    # Visualize sample images with their actual and generated captions
    fig, axes = plt.subplots(nrows=len(samples), ncols=1, figsize=(8, 12))

    for i, sample in enumerate(samples):
        image = sample['image']
        actual_caption = sample['actual_caption']
        predicted_caption = sample['predicted_caption']

        # Plot the image
        axes[i].imshow(image.squeeze(0).permute(1, 2, 0))
        axes[i].axis('off')

        # Add actual and predicted captions as titles
        axes[i].set_title(f"Actual: {actual_caption}\nPredicted: {predicted_caption}", fontsize=10)

    plt.tight_layout()
    plt.savefig(args.experiment_path+"/predictions.png")

if __name__ == "__main__":
    args = parse_args()
    global device
    device = get_device(args.device)

    test_dataset = load_test_set(args)
    model = load_model(args)
    model = model.to(device)

    samples = generate_predictions(model, test_dataset)
    visualize_samples(samples, args)
