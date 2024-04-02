import torch
from torch.utils.data import Dataset
import pandas as pd
import os
from PIL import Image


class Flickr30kDataset(Dataset):
    def __init__(self, root_dir, split, image_processor=None, tokenizer=None):
        self.root_dir = root_dir
        self.split = split
        self.flickr_captions_df = pd.read_csv(
            os.path.join(root_dir, f"flickr30k-{split}.csv")
        )
        self.image_processor = image_processor
        self.tokenizer = tokenizer

    def __len__(self):
        return self.flickr_captions_df.shape[0]

    def __getitem__(self, index):
        img = Image.open(
            os.path.join(
                self.root_dir, "flickr30k-images",
                f"{self.flickr_captions_df.iloc[index, 0]}.jpg",
            )
        )
        if self.image_processor:
            img = self.image_processor(images=img, return_tensors="pt")
        caption = self.flickr_captions_df.iloc[index, 1].strip()
        if self.tokenizer:
            caption = self.tokenizer(caption, return_tensors="pt", max_length=128, padding="max_length", truncation=True)
        
        return {
            "pixel_values": img.pixel_values.squeeze(0),
            "labels": caption.input_ids.squeeze(0) if self.tokenizer else caption,
            "decoder_attention_mask": caption.attention_mask.squeeze(0) if self.tokenizer else None,
            }



class InstagramCaption(Dataset):
    def __init__(self, root_dir, image_processor=None, tokenizer=None):
        self.root_dir = root_dir
        self.instagram_captions_df = pd.read_csv(
            os.path.join(root_dir, f"captions_csv_cleaned.csv"),
            index_col=0,
        )
        self.image_processor = image_processor
        self.tokenizer = tokenizer

    def __len__(self):
        return self.instagram_captions_df.shape[0]

    def __getitem__(self, index):
        img = Image.open(
            os.path.join(
                self.root_dir,
                f"{self.instagram_captions_df.iloc[index, 0]}.jpg",
            )
        )
        if self.image_processor:
            # img = self.image_processor(images=img, return_tensors="pt")
            img = self.image_processor(img)
        caption = self.instagram_captions_df.iloc[index, 1].strip()
        if self.tokenizer:
            caption = self.tokenizer(caption, return_tensors="pt", padding="max_length", truncation=True)
        
        return img, caption
