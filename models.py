import torch
import timm

import torch.nn as nn

from transformers import AutoModel
from transformers import BertModel
from transformers import GPT2Model

class ImageEncoder(nn.Module):

    def __init__(self, model_name, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.model = AutoModel.from_pretrained(model_name)

    def forward(self, x: dict):
        x = model(**x, return_dict=True)
        return x

class BERTCaptionDecoder(nn.Module):

    def __init__(self, model_name, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.model = BertModel.from_pretrained(model_name)

    def forward(self, encoded_input):
        x = model(**encoded_input)
        return x
    
class GPTCaptionDecoder(nn.Module):

    def __init__(self, model_name, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.model = GPT2Model.from_pretrained(model_name)

    def forward(self, encoded_input):
        x = model(**encoded_input)
        return x

class ImageCaptioningModel(nn.Module):
    def __init__(self, encoder_model, decoder_model, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.encoder = ImageEncoder(encoder_model)
        if "gpt" in decoder_model:
            self.decoder = GPTCaptionDecoder(decoder_model)
        else:
            self.decoder = BERTCaptionDecoder(decoder_model)
        self.projection = nn.Linear(self.encoder.config.hidden_size, self.decoder.config.hidden_size)
        
    def forward(self, x):
        embeddings = self.encoder(x)
        captions = self.decoder(embeddings)
        output = self.fc(captions.pooler_output)
        return output

# Example usage
model = ImageCaptioningModel(num_classes=1000)
input_tensor = torch.randn(1, 3, 224, 224)
output = model(input_tensor)
print(output.shape)

