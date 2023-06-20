import logging, copy
import numpy as np
import torch
import torch.nn as nn
from collections import namedtuple
from torch.nn import CrossEntropyLoss, NLLLoss
from transformers import MT5ForConditionalGeneration, T5ForConditionalGeneration, BartConfig, BartModel, BartForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput, Seq2SeqLMOutput
from torch.nn.modules.activation import MultiheadAttention
from torch.nn.modules.container import ModuleList

logger = logging.getLogger(__name__)

class GenerativeModel(nn.Module):
    def __init__(self, config, tokenizer):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        logger.info(f'Loading pre-trained model {config.model_name}')
        if config.model_name.startswith("google/mt5-"):
            self.model = MT5ForConditionalGeneration.from_pretrained(config.model_name, cache_dir=config.cache_dir)
        elif config.model_name.startswith("t5-"):
            self.model = T5ForConditionalGeneration.from_pretrained(config.model_name, cache_dir=config.cache_dir)
        elif config.model_name.startswith("facebook/bart-") or config.model_name.startswith("bart-"):
            self.model = BartForConditionalGeneration.from_pretrained(config.model_name, cache_dir=config.cache_dir, config=BartConfig(vocab_size=50265, output_past=True))
        else:
            raise NotImplementedError
        # self.model.resize_token_embeddings(len(self.tokenizer))
        
    def forward(self, batch):
        outputs = self.model(input_ids=batch.enc_idxs, 
                             attention_mask=batch.enc_attn, 
                             decoder_input_ids=batch.dec_idxs, 
                             decoder_attention_mask=batch.dec_attn, 
                             labels=batch.lbl_idxs, 
                             return_dict=True)
        
        loss = outputs['loss']
        
        return loss
        
    def predict(self, batch, num_beams=1, max_length=100):
        self.eval()
        with torch.no_grad():
            if num_beams == 1:
                self.model._cache_input_ids = batch.enc_idxs
            else:
                expanded_return_idx = (
                    torch.arange(batch.enc_idxs.shape[0]).view(-1, 1).repeat(1, num_beams).view(-1).to(batch.enc_idxs.device)
                )
                input_ids = batch.enc_idxs.index_select(0, expanded_return_idx)
                self.model._cache_input_ids = input_ids
                
            outputs = self.model.generate(input_ids=batch.enc_idxs, 
                                          attention_mask=batch.enc_attn, 
                                          num_beams=num_beams, 
                                          max_length=max_length, 
                                          forced_bos_token_id=None)
            
            if self.config.model_name.startswith("facebook/bart-") or self.config.model_name.startswith("bart-"):
                outputs = self.model.generate(input_ids=batch.enc_idxs, 
                                            attention_mask=batch.enc_attn, 
                                            num_beams=num_beams, 
                                            max_length=max_length, 
                                            min_length=0,
                                            forced_bos_token_id=None)
        # decode outputs
        final_output = []
        for bid in range(len(batch.enc_idxs)):
            output_sentence = self.tokenizer.decode(outputs[bid], skip_special_tokens=True, clean_up_tokenization_spaces=True)
            final_output.append(output_sentence)
        self.train()

        return final_output

# for constrained decoding
class Prefix_fn_cls():
    def __init__(self, tokenizer, special_tokens, input_enc_idxs):
        self.tokenizer=tokenizer
        self.input_enc_idxs=input_enc_idxs
        self.special_ids = [element for l in self.tokenizer(special_tokens, add_special_tokens=False)['input_ids'] for element in l]
    def get(self, batch_id, previous_token):
        # get input
        inputs = list(set(self.input_enc_idxs[batch_id].tolist()))+self.special_ids
        return inputs
    