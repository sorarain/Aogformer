import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.roberta.modeling_roberta import RobertaLayer,RobertaEncoder
from transformers import RobertaForMaskedLM#RobertaSelfAttention is modified
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions
from transformers.utils import logging
from typing import Optional, Tuple, Union
from transformers import RobertaTokenizerFast

from scripts.AOG import AOG_Building_Block

logger = logging.get_logger(__name__)

class AOGEncoder(RobertaEncoder):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        # self.layer = nn.ModuleList([RobertaLayer(config) for _ in range(config.num_hidden_layers)])#now use
        self.layer = nn.ModuleList([AOG_Building_Block(config.num_blocks, config) for _ in range(config.num_hidden_layers)])#try new(only aog)
        self.num_aog_block = 4
        self.aoglayers = nn.ModuleList([AOG_Building_Block(config.num_blocks, config) for _ in range(config.num_hidden_layers // self.num_aog_block)])
        # self.extern_aog_token = nn.Embedding(1,config.hidden_size)
        self.num_blocks = config.num_blocks
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPastAndCrossAttentions]:
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        next_decoder_cache = () if use_cache else None

        ################
        __hidden_states = hidden_states
        bz, sql, hz = hidden_states.size()
        # assert sql % self.num_blocks == 0,"error can't blocks"

        # self.per_token = sql // self.num_blocks
        self.per_token = 64#512
        self.num_blocks = sql // self.per_token
        assert sql % self.per_token == 0,"error can't blocks"


        self.max_length = sql
        hidden_states = hidden_states.reshape([bz, self.num_blocks, self.per_token, hz]).contiguous()
        extern_aog_token = torch.zeros([bz, self.num_blocks, 1, hz],device=hidden_states.device,dtype=hidden_states.dtype)#now use&try new(only aog)
        # extern_aog_token = self.extern_aog_token(torch.zeros([bz, self.num_blocks,1],device=hidden_states.device,dtype=torch.int64))#try new(v2_text8)
        if attention_mask is not None:
            attention_mask_size = list(attention_mask.size())
            assert len(attention_mask.size()) == 4,f"{attention_mask.size()}"
            attention_mask_size[2]+=1
            attention_mask_size[3]+=1
            # new_mask = torch.zeros(attention_mask_size,device=attention_mask.device)
            new_mask = torch.ones(attention_mask_size,device=attention_mask.device)
            new_mask[:,:,1:,1:] = attention_mask
            attention_mask = new_mask
        ################

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                ################
                # all_hidden_states = all_hidden_states + (hidden_states.reshape([bz, self.num_blocks * self.per_token, hz]),)
                all_hidden_states = all_hidden_states + (hidden_states,)
                ################

            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    layer_module.__call__,
                    torch.cat([extern_aog_token,hidden_states],dim=2).reshape([bz * self.num_blocks, self.per_token + 1, hz]).contiguous(),
                    # hidden_states.reshape([bz * self.num_blocks, self.per_token, hz]).contiguous(),
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                )
            else:
                # layer_outputs = layer_module(
                #     torch.cat([extern_aog_token,hidden_states],dim=2).reshape([bz * self.num_blocks, self.per_token + 1, hz]).contiguous(),
                #     # hidden_states.reshape([bz * self.num_blocks, self.per_token, hz]).contiguous(),
                #     attention_mask,
                #     layer_head_mask,
                #     encoder_hidden_states,
                #     encoder_attention_mask,
                #     past_key_value,
                #     output_attentions,
                # )#now use & try new(v2_text8)
                layer_outputs = layer_module(
                    torch.cat([extern_aog_token,hidden_states],dim=2).reshape([bz * self.num_blocks, self.per_token + 1, hz]).contiguous(),
                )#try new(only aog)
                layer_outputs = (layer_outputs, )#try new(only aog)
            # hidden_states = layer_outputs[0][:, :-1, :].reshape([bz, self.num_blocks, self.per_token, hz]).contiguous()#now use
            hidden_states = layer_outputs[0][:, 1:, :].reshape([bz, self.num_blocks, self.per_token, hz]).contiguous()#try new(v2_text8)&try new(only aog)
            # hidden_states = layer_outputs[0]
            ################
            if i % self.num_aog_block == 0:
                pass
                extern_aog_token = self.aoglayers[i // self.num_aog_block](hidden_states.mean(dim=2).reshape([bz, self.num_blocks, hz])).reshape([bz, self.num_blocks, 1, hz])#now use&try new(only aog)
                # extern_aog_token = self.aoglayers[i // self.num_aog_block](layer_outputs[0][:, 0, :].reshape([bz, self.num_blocks, hz])).reshape([bz, self.num_blocks, 1, hz])#try new(v2_text8)
                
                # hidden_states = hidden_states.reshape([bz, self.num_blocks, self.per_token + i // self.num_aog_block, hz])
                # hidden_states = torch.cat([tmp, hidden_states],dim=2).reshape([bz * self.num_blocks, self.per_token + i // self.num_aog_block + 1, hz])
            ################
            if use_cache:
                print(layer_outputs[-1].size())
                next_decoder_cache += (layer_outputs[-1],)
            if output_attentions:
                print(f"layer_outputs[1] size {layer_outputs[1].size()}")
                print(f"layer_outputs[2] size {layer_outputs[2].size()}")
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        ################
        hidden_states = hidden_states.reshape([bz, sql, hz])
        ################
        # return hidden_states

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_decoder_cache,
                    all_hidden_states,
                    all_self_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=hidden_states,#all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )

class RobertaAOGForMaskedLM(RobertaForMaskedLM):
    def __init__(self, config):
        super().__init__(config)
        self.roberta.encoder = AOGEncoder(config)


def pad_to_window_size(input_ids: torch.Tensor,
                       one_sided_window_size: int, pad_token_id: int):
    '''A helper function to pad tokens and mask to work with the sliding_chunks implementation of Longformer selfattention.
    Input:
        input_ids = torch.Tensor(bsz x seqlen): ids of wordpieces
        attention_mask = torch.Tensor(bsz x seqlen): attention mask
        one_sided_window_size = int: window size on one side of each token
        pad_token_id = int: tokenizer.pad_token_id
    Returns
        (input_ids, attention_mask) padded to length divisible by 2 * one_sided_window_size
    '''
    w = int(2 * one_sided_window_size)
    seqlen = input_ids.size(1)
    padding_len = (w - seqlen % w) % w
    input_ids = F.pad(input_ids, (0, padding_len), value=pad_token_id)
    return input_ids


def create_aog_model(save_model_to, attention_window, max_pos, num_blocks):
    model = RobertaForMaskedLM.from_pretrained('/root/autodl-tmp/users/hyb/longformer/roberta-base')
    tokenizer = RobertaTokenizerFast.from_pretrained('/root/autodl-tmp/users/hyb/longformer/roberta-base', model_max_length=max_pos)
    config = model.config
    config.num_blocks = num_blocks

    # extend position embeddings
    tokenizer.model_max_length = max_pos
    tokenizer.init_kwargs['model_max_length'] = max_pos
    current_max_pos, embed_size = model.roberta.embeddings.position_embeddings.weight.shape
    max_pos += 2  # NOTE: RoBERTa has positions 0,1 reserved, so embedding size is max position + 2
    config.max_position_embeddings = max_pos
    assert max_pos > current_max_pos
    # allocate a larger position embedding matrix
    new_pos_embed = model.roberta.embeddings.position_embeddings.weight.new_empty(max_pos, embed_size)
    # copy position embeddings over and over to initialize the new position embeddings
    k = 2
    step = current_max_pos - 2
    while k < max_pos - 1:
        new_pos_embed[k:(k + step)] = model.roberta.embeddings.position_embeddings.weight[2:]
        k += step
    model.roberta.embeddings.position_embeddings.weight.data = new_pos_embed
    model.roberta.embeddings.position_ids.data = torch.tensor([i for i in range(max_pos)]).reshape(1, max_pos)

    # replace the `modeling_bert.BertSelfAttention` object with `LongformerSelfAttention`
    aogencoder = AOGEncoder(config)
    config.attention_window = [attention_window] * config.num_hidden_layers
    config.add_t = False
    for layer, aogencoderlayer, aoglayer in zip(model.roberta.encoder.layer,aogencoder.layer, aogencoder.aoglayers):
        aogencoderlayer.attention.query = layer.attention.self.query
        aogencoderlayer.attention.key = layer.attention.self.key
        aogencoderlayer.attention.value = layer.attention.self.value
        # aogencoderlayer.attention.output.dense = layer.attention.output.dense

        # aoglayer.attn.query = copy.deepcopy(layer.attention.self.query)
        # aoglayer.attn.key = copy.deepcopy(layer.attention.self.key)
        # aoglayer.attn.value = copy.deepcopy(layer.attention.self.value)
        # aoglayer.linear = copy.deepcopy(layer.attention.output.dense)

    model.roberta.encoder = aogencoder

    logger.info(f'saving model to {save_model_to}')
    model.save_pretrained(save_model_to)
    tokenizer.save_pretrained(save_model_to)
    return model, tokenizer
    

if __name__ =='__main__':
    # max_pos = 4096
    # model = RobertaAOGForMaskedLM.from_pretrained('/root/autodl-tmp/users/hyb/longformer/tmp/roberta-base-4096')
    # tokenizer = RobertaTokenizerFast.from_pretrained('/root/autodl-tmp/users/hyb/longformer/tmp/roberta-base-4096',model_max_length=max_pos)
    model_path = '/root/autodl-tmp/users/hyb/longformer/tmp/aogformer-4096'
    model, tokenizer = create_aog_model(save_model_to=model_path,attention_window=512,max_pos=4096)
    model = RobertaAOGForMaskedLM.from_pretrained(model_path)
    tokenizer = RobertaTokenizerFast.from_pretrained(model_path)
    config = model.config
    

    SAMPLE_TEXT = ' '.join(['Hello world! '] * 1000)  # long input document
 
    input_ids = torch.tensor(tokenizer.encode(SAMPLE_TEXT)).unsqueeze(0)  # batch of size 1

    # TVM code doesn't work on CPU. Uncomment this if `config.attention_mode = 'tvm'`
    # model = model.cuda(); input_ids = input_ids.cuda()

    # Attention mask values -- 0: no attention, 1: local attention, 2: global attention

    # padding seqlen to the nearest multiple of 512. Needed for the 'sliding_chunks' attention
    input_ids = pad_to_window_size(
            input_ids, config.attention_window[0], tokenizer.pad_token_id)
    attention_mask = torch.ones([1,input_ids.size(1) // config.num_blocks,input_ids.size(1) // config.num_blocks], dtype=torch.long, device=input_ids.device) # initialize to local attention
    

    print(input_ids.size(),attention_mask.size())

    output = model(input_ids,attention_mask=attention_mask)[0]

    loss = output.sum()
    loss.backward()