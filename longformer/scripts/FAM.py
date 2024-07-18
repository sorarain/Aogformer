import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.roberta.modeling_roberta import RobertaLayer,RobertaEncoder,RobertaAttention
from transformers import RobertaForMaskedLM#RobertaSelfAttention is modified
from transformers.utils import logging
from transformers import RobertaTokenizerFast


logger = logging.get_logger(__name__)


class RobertaFAMAttention(RobertaAttention):
    def __init__(self, config, position_embedding_type=None):
        super().__init__(config)
        self.FAM = nn.Embedding(1,config.hidden_size)
        self.segment_size = config.segment_size
        self.num_pre_seg = config.num_pre_seg
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.num_heads = config.num_attention_heads
        self.head_dim = int(config.hidden_size / config.num_attention_heads)
        self.embed_dim = config.hidden_size


        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.position_embedding_type = position_embedding_type or getattr(
            config, "position_embedding_type", "absolute"
        )
        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(2 * config.max_position_embeddings - 1, self.attention_head_size)

        self.is_decoder = config.is_decoder
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=False,
        past_key_value=None,
    ):
        hidden_states = hidden_states.transpose(0, 1)

        # project hidden states
        query_vectors = self.query(hidden_states)
        key_vectors = self.key(hidden_states)
        value_vectors = self.value(hidden_states)

        seq_len, batch_size, embed_dim = hidden_states.size()
        assert (
            embed_dim == self.embed_dim
        ), f"hidden_states should have embed_dim = {self.embed_dim}, but has {embed_dim}"

        # normalize query
        query_vectors /= math.sqrt(self.head_dim)

        query_vectors = query_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
        key_vectors = key_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
        value_vectors = value_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)

        if self.is_decoder:
            past_key_value = (key_vectors, value_vectors)

        num_seg = int(math.ceil(seq_len / self.segment_size))
        fam = self.FAM(torch.zeros([1, batch_size,1],device=hidden_states.device,dtype=torch.int64))
        context = torch.zeros([batch_size, 0, embed_dim],device=hidden_states.device)
        for i in range(num_seg):
            x,y = i * self.segment_size, (i + 1) * self.segment_size
            y = min(y, seq_len)
            q_tau, k_tau, v_tau = query_vectors[:, x : y, :, :], key_vectors[:, x : y, :, :,], value_vectors[:, x : y, :, :]
            q_tau_f, k_tau_f, v_tau_f = self.query(fam).view(1, batch_size, self.num_heads, self.head_dim).transpose(0, 1), \
                                        self.key(fam).view(1, batch_size, self.num_heads, self.head_dim).transpose(0, 1), \
                                            self.value(fam).view(1, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
            pre_x = (i - self.num_pre_seg) * self.segment_size
            pre_y = x
            pre_x = min(max(pre_x, 0), pre_y)
            k_hat = torch.concat([key_vectors[:, pre_x : pre_y, :, :], k_tau_f, k_tau], dim=1).transpose(1, 2)
            v_hat = torch.concat([value_vectors[:, pre_x : pre_y, :, :], v_tau_f, v_tau], dim=1).transpose(1, 2)
            attn_scores = torch.einsum("bcxd,bcyd->bcxy", (q_tau.transpose(1, 2), k_hat))
            attn_probs = nn.functional.softmax(
                attn_scores, dim=-1, dtype=torch.float32
            )  # use fp32 for numerical stability
            # softmax sometimes inserts NaN if all positions are masked, replace them with 0
            attn_probs = attn_probs.type_as(attn_scores)  
            # free memory
            del attn_scores

            # apply dropout
            attn_probs = self.dropout(attn_probs)

            seg_context = torch.einsum("bcwd,bcdh->bcwh", (attn_probs, v_hat)) 

            seg_context = seg_context.permute(0, 2, 1, 3).reshape([batch_size, (y - x), self.num_heads * self.head_dim])

            seg_context = self.output(seg_context, hidden_states[x : y, : , :].transpose(0, 1))

            context = torch.concat([context, seg_context], dim=1)


            k_tau_F = torch.concat([k_tau_f, k_tau], dim=1).transpose(1, 2)
            v_tau_F = torch.concat([v_tau_f, v_tau], dim=1).transpose(1, 2)
            attn_scores_f = torch.einsum("bcxd,bcyd->bcxy", (q_tau_f.transpose(1, 2), k_tau_F))

            attn_probs_f = nn.functional.softmax(
                attn_scores_f, dim=-1, dtype=torch.float32
            )  # use fp32 for numerical stability
            # softmax sometimes inserts NaN if all positions are masked, replace them with 0
            attn_probs_f = attn_probs_f.type_as(attn_scores_f)  
            # free memory
            del attn_scores_f
            
            attn_probs_f = self.dropout(attn_probs_f) 

            seg_f = torch.einsum("bcwd,bcdh->bcwh", (attn_probs_f, v_tau_F)) 

            seg_f = seg_f.permute(0, 2, 1, 3).reshape([batch_size, 1, self.num_heads * self.head_dim])

            fam = self.output(seg_f, fam)


        
        # attn_output = self._sliding_chunks_matmul_attn_probs_value(
        #     attn_probs, value_vectors, self.one_sided_attn_window_size
        # )

        # assert attn_output.size() == (batch_size, seq_len, self.num_heads, self.head_dim), "Unexpected size"
        # attn_output = attn_output.transpose(0, 1).reshape(seq_len, batch_size, embed_dim).contiguous()

        # outputs = (attn_output.transpose(0, 1),)

        # if output_attentions:
        #     outputs += (attn_probs,)
        outputs = (context, attn_probs) if output_attentions else (context,)

        if self.is_decoder:
            outputs = outputs + (past_key_value,)

        return outputs
        # return super().forward(hidden_states, attention_mask=attention_mask.squeeze(1,2), output_attentions=output_attentions)

class FAMEncoder(RobertaEncoder):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.layer = nn.ModuleList([RobertaLayer(config) for _ in range(config.num_hidden_layers)])
        for i, layer in enumerate(self.layer):
            # replace the `modeling_bert.BertSelfAttention` object with `LongformerSelfAttention`
            layer.attention = RobertaFAMAttention(config)
        self.gradient_checkpointing = False

    

class RobertaFAMForMaskedLM(RobertaForMaskedLM):
    def __init__(self, config):
        super().__init__(config)
        self.roberta.encoder = FAMEncoder(config)


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


def create_fam_model(save_model_to, attention_window, max_pos):
    model = RobertaForMaskedLM.from_pretrained('roberta-base')
    tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base', model_max_length=max_pos)
    config = model.config
    config.segment_size = 512
    config.num_pre_seg = 1

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
    config.attention_window = [attention_window] * config.num_hidden_layers
    for i, layer in enumerate(model.roberta.encoder.layer):
        longformer_self_attn = RobertaFAMAttention(config)
        longformer_self_attn.query = layer.attention.self.query
        longformer_self_attn.key = layer.attention.self.key
        longformer_self_attn.value = layer.attention.self.value

        layer.attention = longformer_self_attn

    logger.info(f'saving model to {save_model_to}')
    model.save_pretrained(save_model_to)
    tokenizer.save_pretrained(save_model_to)
    return model, tokenizer
    

if __name__ =='__main__':
    # max_pos = 4096
    # model = RobertaAOGForMaskedLM.from_pretrained('/root/autodl-tmp/users/hyb/longformer/tmp/roberta-base-4096')
    # tokenizer = RobertaTokenizerFast.from_pretrained('/root/autodl-tmp/users/hyb/longformer/tmp/roberta-base-4096',model_max_length=max_pos)
    model_path = '/root/autodl-tmp/users/hyb/longformer/tmp/famformer-4096'
    # model, tokenizer = create_fam_model(save_model_to=model_path,attention_window=512,max_pos=4096)
    model = RobertaFAMForMaskedLM.from_pretrained(model_path)
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