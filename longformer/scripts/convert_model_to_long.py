import logging
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "4"
import math
import copy
import torch
from dataclasses import dataclass, field
from transformers import RobertaForMaskedLM, RobertaTokenizerFast, TextDataset, DataCollatorForLanguageModeling, Trainer
from transformers import TrainingArguments, HfArgumentParser, PretrainedConfig
from transformers.models.longformer.modeling_longformer import LongformerSelfAttention



logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class RobertaLongSelfAttention(LongformerSelfAttention):
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
        
        return super().forward(hidden_states, attention_mask=attention_mask.squeeze(1,2), output_attentions=output_attentions)


class RobertaLongForMaskedLM(RobertaForMaskedLM):
    def __init__(self, config):
        super().__init__(config)
        for i, layer in enumerate(self.roberta.encoder.layer):
            # replace the `modeling_bert.BertSelfAttention` object with `LongformerSelfAttention`
            layer.attention.self = RobertaLongSelfAttention(config, layer_id=i)
            

def create_long_model(save_model_to, attention_window, max_pos):
    model = RobertaForMaskedLM.from_pretrained('roberta-base')
    tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base', model_max_length=max_pos)
    config = model.config

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
        longformer_self_attn = LongformerSelfAttention(config, layer_id=i)
        longformer_self_attn.query = layer.attention.self.query
        longformer_self_attn.key = layer.attention.self.key
        longformer_self_attn.value = layer.attention.self.value

        longformer_self_attn.query_global = copy.deepcopy(layer.attention.self.query)
        longformer_self_attn.key_global = copy.deepcopy(layer.attention.self.key)
        longformer_self_attn.value_global = copy.deepcopy(layer.attention.self.value)

        layer.attention.self = longformer_self_attn

    logger.info(f'saving model to {save_model_to}')
    model.save_pretrained(save_model_to)
    tokenizer.save_pretrained(save_model_to)
    return model, tokenizer



def copy_proj_layers(model):
    for i, layer in enumerate(model.roberta.encoder.layer):
        layer.attention.self.query_global = copy.deepcopy(layer.attention.self.query)
        layer.attention.self.key_global = copy.deepcopy(layer.attention.self.key)
        layer.attention.self.value_global = copy.deepcopy(layer.attention.self.value)
    return model


def pretrain_and_evaluate(args, model, tokenizer, eval_only, model_path):
    val_dataset = TextDataset(tokenizer=tokenizer,
                              file_path=args.val_datapath,
                            #   block_size=tokenizer.max_len
                              block_size=tokenizer.model_max_length##################
                              )
    if eval_only:
        train_dataset = val_dataset
    else:
        logger.info(f'Loading and tokenizing training data is usually slow: {args.train_datapath}')
        train_dataset = TextDataset(tokenizer=tokenizer,
                                    file_path=args.train_datapath,
                                    # block_size=tokenizer.max_len
                                    block_size=tokenizer.model_max_length
                                    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)
    trainer = Trainer(model=model, args=args, data_collator=data_collator,
                      train_dataset=train_dataset, eval_dataset=val_dataset,)#, prediction_loss_only=True

    eval_loss = trainer.evaluate()
    eval_loss = eval_loss['eval_loss']
    logger.info(f'Initial eval bpc: {eval_loss/math.log(2)}')
    
    if not eval_only:
        trainer.train(model_path=model_path)
        trainer.save_model()

        eval_loss = trainer.evaluate()
        eval_loss = eval_loss['eval_loss']
        logger.info(f'Eval bpc after pretraining: {eval_loss/math.log(2)}')

@dataclass
class ModelArgs:
    attention_window: int = field(default=4096, metadata={"help": "Size of attention window"})
    max_pos: int = field(default=4096, metadata={"help": "Maximum position"})

parser = HfArgumentParser((TrainingArguments, ModelArgs,))


training_args, model_args = parser.parse_args_into_dataclasses(look_for_args_file=False, args=[
    '--output_dir', '/root/autodl-tmp/users/hyb/longformer/tmp/longformer_from_scratch/v1_text8',
    '--warmup_steps', '2000',
    '--learning_rate', '0.0003',
    '--weight_decay', '0.01',
    '--adam_epsilon', '1e-6',
    '--max_steps', '6000',
    '--logging_steps', '500',
    '--save_steps', '500',
    '--max_grad_norm', '5.0',
    '--per_device_eval_batch_size', '4',
    '--per_device_train_batch_size', '2',  # 32GB gpu with fp32
    '--gradient_accumulation_steps', '32',
    '--fp16',
    '--fp16_opt_level', 'O2',
    '--half_precision_backend','auto',
    # '--fsdp', 'auto_wrap',
    # '--use_cpu',
    '--prediction_loss_only',
    '--eval_strategy', 'epoch',
    '--do_train',
    '--do_eval',
    
])
training_args.val_datapath = '/root/autodl-tmp/users/hyb/longformer/data/wikitext-103-raw/wiki.valid.raw'
training_args.train_datapath = '/root/autodl-tmp/users/hyb/longformer/data/wikitext-103-raw/wiki.train.raw'
# training_args.val_datapath = '/root/autodl-tmp/users/hyb/longformer/data/text8/text8'
# training_args.train_datapath = '/root/autodl-tmp/users/hyb/longformer/data/text8/text8'
# Choose GPU




# roberta_base = RobertaForMaskedLM.from_pretrained('/root/autodl-tmp/users/hyb/longformer/roberta-base', local_files_only=True)
# roberta_base_tokenizer = RobertaTokenizerFast.from_pretrained('/root/autodl-tmp/users/hyb/longformer/roberta-base', local_files_only=True)
# logger.info('Evaluating roberta-base (seqlen: 512) for refernece ...')
# pretrain_and_evaluate(training_args, roberta_base, roberta_base_tokenizer, eval_only=True, model_path=None)



# model_path = f'{training_args.output_dir}/roberta-base-{model_args.max_pos}'
# if not os.path.exists(model_path):
#     os.makedirs(model_path)

# logger.info(f'Converting roberta-base into roberta-base-{model_args.max_pos}')
# model, tokenizer = create_long_model(
#     save_model_to=model_path, attention_window=model_args.attention_window, max_pos=model_args.max_pos)

model_path = '/root/autodl-tmp/users/hyb/longformer/tmp/roberta-base-4096'
logger.info(f'Loading the model from {model_path}')
tokenizer = RobertaTokenizerFast.from_pretrained(model_path)
model = RobertaLongForMaskedLM.from_pretrained(model_path)
# model = RobertaLongForMaskedLM(config=model.config)
# model = RobertaLongForMaskedLM.from_pretrained('/root/autodl-tmp/users/hyb/longformer/tmp/longformer_from_scratch/checkpoint-6000')
# model = RobertaLongForMaskedLM(config=model.config)

logger.info(f'Pretraining roberta-base-{model_args.max_pos} ... ')

pretrain_and_evaluate(training_args, model, tokenizer, eval_only=True, model_path=None)

# training_args.max_steps = 3   ## <<<<<<<<<<<<<<<<<<<<<<<< REMOVE THIS <<<<<<<<<<<<<<<<<<<<<<<<

# pretrain_and_evaluate(training_args, model, tokenizer, eval_only=False, model_path=None)#os.path.join(training_args.output_dir,f"roberta-base-{model_args.max_pos}")
# notebook_launcher(training_args, args=(training_args, model, tokenizer, False, training_args.output_dir), num_processes=2)


# logger.info(f'Copying local projection layers into global projection layers ... ')
# model = copy_proj_layers(model)
# logger.info(f'Saving model to {model_path}')
# model.save_pretrained(model_path)


# logger.info(f'Loading the model from {model_path}')
# tokenizer = RobertaTokenizerFast.from_pretrained(model_path)
# model = RobertaLongForMaskedLM.from_pretrained(model_path)