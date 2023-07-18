from transformers.models.t5.modeling_t5 import T5Stack, T5PreTrainedModel
from transformers.modeling_outputs import (BaseModelOutput,
                                           Seq2SeqLMOutput)
from transformers.utils.model_parallel_utils import get_device_map, assert_device_map
from pytorch_lightning.core.lightning import LightningModule
import warnings
import copy
import torch
import torch.nn as nn
import numpy as np

from utils import apply_noise
from loss import BarlowTwinsLoss

__HEAD_MASK_WARNING_MSG = """
The input argument `head_mask` was split into two arguments `head_mask` and `decoder_head_mask`. Currently,
`decoder_head_mask` is set to copy `head_mask`, but this feature is deprecated and will be removed in future versions.
If you do not want to use any `decoder_head_mask` now, please set `decoder_head_mask = torch.ones(num_layers,
num_heads)`.
"""


class T5ForConditionalGenerationWithExtractor(T5PreTrainedModel):
    _keys_to_ignore_on_load_missing = [
        r"encoder\.embed_tokens\.weight",
        r"decoder\.embed_tokens\.weight",
        r"lm_head\.weight",
    ]
    _keys_to_ignore_on_load_unexpected = [
        r"decoder\.block\.0\.layer\.1\.EncDecAttention\.relative_attention_bias\.weight",
    ]

    def __init__(self, config):
        super().__init__(config)
        self.model_dim = config.d_model
        self.lambda_factor = 1
        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = T5Stack(encoder_config, self.shared)

        extractor_config = copy.deepcopy(config)
        extractor_config.is_decoder = False
        extractor_config.use_cache = False
        extractor_config.is_encoder_decoder = False
        self.extractor = T5Stack(extractor_config, self.shared)

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = T5Stack(decoder_config, self.shared)

        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        # Initialize weights and apply final processing
        self.post_init()

        # Model parallel
        self.model_parallel = False
        self.device_map = None

    def parallelize(self, device_map=None):
        self.device_map = (
            get_device_map(len(self.encoder.block),
                           range(torch.cuda.device_count()))
            if device_map is None
            else device_map
        )
        assert_device_map(self.device_map, len(self.encoder.block))
        self.encoder.parallelize(self.device_map)
        self.decoder.parallelize(self.device_map)
        self.extractor.parallelize(self.device_map)
        self.lm_head = self.lm_head.to(self.decoder.first_device)
        self.model_parallel = True

    def deparallelize(self):
        self.encoder.deparallelize()
        self.extractor.deparallelize()
        self.decoder.deparallelize()
        self.encoder = self.encoder.to("cpu")
        self.extractor = self.extractor.to("cpu")
        self.decoder = self.decoder.to("cpu")
        self.lm_head = self.lm_head.to("cpu")
        self.model_parallel = False
        self.device_map = None
        torch.cuda.empty_cache()

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings
        self.encoder.set_input_embeddings(new_embeddings)
        self.extractor.set_input_embeddings(new_embeddings)
        self.decoder.set_input_embeddings(new_embeddings)

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def get_output_embeddings(self):
        return self.lm_head

    def get_encoder(self):
        return self.encoder

    def get_extractor(self):
        return self.extractor

    def get_decoder(self):
        return self.decoder

    def get_extractor_output(self,
                             input_ids=None,
                             # use cache is simply to a trick to use the generator mixin
                             use_cache_context_ids=None,
                             use_cache_target_examplars_ids=None,
                             use_cache_origin_examplars_ids=None,
                             attention_mask=None,
                             decoder_input_ids=None,
                             decoder_attention_mask=None,
                             head_mask=None,
                             decoder_head_mask=None,
                             cross_attn_head_mask=None,
                             encoder_outputs=None,
                             extractor_outputs=None,
                             past_key_values=None,
                             inputs_embeds=None,
                             context_embeds=None,
                             decoder_inputs_embeds=None,
                             labels=None,
                             use_cache=None,
                             output_attentions=None,
                             output_hidden_states=None,
                             return_dict=None,):
        extractor_hidden = None
        if use_cache_context_ids is None:
            target_styles = ()
            for target_ids in use_cache_target_examplars_ids:
                extractor_hidden = self.extractor(
                    input_ids=target_ids,
                    attention_mask=attention_mask,
                    inputs_embeds=context_embeds,
                    head_mask=head_mask,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                )[0]
                target_styles += (extractor_hidden,)

            original_styles = ()
            for origin_ids in use_cache_origin_examplars_ids:
                extractor_hidden = self.extractor(
                    input_ids=origin_ids,
                    attention_mask=attention_mask,
                    inputs_embeds=context_embeds,
                    head_mask=head_mask,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                )[0]
                original_styles += (extractor_hidden,)

            input_style = self.extractor(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=context_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )[0]
            extractor_hidden = self.lambda_factor * (torch.mean(torch.vstack(
                target_styles), 0) - (torch.mean(torch.vstack(original_styles), 0))) + input_style

        else:
            if extractor_outputs is None:
                extractor_outputs = self.extractor(
                    input_ids=use_cache_context_ids,
                    attention_mask=attention_mask,
                    inputs_embeds=context_embeds,
                    head_mask=head_mask,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                )
            elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
                extractor_outputs = BaseModelOutput(
                    last_hidden_state=extractor_outputs[0],
                    hidden_states=extractor_outputs[1] if len(
                        extractor_outputs) > 1 else None,
                    attentions=extractor_outputs[2] if len(extractor_outputs) > 2 else None,)
            extractor_hidden = extractor_outputs[0]
        return extractor_hidden

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        use_cache_extractor_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        context_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[-100, 0, ...,
            config.vocab_size - 1]`. All labels set to `-100` are ignored (masked), the loss is only computed for
            labels in `[0, ..., config.vocab_size]`
        Returns:
        Examples:
        ```python
        >>> from transformers import T5Tokenizer, T5ForConditionalGeneration
        >>> tokenizer = T5Tokenizer.from_pretrained("t5-small")
        >>> model = T5ForConditionalGeneration.from_pretrained("t5-small")
        >>> # training
        >>> input_ids = tokenizer("The <extra_id_0> walks in <extra_id_1> park", return_tensors="pt").input_ids
        >>> labels = tokenizer("<extra_id_0> cute dog <extra_id_1> the <extra_id_2>", return_tensors="pt").input_ids
        >>> outputs = model(input_ids=input_ids, labels=labels)
        >>> loss = outputs.loss
        >>> logits = outputs.logits
        >>> # inference
        >>> input_ids = tokenizer(
        ...     "summarize: studies have shown that owning a dog is good for you", return_tensors="pt"
        >>> ).input_ids  # Batch size 1
        >>> outputs = model.generate(input_ids)
        >>> print(tokenizer.decode(outputs[0], skip_special_tokens=True))
        >>> # studies have shown that owning a dog is good for you.
        ```"""
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                warnings.warn(__HEAD_MASK_WARNING_MSG, FutureWarning)
                decoder_head_mask = head_mask

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            # Convert encoder inputs in embeddings if needed
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(
                    encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(
                    encoder_outputs) > 2 else None,
            )

        hidden_states = encoder_outputs[0] + use_cache_extractor_outputs

        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)

        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
            hidden_states = hidden_states.to(self.decoder.first_device)
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids.to(
                    self.decoder.first_device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.decoder.first_device)
            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask.to(
                    self.decoder.first_device)

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = decoder_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.encoder.first_device)
            self.lm_head = self.lm_head.to(self.encoder.first_device)
            sequence_output = sequence_output.to(self.lm_head.weight.device)

        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            sequence_output = sequence_output * (self.model_dim**-0.5)

        lm_logits = self.lm_head(sequence_output)

        loss = None
        if labels is not None:
            # loss_extractor_fct = BarlowTwinsLoss(batch_size=64)
            loss_output_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
            # loss_extractor = loss_extractor_fct()
            loss = loss_output_fct(
                lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
            # TODO(thom): Add z_loss https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L666

        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        use_cache_extractor_outputs=None,
        past=None,
        attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs
    ):

        # cut decoder_input_ids if past is used
        if past is not None:
            input_ids = input_ids[:, -1:]

        return {
            # "input_ids": input_ids,
            # "use_cache_context_ids": use_cache_context_ids,
            # "use_cache_target_examplars_ids": use_cache_target_examplars_ids,
            # "use_cache_origin_examplars_ids": use_cache_origin_examplars_ids,
            "decoder_input_ids": input_ids,
            "past_key_values": past,
            "encoder_outputs": encoder_outputs,
            "use_cache_extractor_outputs": use_cache_extractor_outputs,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,
        }

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return self._shift_right(labels)

    def _reorder_cache(self, past, beam_idx):
        # if decoder past is not included in output
        # speedy decoding is disabled and no need to reorder
        if past is None:
            warnings.warning(
                "You might want to consider setting `use_cache=True` to speed up decoding")
            return past

        reordered_decoder_past = ()
        for layer_past_states in past:
            # get the correct batch idx from layer past batch dim
            # batch dim of `past` is at 2nd position
            reordered_layer_past_states = ()
            for layer_past_state in layer_past_states:
                # need to set correct `past` for each of the four key / value states
                reordered_layer_past_states = reordered_layer_past_states + (
                    layer_past_state.index_select(
                        0, beam_idx.to(layer_past_state.device)),
                )

            assert reordered_layer_past_states[0].shape == layer_past_states[0].shape
            assert len(reordered_layer_past_states) == len(layer_past_states)

            reordered_decoder_past = reordered_decoder_past + \
                (reordered_layer_past_states,)
        return reordered_decoder_past


class TextSettrModel(LightningModule):
    def __init__(self, lambda_val, sent_length, tokenizer):
        super().__init__()
        self.net = T5ForConditionalGenerationWithExtractor.from_pretrained(
            "./pretrained_model/t5-base-with-extractor")
        self.net.extractor = copy.deepcopy(self.net.encoder)
        self.lambda_val = lambda_val
        self.sent_length = sent_length
        self.tokenizer = tokenizer

    def training_step(self, batch, batch_idx):
        context_ids, input_ids = batch[0], batch[1]
        # print('input ids size', input_ids.size())
        noisy_input_ids = apply_noise(
            input_ids, self.tokenizer, self.sent_length)
        if np.random.choice([False, True]):
            # Noisy back translation
            noisy_input_ids = self.net.generate(input_ids=noisy_input_ids, use_cache_extractor_outputs=0,
                                                do_sample=True, max_length=self.sent_length, min_length=self.sent_length)
        extractor_output = self.net.get_extractor_output(
            use_cache_context_ids=context_ids)

        extractor_output_input = self.net.get_extractor_output(
            use_cache_context_ids=input_ids)
        barlow_twins_loss_func = BarlowTwinsLoss(batch_size=64)
        barlow_twins_loss = barlow_twins_loss_func(
            extractor_output_input, extractor_output)

        output_loss = self.net(input_ids=noisy_input_ids, labels=input_ids,
                               use_cache_extractor_outputs=extractor_output).loss

        if output_loss is not None:
            return output_loss + self.lambda_val * barlow_twins_loss
        else:
            return None
        # return self.net(input_ids=noisy_input_ids, labels = input_ids, use_cache_extractor_outputs=extractor_output).loss + barlow_twins_loss

    def validation_step(self, batch, batch_idx):
        context_ids, input_ids = batch[0], batch[1]
        noisy_input_ids = apply_noise(
            input_ids, self.tokenizer, self.sent_length)
        noisy_input_ids = self.net.generate(input_ids=noisy_input_ids, use_cache_extractor_outputs=0,
                                            do_sample=True, max_length=self.sent_length, min_length=self.sent_length)
        extractor_output = self.net.get_extractor_output(
            use_cache_context_ids=context_ids)

        extractor_output_input = self.net.get_extractor_output(
            use_cache_context_ids=input_ids)
        barlow_twins_loss_func = BarlowTwinsLoss(batch_size=64)
        barlow_twins_loss = barlow_twins_loss_func(
            extractor_output_input, extractor_output)

        output_loss = self.net(input_ids=noisy_input_ids, labels=input_ids,
                               use_cache_extractor_outputs=extractor_output).loss

        if output_loss is not None:
            self.log("val_loss", output_loss +
                     self.lambda_val * barlow_twins_loss)
        # else:
        #   self.log("val_loss", output_loss)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.net.parameters(), 1e-3)
