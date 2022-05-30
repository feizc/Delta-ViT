import torch 
import torch.nn as nn 
from .vit import ViTLayer, ViTPreTrainedModel, ViTEmbeddings, ViTPooler


class Perturbator(nn.Module): 
    def __init__(self, config, bias=True, act=nn.Tanh): 
        super(Perturbator, self).__init__() 
        self.layers = nn.ModuleList()
        for i in range(config.num_hidden_layers):
            self.layers.append(nn.Linear(config.hidden_size, 1, bias=bias)) 
            self.layers.append(act()) 
    
    def forward(self, hidden_states, index): 
        outputs = self.layers[2 * index](hidden_states) 
        outputs = self.layers[2 * index + 1](outputs) 
        return outputs 



class DeltaViTEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([ViTLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states,
        head_mask=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    layer_head_mask,
                )
            else:
                layer_outputs = layer_module(hidden_states, layer_head_mask, output_attentions)

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None) 
    

    # one step for index layer
    def step(self, hidden_states, index):
        layer_outputs = self.layer[index](hidden_states) 
        return layer_outputs 


class DeltaViTModel(ViTPreTrainedModel):
    def __init__(self, config, add_pooling_layer=True, use_mask_token=False):
        super().__init__(config)
        self.config = config 
        self.num_layers = config.num_hidden_layers 

        self.embeddings = ViTEmbeddings(config, use_mask_token=use_mask_token)
        self.encoder = DeltaViTEncoder(config) 
        self.perturbator = Perturbator(config)

        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.pooler = ViTPooler(config) if add_pooling_layer else None

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.patch_embeddings
    
    def forward(
        self,
        pixel_values=None,
        bool_masked_pos=None,
        output_attentions=None,
        output_hidden_states=None,
        interpolate_pos_encoding=None,
        return_dict=None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        embedding_output = self.embeddings(
            pixel_values, bool_masked_pos=bool_masked_pos, interpolate_pos_encoding=interpolate_pos_encoding
        )

        for i in range(self.num_layers): 
            encoder_outputs = self.encoder.step(embedding_output, i)[0]
            perturbations = self.perturbator(encoder_outputs, i) # add learnable perturbations
            embedding_output = encoder_outputs + perturbations 

        sequence_output = embedding_output
        sequence_output = self.layernorm(sequence_output)
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        
        return (sequence_output, pooled_output) + (encoder_outputs[1:],)



class DeltaViTForImageClassification(ViTPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.num_labels = config.num_labels
        self.vit = DeltaViTModel(config, add_pooling_layer=False)

        # Classifier head
        self.classifier = nn.Linear(config.hidden_size, config.num_labels) if config.num_labels > 0 else nn.Identity()

        # Initialize weights and apply final processing
        self.post_init()


    def forward(
        self,
        pixel_values=None,
        output_attentions=None,
        output_hidden_states=None,
        interpolate_pos_encoding=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.vit(
            pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            interpolate_pos_encoding=interpolate_pos_encoding,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        logits = self.classifier(sequence_output[:, 0, :])

        
        output = (logits,) + outputs[2:]
        return output


    def train_only_perturbator(self): 
        for name, parameter in self.vit.named_parameters(): 
            if 'perturbator' not in name:
                parameter.requires_grad = False 
        

    def train_without_perturbator(self): 
        for name, parameter in self.vit.named_parameters(): 
            if 'perturbator' in name:
                parameter.requires_grad = False 


