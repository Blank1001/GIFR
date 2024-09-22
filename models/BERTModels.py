import math

import torch
from torch import nn
from torchcrf import CRF
from transformers import BertModel, ViTModel, BertConfig


class SelfAttentionLayer(torch.nn.Module):
    def __init__(self, config):
        super(SelfAttentionLayer, self).__init__()
        self.config = config
        self.num_attention_heads = self.config.num_attention_heads
        self.hidden_size = self.config.hidden_size
        self.attention_head_size = int(self.hidden_size / self.num_attention_heads)

        self.query = torch.nn.Linear(self.hidden_size, self.hidden_size)
        self.key = torch.nn.Linear(self.hidden_size, self.hidden_size)
        self.value = torch.nn.Linear(self.hidden_size, self.hidden_size)

        self.dropout = torch.nn.Dropout(0.1)
        self.norm1 = torch.nn.LayerNorm(768)
        self.norm2 = torch.nn.LayerNorm(768)
        self.feedforward = torch.nn.Sequential(
            torch.nn.Linear(768, 2 * 768),
            torch.nn.ReLU(),
            torch.nn.Linear(2 * 768, 768)
        )

    # (bsz, seq_len, hidden_size) -> (bsz, num_heads, seq_len, head_size)
    def transpose_for_scores(self, x: torch.Tensor):
        # (bsz, seq_len, hidden_size) -> (bsz, seq_len, num_heads, head_size)
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.reshape(*new_x_shape)
        # (bsz, seq_len, num_heads, head_size) -> (bsz, num_heads, seq_len, head_size)
        return x.permute(0, 2, 1, 3)

    def forward(self, x1, weight_mask: torch.Tensor):
        mixed_query_layer = self.query(x1)
        mixed_key_layer = self.key(x1)
        mixed_value_layer = self.value(x1)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # (bsz, num_heads, seq_len, head_size) * (bsz, num_heads, head_size, seq_len)
        # -> (bsz, num_heads, seq_len, seq_len)
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # (bsz, seq_len) -> (bsz, 1, 1, seq_len) -> (bsz, num_heads, seq_len, seq_len)

        attention_probs = torch.nn.Softmax(dim=-1)(attention_scores)

        attention_probs = attention_probs * weight_mask

        # (bsz, num_heads, seq_len, seq_len) * (bsz, num_heads, seq_len, head_size)
        # (bsz, num_heads, seq_len, head_size)
        context_layer = torch.matmul(attention_probs, value_layer)
        # (bsz, num_heads, seq_len, head_size) -> (bsz, seq_len, num_heads, head_size)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.hidden_size,)
        # (bsz, seq_len, num_heads, head_size) -> (bsz, seq_len, hidden_size)
        context_layer = context_layer.reshape(*new_context_layer_shape)

        x = self.dropout(self.norm1(context_layer + mixed_query_layer))
        forward = self.feedforward(x)
        out = self.dropout(self.norm2(x + forward))

        return out


class GIFR(torch.nn.Module):
    def __init__(self, label_list, args):
        super().__init__()
        self.num_labels = len(label_list)
        self.bert = BertModel.from_pretrained(args.bert_model)
        self.bert.config.hidden_dropout_prob = 0.1
        self.vit = ViTModel.from_pretrained(args.vit_model)
        self.config = self.bert.config
        self.n_layer = self.config.num_hidden_layers
        self.n_head = self.config.num_attention_heads
        self.n_embd = self.config.hidden_size // self.config.num_attention_heads

        self.img_prompt_encoder = torch.nn.Sequential(
            nn.Linear(in_features=768, out_features=1024),
            nn.Tanh(),
            nn.Linear(in_features=1024, out_features=self.n_layer * 2 * self.config.hidden_size)
        )

        self.temporalEmbeddingIn = torch.nn.Embedding(5, 768)
        self.selfattentionlayer = SelfAttentionLayer(self.config)
        self.fc = torch.nn.Linear(768, self.num_labels)
        self.crf = CRF(self.num_labels, batch_first=True)

    def get_prompt(self, img_features=None):
        bsz, img_len, _ = img_features.size()
        # (bsz, 4, 2048)-> (bsz, 4, n_layer*2*hidden_size)
        past_key_values_resnet = self.img_prompt_encoder(img_features)
        past_key_values_resnet = past_key_values_resnet.view(
            bsz,
            img_len,
            self.n_layer * 2,
            self.n_head,
            self.n_embd
        )

        # (bsz, seq_len + seq_len2, 12*2, n_head(12), 64)
        past_key_values = past_key_values_resnet
        # (12*2, bsz, n_head, len, 64)
        # 12*[2,bsz,n_head,len,64]
        past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)

        return past_key_values

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None,
                img_features=None, labels=None, num_images=None, image_attention_mask=None):
        bsz, img_len, channels, height, width = img_features.shape
        # (bsz, 4, 3, 224, 224) -> (bsz*4, 3, 224, 224)
        pixel_values = img_features.reshape(bsz * img_len, channels, height, width)
        # (bsz*4, 3, 224, 224) -> (bsz*4, 768)
        img_features = self.vit(pixel_values=pixel_values)[1]
        # (bsz*4, 768) -> (bsz, 4, 768)
        img_features = img_features.reshape(bsz, img_len, -1)

        temp_weight_indexs = torch.zeros((bsz, 4), dtype=torch.int).to(img_features.device)
        for i in range(bsz):
            num_image_in = torch.tensor(int(num_images[i][0])).to(img_features.device)
            temp_weight_index_in = num_image_in + torch.zeros(4, dtype=torch.int).to(img_features.device)
            temp_weight_indexs[i] = temp_weight_index_in
        temp_embeddings = self.temporalEmbeddingIn(temp_weight_indexs)

        image_attention_mask.to(img_features.device)

        img_features = img_features + temp_embeddings
        # (bsz, 4, 768)
        tempral_features = self.selfattentionlayer(img_features, image_attention_mask)

        # img_features (bsz, 4, 768) -> past_key_values 12*[2, bsz, n_head, len, 64]
        past_key_values = self.get_prompt(tempral_features)
        prompt_guids_length = past_key_values[0][0].size(2)
        bsz = attention_mask.size(0)
        prompt_guids_mask = torch.ones((bsz, prompt_guids_length)).to(attention_mask.device)
        prompt_attention_mask = torch.cat((prompt_guids_mask, attention_mask), dim=1)

        output_bert_result = self.bert(input_ids=input_ids,
                                       attention_mask=prompt_attention_mask,
                                       token_type_ids=token_type_ids,
                                       past_key_values=past_key_values)
        sequence_output = output_bert_result[0]
        emissions = self.fc(sequence_output)
        logits = self.crf.decode(emissions, attention_mask.byte())

        if labels is not None:
            loss = - self.crf(emissions, labels, mask=attention_mask.byte(), reduction='mean')
            return (logits, loss)
