import torch
import numpy as np
import lightning as L

from helpers import batch_data
from transformer.encoder import TransformerEncoder
from constants import output_classes, special_tokens
from transformer.transformer_layers import PositionalEncoding
from transformer.decoder import TransformerDecoder, TransformerDecoderLayer


def select_model(cfg: dict, model_dir: str, return_object: bool = False) -> L.LightningModule:
    if cfg['model']['model_type'] == 'encoder':
        model = Transformer_Encoder(
            cfg["model"],
            train_batch_size=cfg["data"]["train_batch_size"],
            val_batch_size=cfg["data"]["test_batch_size"],
            model_dir=model_dir,
        )
        if return_object:
            model = Transformer_Encoder
    else:
        raise ValueError(f"Unknown model type, {cfg['model']['model_type']}")
    return model


class Transformer_Encoder(L.LightningModule):
    def __init__(
        self,
        model_config: dict,
        train_batch_size: int,
        val_batch_size: int,
        model_dir: str,
        make_model: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["text_vocab", "gloss_vocab", "dataset"])
        self.train_cal_metrics = model_config["train_cal_metrics"]

        self.mixer = None
        self.encoder = None
        self.decoder = None
        if not make_model:
            return

        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size

        self.input_type = model_config['input_type']
        self.input_size = 104
        self.hamer_input_size = 288
        
        self.make_encoder(model_config["encoder"])
        self.make_feature_mixer(model_config["mixer"])

        self.adapter_size = 512

        self.input_layer = torch.nn.Sequential(
            torch.nn.Linear(self.adapter_size*2, model_config["encoder"]["hidden_size"]),
            torch.nn.ReLU(),
            torch.nn.Linear(model_config["encoder"]["hidden_size"], model_config["encoder"]["hidden_size"]),
            torch.nn.ReLU(),
            torch.nn.Linear(model_config["encoder"]["hidden_size"], model_config["encoder"]["hidden_size"]),
            torch.nn.ReLU(),
        )

        self.pose_input_layer = torch.nn.Sequential(
            torch.nn.Linear(self.input_size, self.adapter_size),
            torch.nn.ReLU(),
            torch.nn.Linear(self.adapter_size, self.adapter_size),
            torch.nn.ReLU(),
            torch.nn.Linear(self.adapter_size, self.adapter_size),
            torch.nn.ReLU(),
        )
        self.hamer_input_layer = torch.nn.Sequential(
            torch.nn.Linear(self.hamer_input_size, self.adapter_size),
            torch.nn.ReLU(),
            torch.nn.Linear(self.adapter_size, self.adapter_size),
            torch.nn.ReLU(),
            torch.nn.Linear(self.adapter_size, self.adapter_size),
            torch.nn.ReLU(),
        )
        self.pe = PositionalEncoding(self.adapter_size)

        self.sign_layer = torch.nn.Linear(model_config["encoder"]["hidden_size"], len(output_classes))
        self.sentence_layer = torch.nn.Linear(model_config["encoder"]["hidden_size"], len(output_classes))
        gloss_vocab_length = 11265
        self.gloss_layer = torch.nn.Linear(model_config["encoder"]["hidden_size"], gloss_vocab_length)
       

    def encode(self, src, src_length, src_mask, **kwargs):
        encoder_output = self.encoder(src, src_length, src_mask)

        sign_prediction = self.sign_layer(encoder_output)
        sentence_prediction = self.sentence_layer(encoder_output)
        gloss_prediction = self.gloss_layer(encoder_output)
        return sign_prediction, sentence_prediction, gloss_prediction
        
    def model_forward(self, src, src_length, src_mask, **kwargs):
        if self.input_type == 'both':
            x = self.pose_input_layer(src)
            y = kwargs.get("i3d")

            x = self.pe(x)
            y = self.pe(y)

            trg_mask = torch.ones((x.shape[0], x.shape[1], x.shape[1]), dtype=torch.bool).to('cuda')
            src_mask = kwargs['i3d_mask']

            mixed_features, _ = self.mixer(
                x=x,
                memory=y,
                src_mask=src_mask,
                trg_mask=trg_mask,
                return_attention=False,
            )

            mixed_features = self.input_layer(mixed_features)
            mixed_mask = torch.ones_like(mixed_features, dtype=torch.bool)[..., :1].permute(0, 2, 1)
            mixed_length = src_length
            return self.encode(mixed_features, mixed_length, mixed_mask, **kwargs)
        elif self.input_type == 'i3d':
            y = kwargs.get("i3d")
            y = self.pe(y)
            i3d_length = torch.ones_like(src_length) * y.shape[1]
            i3d_mask = torch.ones_like(y, dtype=torch.bool)[..., :1].permute(0, 2, 1)
            return self.encode(self.input_layer(y), i3d_length, i3d_mask, **kwargs)
        elif self.input_type == 'hamer_mix':
            x = self.pose_input_layer(src)
            y = kwargs.get("hamer")
            y = self.hamer_input_layer(y)
            concatenated = torch.cat((x, y), dim=-1)
            src = self.input_layer(concatenated)
            return self.encode(src, src_length, src_mask, **kwargs)
        else:
            src = self.input_layer(self.pose_input_layer(src))
            return self.encode(src, src_length, src_mask, **kwargs)

    def make_encoder(self, enc_cfg):
        self.encoder = TransformerEncoder(
            **enc_cfg, emb_dropout=enc_cfg["embeddings"].get("dropout", 0.0)
        )

    def make_feature_mixer(self, mixer_cfg):
        self.mixer = TransformerDecoderLayer(**mixer_cfg)

    def label_fixes(self, labels):
        labels = labels.clone()  # avoid in-place mutation if needed
        length = len(labels)

        one_counter = 0
        for index in reversed(range(length)):
            if labels[index] == 1:
                one_counter += 1
            else:
                if one_counter >= 2:
                    # Check if the next label isn't already a 2
                    if index + 1 < length and labels[index + 1] != 2:
                        labels[index + 1] = 2
                one_counter = 0

        if one_counter >= 6 and labels[0] != 2:
            labels[0] = 2
        i = 0
        while i < length:
            if labels[i] == 2:
                j = i + 1
                while j < length and labels[j] == 2:
                    labels[j] = 1
                    j += 1
                i = j
            else:
                i += 1
        return labels
    
    def inference_hamer(self, cfg: dict, angels: dict, hamer_features: torch.tensor, concat: bool = False):
        angles = torch.Tensor(angels)
        hamer_features = torch.Tensor(hamer_features)

        angle_sub = cfg['data']['subsample_pose']
        angle_window = cfg['data']['window_size']
        

        # apply subsampling
        # angels = angels[::angle_sub]
        # hamer_features = hamer_features[::angle_sub]

        

        # get remainder
        a_overlap = angles.shape[0] % cfg['data']['window_size']
        
        if angles.shape[0] % angle_window != 0:
            pad_size = angle_window - a_overlap
            angles = torch.cat((angles, angles[-1].repeat(pad_size, 1)), dim=0)
            hamer_features = torch.cat((hamer_features, hamer_features[-1].repeat(pad_size, 1)), dim=0)
            # a = torch.cat((a, angels[-angle_window:].unsqueeze(0)), dim=0)
            # ham = torch.cat((ham, hamer_features[-angle_window:].unsqueeze(0)), dim=0)
        
        # batch data
        a = batch_data(angles, angle_window, angle_window)
        ham = batch_data(hamer_features, angle_window, angle_window)

        batch = {'src': a.to('cuda'),
                 'src_length': torch.tensor([a.shape[1] for _ in range(a.shape[0])]).to('cuda'),
                 'src_mask': torch.ones((a.shape[0], a.shape[1], a.shape[1]), dtype=torch.bool).to('cuda'),
                 'hamer': ham.to('cuda'),
                 'hamer_mask': torch.ones((ham.shape[0], 1, ham.shape[1]), dtype=torch.bool).to('cuda')}

        sign_level, sentence_level, gloss_level = self.model_forward(**batch)

        sign_crop = sign_level[-1, -a_overlap:]
        sign_level = sign_level[:-1].flatten(0, 1)
        sign_level = torch.cat((sign_level, sign_crop), dim=0).detach().cpu()
        _, sign_level = torch.max(sign_level, dim=-1)
        sign_level = self.label_fixes(sign_level)

        sentence_crop = sentence_level[-1, -a_overlap:]
        sentence_level = sentence_level.flatten(0, 1)
        sentence_level = torch.cat((sentence_level, sentence_crop), dim=0).detach().cpu()
        _, sentence_level = torch.max(sentence_level, dim=-1)

        return sign_level, sentence_level, gloss_level