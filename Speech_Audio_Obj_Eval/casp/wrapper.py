import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
from .scheduler import CosineAnnealingWarmupRestarts
from itertools import chain
import torch.distributed as dist

def all_gather(tensor):
    # 获取当前进程组的世界大小（进程数）
    world_size = dist.get_world_size()
    # 创建一个列表来存储所有进程的张量
    tensor_list = [torch.zeros_like(tensor) for _ in range(world_size)]
    # 使用 all_gather 收集张量
    dist.all_gather(tensor_list, tensor)
    # 将所有张量拼接在一起
    gathered_tensor = torch.cat(tensor_list, dim=0)
    return gathered_tensor
def get_audio_encoder(ckpt_path):

    from .beats.BEATs import BEATs, BEATsConfig
    checkpoint = torch.load(ckpt_path)
    cfg = BEATsConfig(checkpoint['cfg'])
    model = BEATs(cfg)
    model.load_state_dict(checkpoint['model'])

    return model


def get_speech_encoder(ckpt_path):

    from .beats.BEATs import BEATs, BEATsConfig
    checkpoint = torch.load(ckpt_path)
    cfg = BEATsConfig(checkpoint['cfg'])
    model = BEATs(cfg)
    # model.load_state_dict(checkpoint['model'])
    return model

class AttentionPooling(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        # 可学习的注意力权重向量
        self.attention = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, 1)
        )

    def forward(self, x):
        # x 形状: (b, t, d)
        attn_weights = self.attention(x).squeeze(-1)  # (b, t)
        attn_weights = F.softmax(attn_weights, dim=-1)  # 归一化权重
        pooled = torch.sum(x * attn_weights.unsqueeze(-1), dim=1)  # (b, d)
        return pooled

class CASP(nn.Module):
    def __init__(self, d_model=512, ckpt_path=None):
        super().__init__()
        # 编码器（假设已定义）
        self.speech_encoder = get_speech_encoder(ckpt_path=ckpt_path)
        self.audio_encoder = get_audio_encoder(ckpt_path=ckpt_path)
        
        # 自注意力池化模块
        self.speech_pooling = AttentionPooling(d_model)
        self.audio_pooling = AttentionPooling(d_model)
        

        # 添加两层线性层，增加模型复杂度
        self.speech_proj = nn.Sequential(
            nn.Linear(d_model, d_model * 2),  # 第一层线性层
            nn.LayerNorm(d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model)   # 第二层线性层
        )

        self.audio_proj = nn.Sequential(
            nn.Linear(d_model, d_model * 2),  # 第一层线性层
            nn.LayerNorm(d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model)   # 第二层线性层
        )

        # 可学习的温度参数
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.tensor(1.0))


    def encode_speech(self, speech, padding_mask=None):
        if padding_mask == None:
            padding_mask = torch.zeros(speech.shape).bool().to(speech.device)
        speech_embeds = self.speech_encoder.extract_features(speech, padding_mask=padding_mask)[0].contiguous()
        return speech_embeds
    
    def encode_audio(self, audio, padding_mask=None):
        if padding_mask == None:
            padding_mask = torch.zeros(audio.shape).bool().to(audio.device)
        audio_embeds = self.audio_encoder.extract_features(audio, padding_mask=padding_mask)[0].contiguous()
        return audio_embeds
    
    def forward(self, audio, speech):
        # 编码 speech 和 audio
        speech_embeds = self.encode_speech(speech)  # (b, t, d)
        audio_embeds = self.encode_audio(audio)    # (b, t, d)

        # 通过自注意力池化聚合时间维度
        speech_global = self.speech_pooling(speech_embeds)  # (b, d)
        audio_global = self.audio_pooling(audio_embeds)     # (b, d)


        # 通过两层线性层投影，增加模型复杂度
        speech_global = self.speech_proj(speech_global)  # (b, d)
        audio_global = self.audio_proj(audio_global)     # (b, d)

        # 归一化特征
        speech_global = F.normalize(speech_global, p=2, dim=-1)  # (b, d)
        audio_global = F.normalize(audio_global, p=2, dim=-1)    # (b, d)

        # 计算余弦相似度（logits）
        logit_scale = self.logit_scale.exp()  # 标量

        # speech_global = all_gather(speech_global)
        # audio_global = all_gather(audio_global)
        logits_per_speech = logit_scale * speech_global @ audio_global.t()  # (b, b)
        logits_per_audio = logit_scale * audio_global @ speech_global.t()   # (b, b)

        return logits_per_speech, logits_per_audio
    def inference_noscale(self, audio, speech, audio_mask, speech_mask):
        # 编码 speech 和 audio
        speech_embeds = self.encode_speech(speech, padding_mask=speech_mask)  # (b, t, d)
        audio_embeds = self.encode_audio(audio, padding_mask=audio_mask)    # (b, t, d)

        # 通过自注意力池化聚合时间维度
        speech_global = self.speech_pooling(speech_embeds)  # (b, d)
        audio_global = self.audio_pooling(audio_embeds)     # (b, d)


        # 通过两层线性层投影，增加模型复杂度
        speech_global = self.speech_proj(speech_global)  # (b, d)
        audio_global = self.audio_proj(audio_global)     # (b, d)

        # 归一化特征
        speech_global = F.normalize(speech_global, p=2, dim=-1)  # (b, d)
        audio_global = F.normalize(audio_global, p=2, dim=-1)    # (b, d)

        logits_per_speech = speech_global @ audio_global.t()  # (b, b)
        logits_per_audio = audio_global @ speech_global.t()   # (b, b)

        return logits_per_speech, logits_per_audio




class CASPWrapper(pl.LightningModule):
    def __init__(self,
                 d_model = 512,
                 max_train_steps = 1000,
                 learning_rate = 1e-4,
                 log_step = 5,
                 ckpt_path = None,
                 ):

        super().__init__()
        self.model = CASP(d_model=d_model, ckpt_path=ckpt_path)
        self.d_model = d_model
        self.max_train_steps = max_train_steps
        self.learning_rate = learning_rate
        self.log_step = log_step
        self.automatic_optimization = False
    
    # Sourced from https://github.com/PyTorchLightning/pytorch-lightning/issues/5449
    @property
    def num_training_steps(self) -> int:
        """Total training steps inferred from datamodule and devices."""
        dataset = self.train_dataloader()
        if self.trainer.max_steps:
            return self.trainer.max_steps

        dataset_size = len(dataset)

        num_devices = max(1, self.trainer.num_gpus, self.trainer.num_processes)
        if self.trainer.tpu_cores:
            num_devices = max(num_devices, self.trainer.tpu_cores)

        effective_batch_size = dataset.batch_size * self.trainer.accumulate_grad_batches * num_devices
        return (dataset_size // effective_batch_size) * self.trainer.max_epochs

    # Training loss: https://github.com/openai/CLIP/issues/83
    # Mini-batching thanks to https://github.com/crowsonkb / https://twitter.com/RiversHaveWings
    # Multi-GPU support: https://github.com/MicPie/clasp
    def training_step(self, train_batch, batch_idx):
        
        audio, speech = train_batch
        device = speech.device
        
        
        logits_per_speech, logits_per_audio = self.model(audio, speech)

        optimizer = self.optimizers()
        if isinstance(optimizer, list):
            optimizer = optimizer[0]
        optimizer.zero_grad()

        ground_truth = torch.arange(len(logits_per_speech)).long().to(device)
        loss = (F.cross_entropy(logits_per_speech, ground_truth) + F.cross_entropy(logits_per_audio, ground_truth))/2
    
        if torch.isnan(loss):
            print("Loss is NaN!")
            torch.cuda.empty_cache()
            return
        
        self.manual_backward(loss)


        optimizer.step()
        lr_scheduler = self.lr_schedulers()
        lr_scheduler.step()
        self.model.logit_scale.data.clamp_(-np.log(100), np.log(100))
        
        
        # if (batch_idx + 1) % self.log_step == 0:
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

    def validation_step(self, val_batch, idx):
        speech, audio = val_batch
        logits_per_speech, logits_per_audio = self.model(speech, audio)
        ground_truth = torch.arange(len(logits_per_speech))
        loss = (F.cross_entropy(logits_per_speech, ground_truth) + F.cross_entropy(logits_per_audio, ground_truth)).div(2)
        self.log('val_loss', loss)

    def configure_optimizers(self):
        
        optimizer = torch.optim.AdamW(
            chain(filter(lambda p: p.requires_grad, self.model.parameters())) , 
            lr=self.learning_rate,
            betas=(
                0.9,
                0.99 
            ),
            eps= 1e-8,
            weight_decay=0.2
        )

        # Source: https://github.com/openai/CLIP/issues/107
        # Use pip install 'git+https://github.com/katsura-jp/pytorch-cosine-annealing-with-warmup'
        lr_scheduler = CosineAnnealingWarmupRestarts(
            optimizer,
            # first_cycle_steps=self.num_training_steps,
            first_cycle_steps=self.max_train_steps,
            cycle_mult=1.0,
            max_lr=self.learning_rate,
            min_lr=0,
            warmup_steps=2000
        )

        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler}

