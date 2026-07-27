import torch
import torch.nn as nn
import numpy as np
import gymnasium as gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class AttentionFeatureExtractor(BaseFeaturesExtractor):
    """
    Cross-attention feature extractor for multi-agent intersection control.
    """
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256, 
                 ego_features: int = 4, neighbor_features: int = 5, max_neighbors: int = 8, 
                 embed_dim: int = 64, num_heads: int = 4):
        
        # Initialize the base class with the expected output dimension
        super().__init__(observation_space, features_dim)

        self.ego_features = ego_features
        self.neighbor_features = neighbor_features
        self.max_neighbors = max_neighbors
        self.embed_dim = embed_dim

        # --- Encoders ---
        self.ego_encoder = nn.Sequential(
            nn.Linear(self.ego_features, self.embed_dim),
            nn.LayerNorm(self.embed_dim), 
            nn.ReLU(),
        )
        self.neighbor_encoder = nn.Sequential(
            nn.Linear(self.neighbor_features, self.embed_dim),
            nn.LayerNorm(self.embed_dim), 
            nn.ReLU(),
        )

        # --- Multi-Head Cross-Attention ---
        self.attention = nn.MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=num_heads,
            batch_first=True,  
        )

        # --- Context Processing ---
        context_dim = self.embed_dim * 2  
        self.context_norm = nn.LayerNorm(context_dim)

        # Project the concatenated context down to the features_dim expected by SB3
        self.projection = nn.Sequential(
            nn.Linear(context_dim, features_dim),
            nn.ReLU()
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        obs = observations.float()
        
        # --- 1. Split observation ---
        ego_end = self.ego_features
        neighbor_end = ego_end + (self.max_neighbors * self.neighbor_features)
        
        ego_raw = obs[:, :ego_end]                         
        neighbor_raw = obs[:, ego_end:neighbor_end]         
        mask_raw = obs[:, neighbor_end:]                    

        neighbor_raw = neighbor_raw.reshape(-1, self.max_neighbors, self.neighbor_features)

        # --- 2. Encode ---
        ego_embed = self.ego_encoder(ego_raw)                
        neighbor_embeds = self.neighbor_encoder(neighbor_raw) 

        # --- 3. Build attention mask ---
        key_padding_mask = (mask_raw < 0.5)  
        all_masked = key_padding_mask.all(dim=1)  
        
        # --- 4. Cross-Attention ---
        query = ego_embed.unsqueeze(1)   
        
        safe_mask = key_padding_mask.clone()
        safe_mask[all_masked] = False  

        attn_output, _ = self.attention(
            query=query,
            key=neighbor_embeds,
            value=neighbor_embeds,
            key_padding_mask=safe_mask,
        )  

        attn_output = attn_output.squeeze(1)  

        attn_output = torch.where(
            all_masked.unsqueeze(-1),
            torch.zeros_like(attn_output),
            attn_output
        )

        # --- 5. Concatenate and decode ---
        context = torch.cat([ego_embed, attn_output], dim=-1)  
        context = self.context_norm(context)
        
        # Return the final latent representation to SB3
        return self.projection(context)
