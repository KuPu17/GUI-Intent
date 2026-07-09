"""
architectures.py

Contains the core deep learning neural network definitions and custom loss engines 
for the UI Behavior Engine.

Author: Kunchit Pujari
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class StableFocalLoss(nn.Module):
    """
    A numerically stable implementation of Focal Loss for highly imbalanced 
    multi-class classification problems.
    
    Prevents floating-point explosion by clamping probabilities before 
    applying the exponential focusing parameter.
    """
    def __init__(self, gamma=2.0, reduction='mean'):
        super(StableFocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # Calculate standard cross entropy securely
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        
        # Clamp inputs to prevent log(0) or exp(massive) numerical shock
        pt = torch.exp(-ce_loss)
        pt = torch.clamp(pt, min=1e-6, max=1.0 - 1e-6)
        
        # Apply the Focal Loss focusing parameter: (1 - pt)^gamma * CE
        focal_loss = ((1.0 - pt) ** self.gamma) * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        return focal_loss


class IntentForecasterGRU_V2(nn.Module):
    """
    Multimodal Gated Recurrent Unit (GRU) designed to process spatial and temporal 
    telemetry to forecast overarching user intent.
    
    Fuses discrete topological zones (GMM tokens) with continuous log-scaled 
    time-deltas prior to recurrent sequential processing.
    """
    def __init__(self, num_zones=12, embedding_dim=32, hidden_dim=64, num_classes=6):
        super(IntentForecasterGRU_V2, self).__init__()
        
        # Spatial Tokenizer: Translates discrete zones (0-11) into 32D dense vectors.
        # Index 99 is reserved for sequence padding.
        self.embedding = nn.Embedding(
            num_embeddings=100, 
            embedding_dim=embedding_dim, 
            padding_idx=99
        )
        
        # Multimodal GRU: Input is spatial embedding (32) + continuous time scalar (1)
        self.gru = nn.GRU(
            input_size=embedding_dim + 1, 
            hidden_size=hidden_dim, 
            batch_first=True
        )
        
        # Classification Head: Projects final hidden state to Macro-Intent logits
        self.fc = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, zone_sequence, time_sequence):
        """
        Forward pass for the multimodal network.
        
        Args:
            zone_sequence (Tensor): [Batch, Sequence Length] - Padded integer tokens
            time_sequence (Tensor): [Batch, Sequence Length] - Padded log-scaled time-deltas
            
        Returns:
            intent_logits (Tensor): [Batch, Num Classes] - Unnormalized classification scores
        """
        # 1. Convert discrete topological zones into continuous mathematical vectors
        embedded_zones = self.embedding(zone_sequence)  # Shape: [Batch, Seq, 32]
        
        # 2. Reshape time to allow for tensor concatenation
        time_sequence = time_sequence.unsqueeze(-1)     # Shape: [Batch, Seq, 1]
        
        # 3. Multimodal Fusion (Space + Time)
        fused_input = torch.cat((embedded_zones, time_sequence), dim=-1)  # Shape: [Batch, Seq, 33]
        
        # 4. Sequential Processing
        gru_out, hidden = self.gru(fused_input)
        
        # 5. Extract the final hidden state summary across the entire task
        final_state = hidden.squeeze(0)  # Shape: [Batch, 64]
        
        # 6. Predict macro-intent
        intent_logits = self.fc(final_state)
        
        return intent_logits