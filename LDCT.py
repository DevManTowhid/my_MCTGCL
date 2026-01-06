import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossTransformerBlock(nn.Module):
    """
    Cross Transformer (CT) Block as shown in Figure 5 and Eq (4)-(11).
    It uses Multi-Head Cross-Attention (MHCA) where the Keys/Values are augmented 
    with global information (AP/MP) from the second feature split (F2).
    """
    def __init__(self, dim, num_heads=4, mlp_ratio=4., dropout=0.1):
        super(CrossTransformerBlock, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Linear projections for Q, K', V', I'
        # Q, K', V' come from F1
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        
        # [cite_start]I' comes from F2 (pooled) [cite: 4]
        self.i_proj = nn.Linear(dim, dim)

        self.proj_out = nn.Linear(dim, dim)
        
        # Norms
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
        # MLP
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(dropout)
        )

    def forward(self, f1, f2):
        """
        Args:
            f1: Input feature map 1 (Batch, Channels, Height, Width)
            f2: Input feature map 2 (Batch, Channels, Height, Width)
        """
        b, c, h, w = f1.shape
        
        # Flatten F1 for Transformer processing: (B, H*W, C)
        f1_flat = f1.permute(0, 2, 3, 1).flatten(1, 2)
        
        # --- Pre-Norm ---
        # "F1 -> LayerNorm -> MHCA" structure in Fig 5
        x = self.norm1(f1_flat)
        
        # --- 1. Generate Q, K', V' from F1 ---
        # Eq (4): Q = F1' Wq, etc.
        q = self.q_proj(x).reshape(b, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k_prime = self.k_proj(x).reshape(b, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v_prime = self.v_proj(x).reshape(b, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # --- 2. Generate I' from F2 ---
        # "AP and max pooling (MP) are applied to F2 to retain valuable information"
        # We generate global tokens.
        ap_f2 = F.adaptive_avg_pool2d(f2, (1, 1)).flatten(1) # (B, C)
        mp_f2 = F.adaptive_max_pool2d(f2, (1, 1)).flatten(1) # (B, C)
        
        # Stack them to form the context input F2' (B, 2, C)
        f2_tokens = torch.stack([ap_f2, mp_f2], dim=1)
        
        # Eq (4): I' = F2' Wi
        i_prime = self.i_proj(f2_tokens).reshape(b, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        # --- 3. Concatenate to form K and V ---
        # Eq (9): Ki = Concat(Ki', Ii'), Vi = Concat(Vi', Ii')
        # We concat along the token dimension (dim 2)
        k = torch.cat([k_prime, i_prime], dim=2) # (B, Heads, H*W + 2, HeadDim)
        v = torch.cat([v_prime, i_prime], dim=2)
        
        # --- 4. Attention ---
        # Eq (10): Softmax(Q K^T / sqrt(d)) V
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        
        x_attn = (attn @ v).transpose(1, 2).reshape(b, h*w, c)
        x_attn = self.proj_out(x_attn)
        
        # --- Residual 1 ---
        # (f1_flat + MHCA)
        x = f1_flat + x_attn
        
        # --- MLP Block ---
        # "LayerNorm -> MLP -> Add"
        x = x + self.mlp(self.norm2(x))
        
        # Reshape back to (B, C, H, W)
        x = x.permute(0, 2, 1).reshape(b, c, h, w)
        return x

class LDCT(nn.Module):
    """
    Lightweight Dual-Branch CNN-Transformer (LDCT) as described in Section III-C.
    
    Structure:
    1. 1x1 Conv Input
    2. Split into F1, F2
    3. Branch 1: Cross Transformer (uses F1 and pooled F2)
    4. Branch 2: CNN (3x3 -> 5x5) (uses F2)
    5. Concat -> 1x1 Conv
    """
    def __init__(self, in_channels, out_channels=64):
        super(LDCT, self).__init__()
        
        # "Input F... processed through a 1x1 convolutional layer"
        self.conv_in = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)
        
        # "Split... into two sub-matrices F1 and F2... sized (C/2)"
        self.half_dim = in_channels // 2
        
        # --- Transformer Branch ---
        self.ct_block = CrossTransformerBlock(dim=self.half_dim)
        
        # --- CNN Branch ---
        # "Sequentially apply 3x3 and 5x5 convolutions"
        self.cnn_branch = nn.Sequential(
            nn.Conv2d(self.half_dim, self.half_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.half_dim),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(self.half_dim, self.half_dim, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(self.half_dim),
            nn.ReLU(inplace=True)
        )
        
        # --- Fusion ---
        # "Concatenate their outputs and apply a 1x1 convolutional layer"
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels) # Usually added for stability
        )

    def forward(self, x):
        # 1. Input Projection
        x = self.conv_in(x)
        
        # 2. Split into F1 and F2
        # "Dividing the input along the channel dimension"
        f1, f2 = torch.split(x, self.half_dim, dim=1)
        
        # 3. Parallel Processing
        # Transformer Branch uses F1 (main) and F2 (context)
        out_trans = self.ct_block(f1, f2)
        
        # CNN Branch uses F2
        out_cnn = self.cnn_branch(f2)
        
        # 4. Fusion
        # "Combine... outputs... fuse local and non-local features"
        # Note: If out_trans and out_cnn have different spatial sizes due to padding issues,
        # ensure padding is correct (handled above with padding=1 for 3x3, padding=2 for 5x5).
        fused = torch.cat([out_trans, out_cnn], dim=1)
        
        out = self.fusion_conv(fused)
        
        return out