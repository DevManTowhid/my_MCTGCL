import torch
import torch.nn as nn

class IEAM(nn.Module):
    """
    Information-Enhanced Attention Module (IEAM)
    
    As described in Section III-B and Figure 4:
    1. "Segment the input into G sub-features along the channel dimension."
    2. Left Branch (1x1): "Implement two 1-D global AP operations to encode channel information 
       along two spatial directions" (Coordinate Attention style).
    3. Right Branch (3x3): "A single 3x3 kernel is utilized... along with another 2-D global AP operation."
    4. Fusion: "Merge attention information from both branches."
    """
    def __init__(self, in_channels, G=4):
        super(IEAM, self).__init__()
        self.G = G
        self.sub_channels = in_channels // G
        
        # --- 1x1 Branch Components ---
        # "Capture local cross-channel interaction with 1x1 convolution" after concatenating horizontal/vertical pools
        self.conv1x1 = nn.Conv2d(self.sub_channels, self.sub_channels, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.avg_pool_2d = nn.AdaptiveAvgPool2d(1) # For global spatial info

        # --- 3x3 Branch Components ---
        # "Single 3x3 kernel... to capture features at different scales"
        self.conv3x3 = nn.Conv2d(self.sub_channels, self.sub_channels, kernel_size=3, padding=1, bias=False)
        self.avg_pool_3x3_branch = nn.AdaptiveAvgPool2d(1)

        # --- Fusion Components ---
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        b, c, h, w = x.shape
        
        # 1. Group the channels: (B, G, C/G, H, W) -> reshaped to (B*G, C/G, H, W) for parallel processing
        x_reshaped = x.view(b * self.G, self.sub_channels, h, w)
        
        # --- Left Branch (1x1) ---
        # Eq (1) & (2): 1-D Global Average Pooling
        x_h = x_reshaped.mean(dim=3, keepdim=True) # Pool width -> (B*G, C/G, H, 1)
        x_w = x_reshaped.mean(dim=2, keepdim=True) # Pool height -> (B*G, C/G, 1, W)
        
        # Concatenate and interact (Coordinate Attention Logic)
        x_w_permuted = x_w.permute(0, 1, 3, 2)     # (B*G, C/G, W, 1)
        x_cat = torch.cat([x_h, x_w_permuted], dim=2) 
        
        attn = self.conv1x1(x_cat)
        
        # Split back
        x_h_new, x_w_new = torch.split(attn, [h, w], dim=2)
        x_w_new = x_w_new.permute(0, 1, 3, 2)
        
        # "Apply two non-linear sigmoid functions... combine channel-wise attention maps"
        attn_map_1x1 = self.sigmoid(x_h_new) * self.sigmoid(x_w_new)
        
        # "Perform a 2-D global AP in the 1x1 branch" (Eq 3)
        feat_1x1 = x_reshaped * attn_map_1x1
        vec_1x1 = self.avg_pool_2d(feat_1x1)

        # --- Right Branch (3x3) ---
        # 3x3 Conv and Global AP
        feat_3x3 = self.conv3x3(x_reshaped)
        vec_3x3 = self.avg_pool_3x3_branch(feat_3x3)

        # --- Fusion ---
        # "Merge attention information from both branches"
        # Diagram suggests element-wise addition of the pooled vectors, then Softmax
        vec_fused = vec_1x1 + vec_3x3
        attn_vector = self.softmax(vec_fused)

        # "Matrix multiplication... yielding the spatial attention map after a sigmoid calculation"
        # Note: In standard implementation terms, this is channel-wise scaling of the 3x3 features
        out_refined = feat_3x3 * attn_vector
        final_attn = self.sigmoid(out_refined)

        # "Multiply this attention map with X and introduce a shortcut"
        out = x_reshaped * final_attn
        out = out + x_reshaped
        
        # Reshape back to original (B, C, H, W)
        return out.view(b, c, h, w)