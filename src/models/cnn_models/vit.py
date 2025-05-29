import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    """Split image into patches and then embed them.

    Args:
        img_size (int): Size of the image (square).
        patch_size (int): Size of the patch (square).
        in_channels (int): Number of input channels.
        embed_dim (int): Dimensionality of embedding.
    """

    def __init__(self, img_size, patch_size, in_channels=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) * (img_size // patch_size)

        self.proj = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input image of shape (B, C, H, W).
        Returns:
            torch.Tensor: Embedded patches of shape (B, num_patches, embed_dim).
        """
        x = self.proj(x)  # [B, embed_dim, H', W']
        x = x.flatten(2)  # [B, embed_dim, num_patches]
        x = x.transpose(1, 2)  # [B, num_patches, embed_dim]
        return x


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention mechanism.

    Args:
        embed_dim (int): Dimensionality of input sequences.
        num_heads (int): Number of attention heads.
        dropout (float): Dropout probability.
    """

    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim**-0.5

        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input sequence of shape (B, N, embed_dim).
        Returns:
            torch.Tensor: Output sequence of shape (B, N, embed_dim).
        """
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.dropout(x)
        return x


class MLP(nn.Module):
    """Multi-Layer Perceptron module.

    Args:
        in_features (int): Number of input features.
        hidden_features (int): Number of features in the hidden layer.
        out_features (int): Number of output features.
        dropout (float): Dropout probability.
    """

    def __init__(
        self, in_features, hidden_features=None, out_features=None, dropout=0.0
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (B, N, in_features).
        Returns:
            torch.Tensor: Output tensor of shape (B, N, out_features).
        """
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """Transformer Encoder Block.

    Args:
        embed_dim (int): Dimensionality of input sequences.
        num_heads (int): Number of attention heads.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        dropout (float): Dropout probability.
        attention_dropout (float): Dropout probability for attention weights.
    """

    def __init__(
        self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.0, attention_dropout=0.0
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attention = MultiHeadAttention(
            embed_dim, num_heads, dropout=attention_dropout
        )
        self.dropout = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = MLP(
            in_features=embed_dim, hidden_features=mlp_hidden_dim, dropout=dropout
        )

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input sequence of shape (B, N, embed_dim).
        Returns:
            torch.Tensor: Output sequence of shape (B, N, embed_dim).
        """
        h = x
        x = self.norm1(x)
        x = self.attention(x)
        x = self.dropout(x)
        x = h + x

        h = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = h + x
        return x


class VisionTransformer(nn.Module):
    """Vision Transformer model.
    paper: An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale
    url: https://arxiv.org/abs/2010.11929

    Args:
        img_size (int): Size of input image (square).
        patch_size (int): Size of each patch (square).
        in_channels (int): Number of input channels.
        num_classes (int): Number of output classes for classification.
        embed_dim (int): Dimensionality of embedding.
        depth (int): Number of Transformer blocks.
        num_heads (int): Number of attention heads.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        dropout (float): Dropout probability.
        attention_dropout (float): Dropout probability for attention weights.
        representation_size (int, optional): If specified, add an additional layer
            to project embeddings to this size.
    """

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_channels=3,
        num_classes=1000,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        dropout=0.0,
        attention_dropout=0.0,
        representation_size=None,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim

        self.patch_embed = PatchEmbedding(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
        )
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=dropout)

        self.blocks = nn.Sequential(
            *[
                TransformerBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    attention_dropout=attention_dropout,
                )
                for _ in range(depth)
            ]
        )

        self.norm = nn.LayerNorm(embed_dim)

        if representation_size:
            self.has_logits = True
            self.pre_logits = nn.Sequential(
                nn.Linear(embed_dim, representation_size), nn.Tanh()
            )
        else:
            self.has_logits = False
            self.pre_logits = nn.Identity()

        self.head = (
            nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        )

        # Weight initialization
        self._init_weights()

    def _init_weights(self):
        """Initialize weights in the model."""
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        """
        Args:
            x (torch.Tensor): Input image of shape (B, C, H, W).
        Returns:
            torch.Tensor: Output features of shape (B, embed_dim).
        """
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        x = self.blocks(x)
        x = self.norm(x)
        return self.pre_logits(x[:, 0])

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input image of shape (B, C, H, W).
        Returns:
            torch.Tensor: Output logits of shape (B, num_classes).
        """
        x = self.forward_features(x)
        x = self.head(x)
        return x


# Common Vision Transformer models
def vit_tiny_patch16_224(num_classes=1000, **kwargs):
    """ViT-Tiny (Vit-Ti/16) model."""
    model = VisionTransformer(
        patch_size=16,
        embed_dim=192,
        depth=12,
        num_heads=3,
        num_classes=num_classes,
        **kwargs
    )
    return model


def vit_small_patch16_224(num_classes=1000, **kwargs):
    """ViT-Small (ViT-S/16) model."""
    model = VisionTransformer(
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=6,
        num_classes=num_classes,
        **kwargs
    )
    return model


def vit_base_patch16_224(num_classes=1000, **kwargs):
    """ViT-Base (ViT-B/16) model."""
    model = VisionTransformer(
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        num_classes=num_classes,
        **kwargs
    )
    return model


def vit_large_patch16_224(num_classes=1000, **kwargs):
    """ViT-Large (ViT-L/16) model."""
    model = VisionTransformer(
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        num_classes=num_classes,
        **kwargs
    )
    return model


def get_vit(scale="small", num_classes=1000):
    """Get Vision Transformer model based on scale."""
    if scale == "tiny":
        model = vit_tiny_patch16_224(num_classes=num_classes)
        complete_model_name = "vit_tiny_patch16_224"
    elif scale == "small":
        model = vit_small_patch16_224(num_classes=num_classes)
        complete_model_name = "vit_small_patch16_224"
    elif scale == "base":
        model = vit_base_patch16_224(num_classes=num_classes)
        complete_model_name = "vit_base_patch16_224"
    elif scale == "large":
        model = vit_large_patch16_224(num_classes=num_classes)
        complete_model_name = "vit_large_patch16_224"
    else:
        raise ValueError(f"Invalid scale: {scale}.")
    return model, complete_model_name