import gzip
import html
import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Optional, Tuple, Union

try:
    import ftfy  # type: ignore
except Exception:  # pragma: no cover
    ftfy = None

try:
    import regex as re  # type: ignore
except Exception:  # pragma: no cover
    import re  # type: ignore
import torch
import torch.nn as nn
import torch.nn.functional as F


@lru_cache()
def _default_bpe_path() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    cand1 = os.path.join(here, "bpe_simple_vocab_16e6.txt.gz")
    if os.path.isfile(cand1):
        return cand1
    cand2 = os.path.abspath(os.path.join(here, os.pardir, "bpe_simple_vocab_16e6.txt.gz"))
    if os.path.isfile(cand2):
        return cand2
    return cand1


@lru_cache()
def _bytes_to_unicode() -> Dict[int, str]:
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


def _get_pairs(word: Tuple[str, ...]) -> set:
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


def _basic_clean(text: str) -> str:
    if ftfy is not None:
        text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()


def _whitespace_clean(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text


class SimpleTokenizer:
    def __init__(self, bpe_path: Optional[str] = None):
        self.byte_encoder = _bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}

        merges_path = bpe_path or _default_bpe_path()
        if not os.path.isfile(merges_path):
            raise FileNotFoundError(
                f"Missing BPE vocab file: {merges_path}. "
                "Please place bpe_simple_vocab_16e6.txt.gz under src/ or provide --openai_bpe_path."
            )

        merges = gzip.open(merges_path).read().decode("utf-8").split("\n")
        merges = merges[1 : 49152 - 256 - 2 + 1]
        merges = [tuple(merge.split()) for merge in merges]

        vocab = list(self.byte_encoder.values())
        vocab = vocab + [v + "</w>" for v in vocab]
        for merge in merges:
            vocab.append("".join(merge))
        vocab.extend(["<|startoftext|>", "<|endoftext|>"])

        self.encoder = dict(zip(vocab, range(len(vocab))))
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.bpe_ranks = dict(zip(merges, range(len(merges))))
        self.cache: Dict[str, str] = {"<|startoftext|>": "<|startoftext|>", "<|endoftext|>": "<|endoftext|>"}
        self.pat = re.compile(
            r"<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+",
            re.IGNORECASE,
        )

    def bpe(self, token: str) -> str:
        if token in self.cache:
            return self.cache[token]

        word = tuple(token[:-1]) + (token[-1] + "</w>",)
        pairs = _get_pairs(word)

        if not pairs:
            return token + "</w>"

        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float("inf")))
            if bigram not in self.bpe_ranks:
                break

            first, second = bigram
            new_word: List[str] = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except ValueError:
                    new_word.extend(word[i:])
                    break

                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1

            word = tuple(new_word)
            if len(word) == 1:
                break
            pairs = _get_pairs(word)

        word_str = " ".join(word)
        self.cache[token] = word_str
        return word_str

    def encode(self, text: str) -> List[int]:
        bpe_tokens: List[int] = []
        text = _whitespace_clean(_basic_clean(text)).lower()
        for token in re.findall(self.pat, text):
            token = "".join(self.byte_encoder[b] for b in token.encode("utf-8"))
            bpe_tokens.extend(self.encoder[bpe_token] for bpe_token in self.bpe(token).split(" "))
        return bpe_tokens


_tokenizer = None


def tokenize(
    texts: Union[str, List[str]],
    context_length: int = 77,
    truncate: bool = False,
    bpe_path: Optional[str] = None,
) -> torch.Tensor:
    global _tokenizer
    resolved_bpe_path = bpe_path or _default_bpe_path()
    if _tokenizer is None or getattr(_tokenizer, "_bpe_path", None) != resolved_bpe_path:
        tok = SimpleTokenizer(bpe_path=resolved_bpe_path)
        tok._bpe_path = resolved_bpe_path  # type: ignore[attr-defined]
        _tokenizer = tok

    if isinstance(texts, str):
        texts = [texts]

    sot_token = _tokenizer.encoder["<|startoftext|>"]
    eot_token = _tokenizer.encoder["<|endoftext|>"]

    all_tokens = [[sot_token] + _tokenizer.encode(text) + [eot_token] for text in texts]
    result = torch.zeros(len(all_tokens), context_length, dtype=torch.int)

    for i, tokens in enumerate(all_tokens):
        if len(tokens) > context_length:
            if truncate:
                tokens = tokens[:context_length]
                tokens[-1] = eot_token
            else:
                raise RuntimeError(f"Input {texts[i]} is too long for context length {context_length}")
        result[i, : len(tokens)] = torch.tensor(tokens)

    return result


class LayerNorm(nn.LayerNorm):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            QuickGELU(),
            nn.Linear(d_model * 4, d_model),
        )
        self.ln_2 = LayerNorm(d_model)

    def attention(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        return self.attn(x, x, x, need_weights=False, attn_mask=attn_mask)[0]

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.attention(self.ln_1(x), attn_mask=attn_mask)
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.ModuleList([ResidualAttentionBlock(width, heads) for _ in range(layers)])

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        for blk in self.resblocks:
            x = blk(x, attn_mask=attn_mask)
        return x


class OpenAIClipTextEncoder(nn.Module):
    def __init__(self, state_dict: dict):
        super().__init__()

        self.text_projection = nn.Parameter(state_dict["text_projection"].clone())
        self.positional_embedding = nn.Parameter(state_dict["positional_embedding"].clone())

        self.vocab_size = int(state_dict["token_embedding.weight"].shape[0])
        self.width = int(state_dict["ln_final.weight"].shape[0])
        self.context_length = int(state_dict["positional_embedding"].shape[0])

        self.token_embedding = nn.Embedding(self.vocab_size, self.width)
        self.ln_final = LayerNorm(self.width)

        self.token_embedding.weight.data.copy_(state_dict["token_embedding.weight"].clone())
        self.ln_final.weight.data.copy_(state_dict["ln_final.weight"].clone())
        self.ln_final.bias.data.copy_(state_dict["ln_final.bias"].clone())

        transformer_layers = len({k.split(".")[2] for k in state_dict.keys() if k.startswith("transformer.resblocks")})
        transformer_heads = self.width // 64
        self.transformer = Transformer(width=self.width, layers=transformer_layers, heads=transformer_heads)

        def _copy_param(dst: torch.Tensor, src: torch.Tensor) -> None:
            dst.data.copy_(src.clone())

        for i in range(transformer_layers):
            prefix = f"transformer.resblocks.{i}."
            blk = self.transformer.resblocks[i]

            _copy_param(blk.ln_1.weight, state_dict[prefix + "ln_1.weight"])
            _copy_param(blk.ln_1.bias, state_dict[prefix + "ln_1.bias"])
            _copy_param(blk.ln_2.weight, state_dict[prefix + "ln_2.weight"])
            _copy_param(blk.ln_2.bias, state_dict[prefix + "ln_2.bias"])

            _copy_param(blk.attn.in_proj_weight, state_dict[prefix + "attn.in_proj_weight"])
            _copy_param(blk.attn.in_proj_bias, state_dict[prefix + "attn.in_proj_bias"])
            _copy_param(blk.attn.out_proj.weight, state_dict[prefix + "attn.out_proj.weight"])
            _copy_param(blk.attn.out_proj.bias, state_dict[prefix + "attn.out_proj.bias"])

            _copy_param(blk.mlp[0].weight, state_dict[prefix + "mlp.c_fc.weight"])
            _copy_param(blk.mlp[0].bias, state_dict[prefix + "mlp.c_fc.bias"])
            _copy_param(blk.mlp[2].weight, state_dict[prefix + "mlp.c_proj.weight"])
            _copy_param(blk.mlp[2].bias, state_dict[prefix + "mlp.c_proj.bias"])

        self.register_buffer("attn_mask", self._build_attention_mask(), persistent=False)

    def _build_attention_mask(self) -> torch.Tensor:
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)
        return mask

    def encode_text_from_tokens(self, text: torch.Tensor) -> torch.Tensor:
        x = self.token_embedding(text)
        x = x + self.positional_embedding
        x = x.permute(1, 0, 2)
        x = self.transformer(x, attn_mask=self.attn_mask)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        x = x / (x.norm(dim=-1, keepdim=True) + 1e-6)
        return x


@dataclass
class CoOpAbnormalPromptConfig:
    n_ctx: int = 8
    class_token_position: str = "end"


class CoOpAbnormalPrompt(nn.Module):
    def __init__(
        self,
        text_encoder: OpenAIClipTextEncoder,
        prompt_texts: List[str],
        cfg: CoOpAbnormalPromptConfig,
        bpe_path: Optional[str] = None,
    ):
        super().__init__()
        self.text_encoder = text_encoder
        self.cfg = cfg

        ctx_dim = int(text_encoder.width)
        self.ctx = nn.Parameter(torch.empty(len(prompt_texts), cfg.n_ctx, ctx_dim))
        nn.init.normal_(self.ctx, std=0.02)

        prompt_prefix = " ".join(["X"] * cfg.n_ctx)
        prompted = [f"{prompt_prefix} {p}." for p in prompt_texts]
        tokenized = tokenize(prompted, context_length=text_encoder.context_length, bpe_path=bpe_path)
        self.register_buffer("tokenized", tokenized, persistent=False)

        with torch.no_grad():
            embeddings = text_encoder.token_embedding(tokenized)
        self.register_buffer("token_prefix", embeddings[:, :1, :], persistent=False)
        self.register_buffer("token_suffix", embeddings[:, 1 + cfg.n_ctx :, :], persistent=False)

    def forward(self) -> torch.Tensor:
        prompts = torch.cat([self.token_prefix, self.ctx, self.token_suffix], dim=1)

        x = prompts + self.text_encoder.positional_embedding
        x = x.permute(1, 0, 2)
        x = self.text_encoder.transformer(x, attn_mask=self.text_encoder.attn_mask)
        x = x.permute(1, 0, 2)
        x = self.text_encoder.ln_final(x)
        x = x[torch.arange(x.shape[0]), self.tokenized.argmax(dim=-1)]
        proj = self.text_encoder.text_projection
        if x.dtype != proj.dtype:
            x = x.to(dtype=proj.dtype)
        x = x @ proj
        x = x.float()
        x = x / (x.norm(dim=-1, keepdim=True) + 1e-6)
        return x.mean(dim=0)


def load_openai_clip_state_dict(ckpt_path: str, map_location: str = "cpu") -> dict:
    ckpt = None

    try:
        jit = torch.jit.load(ckpt_path, map_location=map_location)
        ckpt = jit.state_dict()
    except Exception:
        ckpt = torch.load(ckpt_path, map_location=map_location)

    if isinstance(ckpt, dict):
        for key in ("state_dict", "model_state_dict"):
            if key in ckpt and isinstance(ckpt[key], dict):
                ckpt = ckpt[key]
                break

    if not isinstance(ckpt, dict) or len(ckpt) == 0:
        raise RuntimeError(f"Unexpected checkpoint format: {type(ckpt)}")

    first_key = next(iter(ckpt.keys()))
    if isinstance(first_key, str) and first_key.startswith("module."):
        ckpt = {k[len("module."):]: v for k, v in ckpt.items()}

    return ckpt
