"""HFPredictor: load base (+LoRA adapter) and predict answer distributions by
reading the renormalized next-token probs over the item option letters. Matches src/lp.py
exactly, so eval is consistent with training. Run on the GPU box.
"""
from __future__ import annotations

from src.schema import Item
from src import lp


class HFPredictor:
    def __init__(self, base="Qwen/Qwen2.5-1.5B-Instruct", adapter=None,
                 temperature=1.0, load_4bit=False, max_len=None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.temperature = float(temperature)   # >1 softens, <1 sharpens the dist
        self.max_len = max_len
        self.tok = AutoTokenizer.from_pretrained(adapter or base)
        self.tok.padding_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        if load_4bit:                           # 7B only fits the 8 GB card in 4-bit
            from transformers import BitsAndBytesConfig
            self.model = AutoModelForCausalLM.from_pretrained(
                base, device_map="auto", quantization_config=BitsAndBytesConfig(
                    load_in_4bit=True, bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype))
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                base, torch_dtype=dtype, device_map="auto")
        if adapter:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, adapter)
        self.model.eval()
        self._lids = {}

    def _ids(self, k):
        if k not in self._lids:
            self._lids[k] = lp.letter_token_ids(self.tok, k)
        return self._lids[k]

    def raw_letter_logits(self, item: Item):
        """The model's raw logits over the k option letters (pre-softmax).

        Exposed so a single temperature can be grid-searched on a validation set
        without re-running the model per candidate T (see src.tune)."""
        ids = lp.prompt_ids(self.tok, item)
        if self.max_len and len(ids) > self.max_len:
            raise ValueError(
                f"item {item.id} is {len(ids)} tokens, above max_len={self.max_len}")
        inp = self.torch.tensor([ids]).to(self.model.device)
        with self.torch.no_grad():
            logits = self.model(inp).logits[0, -1, :]
        lids = self.torch.tensor(self._ids(item.k())).to(logits.device)
        return logits[lids].float().cpu().tolist()

    def __call__(self, item: Item):
        lg = self.torch.tensor(self.raw_letter_logits(item))
        probs = self.torch.softmax(lg / self.temperature, dim=-1)
        return probs.tolist()
