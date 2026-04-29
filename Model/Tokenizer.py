from transformers import AutoTokenizer



class MTokenizer():
    def __init__(self):
        self._tokenizer = AutoTokenizer.from_pretrained("gpt2", cache_dir=".cache/")
    
    def encode(self, data):
        return self._tokenizer(data, add_special_tokens=False)
    
    def decode(self, data):
        return self._tokenizer.decode(data, skip_special_tokens=True)