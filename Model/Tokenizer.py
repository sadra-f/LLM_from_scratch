from transformers import AutoTokenizer



class MTokenizer():
    def __init__(self):
        self._tokenizer = AutoTokenizer.from_pretrained("gpt2", cache_dir=".cache/")
    
    def tokenize(self, data):
        return self._tokenizer(data)
    
    def decode(self, data):
        return self._tokenizer.decode(data, skip_special_tokens=True)