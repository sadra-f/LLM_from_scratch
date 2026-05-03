from transformers import AutoTokenizer



class MTokenizer():
    def __init__(self):
        # self._tokenizer = AutoTokenizer.from_pretrained("gpt2", cache_dir=".cache/")
        #forcing to use the cache! change to your needs.
        self._tokenizer = AutoTokenizer.from_pretrained(".cache/models--gpt2/snapshots/607a30d783dfa663caf39e06633721c8d4cfcd7e")
    
    def encode(self, data):
        return self._tokenizer(data, add_special_tokens=False)
    
    def decode(self, data):
        return self._tokenizer.decode(data, skip_special_tokens=True)