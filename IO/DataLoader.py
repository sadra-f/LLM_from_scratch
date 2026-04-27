


class MDataLoader:
    def __init__(self, file_path, tokenizer, window_size):
        self._file_path = file_path
        self.tokenizer = tokenizer
        self.window_size = window_size
        self._current_window = []
        self._file_handle = None

    def __iter__(self):
        if not self._file_handle:
            self._file_handle = open(self._file_path, 'r')
        return self.__next__()

    def __next__(self):
        for line in self._file_handle:
            self._current_window.extend(self.tokenizer.tokenize(line)['input_ids'])
            if len(self._current_window) < self.window_size + 1: # +1 since the y_i is x_(i+1) i.e. the next token for the task of next token prediction
                continue
            tmp_x = self._current_window[:self.window_size]
            tmp_y = self._current_window[1:self.window_size+1]
            yield tmp_x, tmp_y
            self._current_window = self._current_window[self.window_size:]
        eos_token = self.tokenizer.tokenize("<|endoftext|>")['input_ids'][0]
        self._current_window.extend([eos_token for _ in range(self.window_size - len(self._current_window) + 1)])
        yield self._current_window[:self.window_size], self._current_window[1:self.window_size+1]



    def __enter__(self):
        self._file_handle = open(self._file_path, 'r')
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._file_handle:
            self._file_handle.close()
            self._file_handle = None