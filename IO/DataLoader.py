import torch


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
        while True:
            line = self.file_handle.readline()
            if not line:
                self.file_handle.close()
                raise StopIteration

            tokens = self.tokenizer.encode(line)["input_ids"]
            self.buffer.extend(tokens)
            while len(self.buffer) >= self.window_size + 1:
                x = self.buffer[:self.window_size]
                y = self.buffer[1:self.window_size + 1]
                self.buffer = self.buffer[self.stride:]

                return (
                    torch.tensor(x, dtype=torch.long),
                    torch.tensor(y, dtype=torch.long)
                )



    def __enter__(self):
        self._file_handle = open(self._file_path, 'r')
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._file_handle:
            self._file_handle.close()
            self._file_handle = None





class StreamingWindowLoader:
    def __init__(self, file_path, tokenizer, window_size, stride=1):
        self.file_path = file_path
        self.tokenizer = tokenizer
        self.window_size = window_size
        self.stride = stride

    def __iter__(self):
        self.file_handle = open(self.file_path, "r", encoding="utf-8")
        self.buffer = []

        return self

    def __next__(self):
        while True:
            line = self.file_handle.readline()

            # End of file
            if not line:
                self.file_handle.close()
                raise StopIteration

            # Correct HuggingFace usage
            tokens = self.tokenizer.encode(line)["input_ids"]
            self.buffer.extend(tokens)

            # Produce multiple windows if possible
            while len(self.buffer) >= self.window_size + 1:
                x = self.buffer[:self.window_size]
                y = self.buffer[1:self.window_size + 1]

                # slide buffer (controlled overlap)
                self.buffer = self.buffer[self.stride:]

                return (
                    torch.tensor(x, dtype=torch.long),
                    torch.tensor(y, dtype=torch.long)
                )