import editdistance


class CerWer():
    def __init__(self):
        self.vocab = "B abcdefghijklmnopqrstuvwxyz'"
        self.char2idx = {char: idx for idx, char in enumerate(self.vocab)}
        self.idx2char = {idx: char for idx, char in enumerate(self.vocab)}

    def __call__(self, predicts, targets, inputs_length, targets_length):
        cer = 0.0
        wer = 0.0
        for predict, target, input_length, target_length in zip(predicts, targets, inputs_length, targets_length):
            predict_string = self.process_string(predict, input_length, remove_repetitions=True)
            target_string = self.process_string(target, target_length)

            predict_words = predict_string.rstrip().split(' ')
            target_words = target_string.rstrip().split(' ')

            dist = editdistance.eval(target_string, predict_string)
            dist_word = editdistance.eval(target_words, predict_words)

            cer += dist / len(target_string)
            wer += dist_word / len(target_words)
        return cer, wer, predict_string, target_string

    def process_string(self, sequence, length, remove_repetitions=False):
        string = ''
        for i in range(length):
            char = self.idx2char[sequence[i]]
            if char != "B":
                if remove_repetitions and i != 0 and char == self.idx2char[sequence[i - 1]]:
                    pass
                else:
                    string = string + char
        return string