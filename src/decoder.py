import editdistance
from src.dataset import prepare_bpe


class CerWer():
    def __init__(self, blank_index=0, space_simbol='▁'):
        bpe = prepare_bpe()
        self.idx2char = bpe.id_to_subword
        self.blank_index = blank_index
        self.space_simbol = space_simbol

    def __call__(self, predicts, targets, inputs_length, targets_length):
        """
        :param predicts: tensor (batch, len_input) with ids of decoded tokens
        :param targets: tensor (batch, len_target) with ids of target texts
        :param inputs_length: tensor (batch, ) original lenth og input
        :param targets_length: tensor (batch, ) original length of target
        :return: sum of cer over batch, sum of wer over batch, last predicted string, last target for logs
        """
        cer = 0.0
        wer = 0.0
        for predict, target, input_length, target_length in zip(predicts, targets, inputs_length, targets_length):
            predict_string = self.process_string(predict, input_length, remove_repetitions=True)
            target_string = self.process_string(target, target_length)

            predict_words = predict_string.rstrip().split(self.space_simbol)
            target_words = target_string.rstrip().split(self.space_simbol)

            dist = editdistance.eval(target_string, predict_string)
            dist_word = editdistance.eval(target_words, predict_words)

            cer += dist / len(target_string)
            wer += dist_word / len(target_words)
        return cer, wer, predict_string, target_string

    def process_string(self, sequence, length, remove_repetitions=False):
        """
        Returns decoded string without blank simbols and just convert target to original text
        :param sequence: tensor with signal
        :param length: original length of signal
        :param remove_repetitions: True for removing repetitions in predicted string
        :return: decoded string
        """
        string = ''
        for i in range(length):
            char = self.idx2char(sequence[i])
            if char != self.idx2char(self.blank_index):
                if remove_repetitions and i != 0 and char == self.idx2char(sequence[i - 1]):
                    pass
                else:
                    string = string + char
        return string

    def inference(self, predicts, input_len):
        """
        :param predicts:
        :param input_len:
        :return:
        """
        predict_string = self.process_string(predicts, input_len, remove_repetitions=True)
        predict_words = predict_string.rstrip().split('_')
        return predict_words

