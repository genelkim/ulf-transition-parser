from transformers import RobertaTokenizer
import torch
from typing import List

class AMRBertTokenizer(BertTokenizer):

    def __init__(self, *args, **kwargs):
        super(AMRBertTokenizer, self).__init__(*args, **kwargs)

    def _align_bpe_to_words(self, bpe_tokens: torch.Tensor, other_tokens: List[str]):
        """Align GPT-2 BPE to other tokenization formats (e.g., spaCy).
        This function is based on code from fairseq.
        https://github.com/pytorch/fairseq/blob/master/fairseq/models/roberta/alignment_utils.py
        Params
        ------
        bpe_tokens : torch.LongTensor
            GPT-2 BPE tokens of shape `(T_bpe)`
        other_tokens : List[str]
            other tokens of shape `(T_words)`
        Returns
        -------
        List[str]
            The mapping from `other_tokens` to corresponding `bpe_tokens`.
        Raises
        ------
        ValueError
            If `bpe_tokens` and `other_tokens` cannot be aligned.
        """
        assert bpe_tokens.dim() == 1
        assert bpe_tokens[0] == 0.  # added after revision in alignment utils from fairseq (Feb11, 2020)

        def clean(text):
            return text.strip()

        # orig_bpe_tokens = bpe_tokens
        # orig_other_tokens = other_tokens

        # remove whitespaces to simplify alignment
        bpe_tokens = [self.model.task.source_dictionary.string([x]) for x in bpe_tokens]
        bpe_tokens = [clean(self.model.bpe.decode(x) if x not in {'<s>', ''} else x) for x in bpe_tokens]
        other_tokens = [clean(str(o)) for o in other_tokens]

        # strip leading <s>
        bpe_tokens = bpe_tokens[1:]
        assert ''.join(bpe_tokens) == ''.join(other_tokens)

        # create alignment from every word to a list of BPE tokens
        alignment = []
        bpe_toks = filter(lambda item: item[1] != '', enumerate(bpe_tokens, start=1))
        j, bpe_tok = next(bpe_toks)
        for other_tok in other_tokens:
            bpe_indices = []
            while True:
                if other_tok.startswith(bpe_tok):
                    bpe_indices.append(j)
                    other_tok = other_tok[len(bpe_tok):]
                    try:
                        j, bpe_tok = next(bpe_toks)
                    except StopIteration:
                        j, bpe_tok = None, None
                elif bpe_tok.startswith(other_tok):
                    # other_tok spans multiple BPE tokens
                    bpe_indices.append(j)
                    bpe_tok = bpe_tok[len(other_tok):]
                    other_tok = ''
                else:
                    raise ValueError('Cannot align "{}" and "{}"'.format(other_tok, bpe_tok))
                if other_tok == '':
                    break
            assert len(bpe_indices) > 0
            alignment.append(bpe_indices)
        assert len(alignment) == len(other_tokens)
        return alignment
