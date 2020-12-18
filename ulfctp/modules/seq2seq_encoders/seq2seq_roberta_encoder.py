import torch
from torch import Tensor
from collections import Counter
from typing import List
import fairseq
import logging

log = logging.getLogger(__name__)


class Singleton(object):
    """Singleton implementation from the Python docs.
    This implementation is presented by Guido van Rossum in a discussion about
    the uses of __new__.
    https://www.python.org/download/releases/2.2/descrintro/#__new__
    """

    def __new__(cls, *args, **kwds):
        it = cls.__dict__.get("__it__")
        if it is not None:
            return it
        cls.__it__ = it = object.__new__(cls)
        it.init(*args, **kwds)
        return it

    def init(self, *args, **kwds):
        pass


class RoBERTa(Singleton):
    """A class to align RoBERTa embeddings to pre-tokenized sentences.
    Usage:
        .. code-block:: python
            from roberta import RoBERTaBase
            embeddings = RoBERTaBase.align(sentence, tokens)
        where sentence is a string, and tokens are the tokens of the sentence.
        `RoBERTaLarge` can be used instead of `RoBERTaBase`.
    Raises
    ------
    NotImplementedError
        If this class is constructed directly.
    """

    def init(self):
        raise NotImplementedError

    def align(self,
              sentence: str,
              tokens: List[str],
              return_all_hiddens=False,
              border_tokens=False) -> Tensor:
        """Aligns roberta embeddings to a tokenized sentence.
        This function is based on code from fairseq.
        https://github.com/pytorch/fairseq/blob/master/fairseq/models/roberta/hub_interface.py
        Parameters
        ----------
        sentence : str
            The input sentence string.
        tokens : List[str]
            The tokenization of `sentence` over which to align embeddings.
        return_all_hiddens : bool
            Whether to return the embeddings for all hidden layers of RoBERTa
            rather than just the last layer. Default: False.
        border_tokens : bool
            Whether to include special border token embeddings (<s>, </s>) in
            the return value. Default: False.
        Returns
        -------
        Tensor
            Roberta embeddings aligned with the input tokens
        """
        # tokenize both with GPT-2 BPE and get alignment with given tokens
        bpe_toks = self.model.encode(sentence)
        alignment = self._align_bpe_to_words(bpe_toks, tokens)

        # extract features and align them
        features = self.model.extract_features(bpe_toks, return_all_hiddens=return_all_hiddens)
        features = features.squeeze(0)  # Batch-size = 1
        aligned_feats = self._align_features_to_words(features, alignment)

        if border_tokens:
            return aligned_feats
        else:
            return aligned_feats[1:-1]  # exclude <s> and </s> tokens

    def _align_bpe_to_words(self, bpe_tokens: torch.LongTensor, other_tokens: List[str]):
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

        orig_bpe_tokens = bpe_tokens
        orig_other_tokens = other_tokens

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

    def _align_features_to_words(self, features, alignment):
        """Align features to word-generated alignment.
        Params
        ------
        features : Tensor
            Feature embeddings to align of shape `(T_bpe x C)`
        alignment
            The alignment between BPE tokens and words returned by the function
            `_align_bpe_to_words`.
        """
        assert features.dim() == 2

        bpe_counts = Counter(j for bpe_indices in alignment for j in bpe_indices)
        assert bpe_counts[0] == 0  # <s> shouldn't be aligned
        denom = features.new([bpe_counts.get(j, 1) for j in range(len(features))])
        weighted_features = features / denom.unsqueeze(-1)

        output = [weighted_features[0]]
        largest_j = -1
        for bpe_indices in alignment:
            output.append(weighted_features[bpe_indices].sum(dim=0))
            largest_j = max(largest_j, *bpe_indices)
        for j in range(largest_j + 1, len(features)):
            output.append(weighted_features[j])
        output = torch.stack(output)
        return output

class RoBERTaLarge(RoBERTa):
    def init(self):
        self.model = torch.hub.load('pytorch/fairseq', 'roberta.large')
        self.model.cuda()
        self.model.eval()
        log.info("Large RoBERTa Model loaded")


class RoBERTaBase(RoBERTa):
    def init(self):
        self.model = torch.hub.load('pytorch/fairseq', 'roberta.base')
        self.model.cuda()
        self.model.eval()
        log.info("Base RoBERTa Model loaded")

# class Seq2SeqRobertaEncoder(RoBERTaBase):

#     def __init__(self):
#         super(Seq2SeqRobertaEncoder, self).init()

#     def forward(self, input_ids, token_type_ids=None, attention_mask=None, known_tokens=None, border_tokens=False):
#         """
#         :param input_ids: same as it in BertModel
#         :param token_type_ids: same as it in BertModel
#         :param attention_mask: same as it in BertModel
#         :param output_all_encoded_layers: same as it in BertModel
#         :param token_subword_index: [batch_size, num_tokens, num_subwords]
#         :return:
#         """
        
#         # extract alignments
#         alignment = self._align_bpe_to_words(input_ids, known_tokens)

#         # hidden_state: [batch_size, num_subword_pieces, hidden_size]
#         last_hidden_state, pooled_output, hidden_states = super(Seq2SeqRobertaEncoder, self).forward(input_ids=input_ids, 
#                                                                                       attention_mask=attention_mask,
#                                                                                       token_type_ids=token_type_ids)
#         last_hidden_state = last_hidden_state.squeeze(0)
#         aligned_feats = self._align_features_to_words(last_hidden_state, alignment)
        
#         if border_tokens:
#             return aligned_feats
#         else:
#             return aligned_feats[1:-1]  # exclude <s> and </s> tokens

#         return aligned_feats

#     def align(self,
#               sentence: str,
#               tokens: List[str],
#               return_all_hiddens=False,
#               border_tokens=False) -> Tensor:
#         """Aligns roberta embeddings to a tokenized sentence.
#         This function is based on code from fairseq.
#         https://github.com/pytorch/fairseq/blob/master/fairseq/models/roberta/hub_interface.py
#         Parameters
#         ----------
#         sentence : str
#             The input sentence string.
#         tokens : List[str]
#             The tokenization of `sentence` over which to align embeddings.
#         return_all_hiddens : bool
#             Whether to return the embeddings for all hidden layers of RoBERTa
#             rather than just the last layer. Default: False.
#         border_tokens : bool
#             Whether to include special border token embeddings (<s>, </s>) in
#             the return value. Default: False.
#         Returns
#         -------
#         Tensor
#             Roberta embeddings aligned with the input tokens
#         """
#         # tokenize both with GPT-2 BPE and get alignment with given tokens
#         bpe_toks = self.model.encode(sentence)
#         alignment = self._align_bpe_to_words(bpe_toks, tokens)

#         # extract features and align them
#         features = self.model.extract_features(bpe_toks, return_all_hiddens=return_all_hiddens)
#         features = features.squeeze(0)  # Batch-size = 1
#         aligned_feats = self._align_features_to_words(features, alignment)

#         if border_tokens:
#             return aligned_feats
#         else:
#             return aligned_feats[1:-1]  # exclude <s> and </s> tokens

#     def _align_bpe_to_words(self, bpe_tokens: torch.Tensor, other_tokens: List[str]):
#         """Align GPT-2 BPE to other tokenization formats (e.g., spaCy).
#         This function is based on code from fairseq.
#         https://github.com/pytorch/fairseq/blob/master/fairseq/models/roberta/alignment_utils.py
#         Params
#         ------
#         bpe_tokens : torch.LongTensor
#             GPT-2 BPE tokens of shape `(T_bpe)`
#         other_tokens : List[str]
#             other tokens of shape `(T_words)`
#         Returns
#         -------
#         List[str]
#             The mapping from `other_tokens` to corresponding `bpe_tokens`.
#         Raises
#         ------
#         ValueError
#             If `bpe_tokens` and `other_tokens` cannot be aligned.
#         """
#         print('roberta', bpe_tokens)
#         print('other', other_tokens)
#         assert bpe_tokens.dim() == 1
#         assert bpe_tokens[0] == 0.  # added after revision in alignment utils from fairseq (Feb11, 2020)

#         def clean(text):
#             return text.strip()

#         orig_bpe_tokens = bpe_tokens
#         orig_other_tokens = other_tokens

#         # remove whitespaces to simplify alignment
#         bpe_tokens = [self.model.task.source_dictionary.string([x]) for x in bpe_tokens]
#         bpe_tokens = [clean(self.model.bpe.decode(x) if x not in {'<s>', ''} else x) for x in bpe_tokens]
#         other_tokens = [clean(str(o)) for o in other_tokens]

#         # strip leading <s>
#         bpe_tokens = bpe_tokens[1:]
#         assert ''.join(bpe_tokens) == ''.join(other_tokens)

#         # create alignment from every word to a list of BPE tokens
#         alignment = []
#         bpe_toks = filter(lambda item: item[1] != '', enumerate(bpe_tokens, start=1))
#         j, bpe_tok = next(bpe_toks)
#         for other_tok in other_tokens:
#             bpe_indices = []
#             while True:
#                 if other_tok.startswith(bpe_tok):
#                     bpe_indices.append(j)
#                     other_tok = other_tok[len(bpe_tok):]
#                     try:
#                         j, bpe_tok = next(bpe_toks)
#                     except StopIteration:
#                         j, bpe_tok = None, None
#                 elif bpe_tok.startswith(other_tok):
#                     # other_tok spans multiple BPE tokens
#                     bpe_indices.append(j)
#                     bpe_tok = bpe_tok[len(other_tok):]
#                     other_tok = ''
#                 else:
#                     raise ValueError('Cannot align "{}" and "{}"'.format(other_tok, bpe_tok))
#                 if other_tok == '':
#                     break
#             assert len(bpe_indices) > 0
#             alignment.append(bpe_indices)
#         assert len(alignment) == len(other_tokens)
#         return alignment

#     def _align_features_to_words(self, features, alignment):
#         """Align features to word-generated alignment.
#         Params
#         ------
#         features : Tensor
#             Feature embeddings to align of shape `(T_bpe x C)`
#         alignment
#             The alignment between BPE tokens and words returned by the function
#             `_align_bpe_to_words`.
#         """
#         assert features.dim() == 2

#         bpe_counts = Counter(j for bpe_indices in alignment for j in bpe_indices)
#         assert bpe_counts[0] == 0  # <s> shouldn't be aligned
#         denom = features.new([bpe_counts.get(j, 1) for j in range(len(features))])
#         weighted_features = features / denom.unsqueeze(-1)

#         output = [weighted_features[0]]
#         largest_j = -1
#         for bpe_indices in alignment:
#             output.append(weighted_features[bpe_indices].sum(dim=0))
#             largest_j = max(largest_j, *bpe_indices)
#         for j in range(largest_j + 1, len(features)):
#             output.append(weighted_features[j])
#         output = torch.stack(output)
#         return output
