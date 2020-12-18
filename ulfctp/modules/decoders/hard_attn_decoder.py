# pylint: disable=C0114
# Instead of module docstring, use a class docstring.
# pylint: disable=E1101
# Errors in torch attributes
# pylint: disable=C0305
# Trailing newlines (I want a single trailing newline)
# pylint: disable=R0913
# Don't care about the number of attributes

import math
import torch

from stog.utils.nn import masked_log_softmax
from stog.modules.stacked_lstm import StackedLstm
from stog.utils.logging import init_logger
from ulfctp.modules.seq2seq_encoders.pytorch_incr_seq2seq_wrapper import PytorchIncrSeq2SeqWrapper
from ulfctp.metrics.seq2seq_metrics import Seq2SeqMetrics
from ulfctp.modules.decoders.mlp import MLP

logger = init_logger()

class HardAttnDecoder(torch.nn.Module):
    """
    A hard attention decoder built specifically for decoding a cache transition system
    with an input text sequence and input buffer sequence and generates actions.

    Actions are generated through a unidirectional LSTM with given word embeddings,
    buffer symbol embeddings, transition system feature embeddings, and previous
    action hidden state embeddings.

    The model takes tensors for the word and symbol embeddings and introduces hard
    attention to the inputs through provided word and symbol indices at each step.
    The indexing should be generated through the cache transition system itself.

    Loss: This model currently supports negative log loss.

    """

    def __init__(
            self,
            encoder,
            decoder,
            action_vocab_size,
            vocab,
            use_ts_feats=True,
    ):
        super(HardAttnDecoder, self).__init__()

        # LSTM
        self.encoder = encoder
        # Linear layer
        self.decoder = decoder
        self.action_vocab_size = action_vocab_size

        self._MAX_EXP = 100

        self.pad_idx = -1
        self._criterion = torch.nn.NLLLoss(
            ignore_index=self.pad_idx, reduction='sum'
        )
        self.metrics = Seq2SeqMetrics()
        self.vocab = vocab

        # Whether to use transition system features.
        self.use_ts_feats = use_ts_feats


    def forward_for_training(self, w_memory_bank, s_memory_bank, mask, ts_feats,
                             action_idx, action_word_idx, action_symbol_idx,
                             action_embeddings, feat_embeddings, word_focus_idx=None, symbol_focus_idx=None):
        """Generates the full action sequence from the given gold action sequence and
        then computes the loss and training metrics.

        w_memory_bank: word encoder output - [batch_size, num_tokens, encoder_output_size]
        s_memory_bank: symbol encoder output - [batch_size, num_symbols, decoder_hidden_size]
        mask: action length mask - [batch_size, num_actions]
        ts_feats: transition system features per action - [batch_size, num_actions, num_features]
        action_idx: action indices per action - [batch_size, num_actions + 1]
        action_word_idx: word indices (w.r.t. sentence) per action
        action_symbol_idx: symbol indices (w.r.t. sentence) per action
        word_focus_idx: focus indices (left_cache, right_cache) (w.r.t. sentence) per action (set to None for testing purpose)
        symbol_focus_idx: focus indices (left_cache, right_cache) (w.r.t. sentence) per action (set to None for testing purpose)
        action_,word_,symbol_,feat_embeddings: embedding dictionairs to convert indexes to
            embeddings
        """
        batch_size = action_idx.shape[0]
        num_actions = action_idx.shape[1] - 1
        # Input actions: Strip off the end sequence token.
        prev_action_idx = action_idx[:, :-1].contiguous()
        # Output actions: Strip off the start sequence token.
        post_action_idx = action_idx[:, 1:].contiguous()
        assert prev_action_idx.shape == post_action_idx.shape
        prev_action_embs = action_embeddings(prev_action_idx)
        if self.use_ts_feats:
            # [batch_size, num_actions, num_features, feature_dim]
            feat_embs = feat_embeddings(ts_feats)
            # [batch_size, num_actions, num_features x feature_dim]
            feat_embs = feat_embs.view(batch_size, num_actions, -1)
        else:
            feat_embs = None

        # Go from the example-specific token indexing to the token embeddings.
        # action_word_idx - [batch_size, num_actions]
        # w_memory_bank - [batch_size, num_tokens, encoder_output_size]
        encoder_output_size = w_memory_bank.shape[2]
        # word_embs - [batch_size, num_actions, encoder_output_size]
        word_embs = prev_action_embs.new_zeros(
            (batch_size, num_actions, encoder_output_size))

        # Add a padding embedding at the end of every sequence.
        # -1 and all values above the actual sequence is considered padding index.
        #   t.gather should work, but I'm not completely sure yet how to do it.
        for batch_idx in range(batch_size):
            widx = action_word_idx[batch_idx]
            word_embs[batch_idx] = w_memory_bank[batch_idx][widx]

        # Go from example-specific symbol indexing to the symbol embeddings.
        # action_symbol_idx - [batch_size, num_actions]
        # s_memory_bank - [batch_size, num_symbols, decoder_hidden_size]
        decoder_hidden_size = s_memory_bank.shape[2]
        max_tgt_len = s_memory_bank.shape[1]
        # symbol_embs - [batch_size, num_actions, decoder_hidden_size]
        symbol_embs = prev_action_embs.new_zeros(
            (batch_size, num_actions, decoder_hidden_size))
        for batch_idx in range(batch_size):
            sidx = action_symbol_idx[batch_idx]
            symbol_embs[batch_idx] = s_memory_bank[batch_idx][sidx]

        encoded_states, _ = self.encode(
            None, word_embs, symbol_embs, feat_embs, prev_action_embs, mask)

        if word_focus_idx is not None and symbol_focus_idx is not None:
            # word focus embeddings
            # embeddings are in order [left_word_embs, right_word_embs] and so on
            left_word_focus_idx, right_word_focus_idx = word_focus_idx
            assert(len(left_word_focus_idx)==len(right_word_focus_idx))
            word_embs_size = w_memory_bank.shape[2]

            word_focus_embs = prev_action_embs.new_zeros((batch_size, num_actions, word_embs_size*2))
                
            for batch_idx in range(batch_size):
                lwfidx = left_word_focus_idx[batch_idx]
                rwfidx = right_word_focus_idx[batch_idx]
                word_focus_embs[batch_idx] = torch.cat([w_memory_bank[batch_idx][lwfidx], w_memory_bank[batch_idx][rwfidx]], dims=2)


            # symbol focus embeddings
            # embeddings are in order [left_symbol_embs, right_symbol_embs] and so on
            left_symbol_focus_idx, right_symbol_focus_idx = symbol_focus_idx
            assert(len(left_symbol_focus_idx)==len(right_symbol_focus_idx))
            symbol_embs_size = s_memory_bank.shape[2]

            symbol_focus_embs = prev_action_embs.new_zeros((batch_size, num_actions, symbol_embs_size*2))
                
            for batch_idx in range(batch_size):
                lsfidx = left_symbol_focus_idx[batch_idx]
                rsfidx = right_symbol_focus_idx[batch_idx]
                symbol_focus_embs[batch_idx] = torch.cat([s_memory_bank[batch_idx][lsfidx],s_memory_bank[batch_idx][rsfidx]], dims=2)

            encoded_states, _ = self.encode(None, word_focus_embs, symbol_focus_embs, feat_embs, prev_action_embs, mask)
            

        # Generate predictions.
        scores = self.decode(encoded_states)
        # Compute loss.
        loss_output = self.get_loss(scores, post_action_idx, mask)
        return dict(
            scores=scores,
            loss=loss_output['loss'],
        )

    def forward(self, state, w, s, f, a, mask):
        """
        Generates inferences from the previous LSTM states, current word
        and symbol embeddings, and previous action states. Returns the
        hidden and final states of the encoder as well as the decoded raw
        scores and log probabilities.

        state: previous lstm state, RnnStateStorage (or None)
        w: word memory embedding (output of word encoder for current attention)
            (batch_size, num_actions, encoder_hidden_dim)
        s: symbol memory embedding (output of symbol encoder for current attention)
            (batch_size, num_actions, decoder_hidden_dim)
        f: transition system feature embeddings, (batch_size, num_actions, feat_dim)
        a: previous action state, (batch_size, num_actions, action_state_dim)
        mask : (batch_size, num_actions)

        Returns
            hidden_states: hidden states of the inputs
            final_states: final encoder hidden states (for continued encoding)
            scores: raw decoder scores
            log_probs: predicted log probabilities
        """
        hidden_states, final_states = self.encode(state, w, s, f, a, mask)
        scores = self.decode(hidden_states)
        log_probs = masked_log_softmax(scores, mask.unsqueeze(2), dim=2)
        return hidden_states, final_states, scores, log_probs

    def encode(self, state, w, s, f, a, mask):
        """Generates inferences from the previous LSTM state, current step word
        and symbol memory banks (from hard attention), and previous action
        states. Examples that include the first step in the inference can give
        None for the previous lstm state (`s` parameter).

        state: previous lstm state, RnnStateStorage (or None)
        w: word memory embedding (output of word encoder for current attention)
            (batch_size, num_actions, encoder_hidden_dim)
        s: symbol memory embedding (output of symbol encoder for current attention)
            (batch_size, num_actions, decoder_hidden_dim)
        f: transition system feature embeddings, (batch_size, num_actions, feat_dim)
        a: previous action state, (batch_size, num_actions, action_state_dim)
        mask : (batch_size, num_actions)

        Returns hidden states and the final/next encoder states. The final/next
        encoder states can be used as the `s` parameter in a future call for
        incremental inference.
        """
        embs = [w, s, f, a] if self.use_ts_feats else [w, s, a]
        inputs = torch.cat(embs, dim=2)
        # hidden_states : (batch_size, 1, hard_attn_hidden_size)
        # new_states : Optional[RnnStorage]
        hidden_states, final_states = self.encoder(inputs, mask, state)
        return hidden_states, final_states

    def decode(self, encoded_states):
        """
        Runs the states encoded using the forward function through the decoder.
        """
        return self.decoder(encoded_states)

    def get_loss(self, scores, targets, mask):
        """
        scores: [batch, max_dec_steps, action_vocab_size]
        targets: [batch, max_dec_steps] (transition parser action indices)
        mask: [batch, max_dec_steps]

        Assume mask is 0/1, where 1 is occurs.
        """
        # Get some named parameters
        batch_size, max_dec_steps, _ = scores.shape

        # Add padding indices to the mask.
        non_pad = (targets != self.pad_idx).type(torch.cuda.BoolTensor)
        new_mask = non_pad & mask.type(torch.cuda.BoolTensor)
        # Get log likelihood from neural net output taking the mask into account.
        log_likelihood = masked_log_softmax(scores, new_mask.unsqueeze(2), dim=2)
        _, predictions = log_likelihood.max(2)

        # Flatten out tensors for easier computation.
        flat_targets = targets.view(-1)
        flat_logs = log_likelihood.view(batch_size * max_dec_steps, -1)

        # Log likelihoods of target values and sum to get negative loss.
        target_logs = flat_logs[range(len(flat_targets)), flat_targets]
        loss = -torch.clamp(
            target_logs.masked_select(mask.eq(1).view(-1)),
            -self._MAX_EXP,
            0,
        ).sum()

        # Compute metrics.
        flat_mask = new_mask.view(-1)
        flat_preds = predictions.view(-1)
        num_correct = flat_preds.eq(flat_targets).masked_select(flat_mask.type(torch.cuda.BoolTensor)).sum().item()
        num_unfiltered = flat_mask.sum().item()
        self.metrics(loss.item(), num_unfiltered, num_correct)

        return dict(
            loss=loss,
            scaled_loss=loss.div(float(num_unfiltered)),
            predictions=predictions,
        )

    @classmethod
    def from_params(cls, vocab, params):
        """
        Generate an instance of this class from Vocab object and configuration parameters.
        """
        logger.info("hard_attn_decoder params: {}".format(params))
        # Unidirectional LSTM since this is decoding in sequence.
        encoder = PytorchIncrSeq2SeqWrapper(
            module=StackedLstm.from_params(params['hard_attn_encoder'])
        )
        # decoder = torch.nn.Sequential(
        #     torch.nn.Linear(encoder.get_output_dim(), params['action_vocab_size']),
        # )

        decoder = MLP(encoder.get_output_dim(),
                      params['mlp_decoder']['hidden_dim'],
                      params['action_vocab_size'],
                      params['mlp_decoder']['n_layers'],
                      params['mlp_decoder']['dropout'])

        return cls(
            encoder=encoder,
            decoder=decoder,
            action_vocab_size=params['action_vocab_size'],
            vocab=vocab,
            use_ts_feats=params['use_ts_feats'],
        )

