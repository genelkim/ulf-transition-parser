from overrides import overrides
import torch
from torch.nn.utils.rnn import pad_packed_sequence
from typing import Optional, Tuple

from stog.utils.checks import ConfigurationError
from stog.modules.encoder_base import RnnStateStorage
from stog.modules.seq2seq_encoders.seq2seq_encoder import Seq2SeqEncoder


class PytorchIncrSeq2SeqWrapper(Seq2SeqEncoder):
    """A wrapper for incremental parsing on seq2seq models.

    Exactly the same functionality as
    stog.modules.seq2seq_encoders.pytorch_seq2seq_wrapper.PytorchSeq2SeqWrapper
    but returns the final states so that the computation can be continued. The
    `stateful` parameters is removed and this acts like `stateful` = True
    except that if a hidden state is provided, it overwrites self._states.
    This is mainly for the purpose of accomodating beam_search, while keeping
    the same API as PytorchSeq2SeqWrapper. Thus it is an incremental seq2seq
    wrapper.
    """
    def __init__(self, module: torch.nn.Module) -> None:
        super(PytorchIncrSeq2SeqWrapper, self).__init__(stateful=True)
        self._module = module
        try:
            if not self._module.batch_first:
                raise ConfigurationError("Our encoder semantics assumes batch is always first!")
        except AttributeError:
            pass

        try:
            self._is_bidirectional = self._module.bidirectional
        except AttributeError:
            self._is_bidirectional = False
        if self._is_bidirectional:
            self._num_directions = 2
        else:
            self._num_directions = 1

    @overrides
    def get_input_dim(self) -> int:
        return self._module.input_size

    @overrides
    def get_output_dim(self) -> int:
        return self._module.hidden_size * self._num_directions

    @overrides
    def is_bidirectional(self) -> bool:
        return self._is_bidirectional

    @overrides
    def forward(self,  # pylint: disable=arguments-differ
                inputs: torch.Tensor,
                mask: torch.Tensor,
                states: Optional[RnnStateStorage] = None) -> Tuple[torch.Tensor, Optional[RnnStateStorage]]:

        if mask is None:
            raise ValueError("Always pass a mask with stateful RNNs.")
        if states is not None:
            self._states = states

        if mask is None:
            return self._module(inputs, None)[0]

        batch_size, total_sequence_length = mask.size()

        packed_sequence_output, final_states, restoration_indices = \
            self.sort_and_run_forward(self._module, inputs, mask, None)

        unpacked_sequence_tensor, _ = pad_packed_sequence(packed_sequence_output, batch_first=True)

        num_valid = unpacked_sequence_tensor.size(0)
        # Some RNNs (GRUs) only return one state as a Tensor.  Others (LSTMs) return two.
        # If one state, use a single element list to handle in a consistent manner below.
        if not isinstance(final_states, (list, tuple)) and self.stateful:
            final_states = [final_states]

        # Add back invalid rows.
        if num_valid < batch_size:
            _, length, output_dim = unpacked_sequence_tensor.size()
            zeros = unpacked_sequence_tensor.new_zeros(batch_size - num_valid, length, output_dim)
            unpacked_sequence_tensor = torch.cat([unpacked_sequence_tensor, zeros], 0)

            # The states also need to have invalid rows added back.
            new_states = []
            for state in final_states:
                num_layers, _, state_dim = state.size()
                zeros = state.new_zeros(num_layers, batch_size - num_valid, state_dim)
                new_states.append(torch.cat([state, zeros], 1))
            final_states = new_states

        # It's possible to need to pass sequences which are padded to longer than the
        # max length of the sequence to a Seq2SeqEncoder. However, packing and unpacking
        # the sequences mean that the returned tensor won't include these dimensions, because
        # the RNN did not need to process them. We add them back on in the form of zeros here.
        sequence_length_difference = total_sequence_length - unpacked_sequence_tensor.size(1)
        if sequence_length_difference > 0:
            zeros = unpacked_sequence_tensor.new_zeros(batch_size,
                                                       sequence_length_difference,
                                                       unpacked_sequence_tensor.size(-1))
            unpacked_sequence_tensor = torch.cat([unpacked_sequence_tensor, zeros], 1)

        self._update_states(final_states, restoration_indices)

        # Restore the original indices and return the sequence.
        return unpacked_sequence_tensor.index_select(0, restoration_indices), self._states

