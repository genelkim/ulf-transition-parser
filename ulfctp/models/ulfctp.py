from typing import Optional, List
from collections import defaultdict, namedtuple
import torch
from torch import Tensor

# STOG imports.
from stog.data.vocabulary import DEFAULT_OOV_TOKEN, DEFAULT_PADDING_TOKEN
from stog.data.tokenizers.character_tokenizer import CharacterTokenizer
from stog.models.model import Model
from stog.modules.attention import DotProductAttention
from stog.modules.attention import MLPAttention
from stog.modules.attention import BiaffineAttention
from stog.modules.attention_layers.global_attention import GlobalAttention
from stog.modules.decoders.generator import Generator
from stog.modules.decoders.pointer_generator import PointerGenerator
from stog.modules.decoders.rnn_decoder import InputFeedRNNDecoder
from stog.modules.input_variational_dropout import InputVariationalDropout
from stog.modules.seq2seq_encoders import Seq2SeqBertEncoder
from stog.modules.seq2vec_encoders.cnn_encoder import CnnEncoder
from stog.modules.seq2seq_encoders.pytorch_seq2seq_wrapper import PytorchSeq2SeqWrapper
from stog.modules.stacked_bilstm import StackedBidirectionalLstm
from stog.modules.stacked_lstm import StackedLstm
from stog.modules.token_embedders.embedding import Embedding
from stog.utils.logging import init_logger
from stog.utils.nn import get_text_field_mask
# Custom ULFCTP imports.
from ulfctp.modules.seq2seq_encoders.pytorch_incr_seq2seq_wrapper import PytorchIncrSeq2SeqWrapper
from ulfctp.modules.decoders.hard_attn_decoder import HardAttnDecoder
from ulfctp.utils.string import START_SYMBOL, END_SYMBOL, NONE_SYMBOL
# RoBERTa
from ulfctp.modules.seq2seq_encoders.seq2seq_roberta_encoder import RoBERTaBase
# The following imports are added for mimick testing.
import os
import subprocess
import math
from stog.utils.nn import masked_log_softmax
from stog.predictors.predictor import Predictor
from ulfctp.commands.predict import _PredictManager
from ulfctp.data.dataset_builder import load_dataset_reader
# Transition parser related imports.
from hypothesis import Hypothesis, generateAMR, sort_hypotheses
from oracle.cacheConfiguration import CacheConfiguration
from oracle.cacheTransition import CacheTransition
from oracle.utils import ReserveGenMethod, UnreserveGenMethod, FeatureType
import oracle.utils
import NP2P_data_stream
# Interface to Common Lisp function for mapping back to plain ULFs.
from lisp.amr2ulf import amr2ulf
from lexicon.ulf_lexicon import ULFLexicon

logger = init_logger()

DecoderStateEmbs = namedtuple('DecoderStateEmbs', ['hard_attn_decoder', 'symbol_decoder'])


class ULFCTP(Model):
    """The ULF cache transition parser.
    """
    def __init__(self,
                 # General parameters
                 vocab,
                 punctuation_ids,
                 use_pos,
                 use_char_cnn,
                 use_ts_feats,
                 use_coverage,
                 use_bert,
                 use_roberta,
                 #max_decode_length,
                 max_action_length,
                 topk_size,
                 use_symbol_inputs,
                 beam_size,
                 transition_system_params,
                 # Encoder
                 bert_encoder,
                 encoder_token_embedding,
                 encoder_pos_embedding,
                 encoder_lemma_embedding,
                 encoder_char_embedding,
                 encoder_char_cnn,
                 encoder_embedding_dropout,
                 encoder,
                 encoder_output_dropout,
                 # symbol encoder (decoder)
                 decoder_token_embedding,
                 decoder_pos_embedding,
                 decoder_char_embedding,
                 decoder_char_cnn,
                 decoder_embedding_dropout,
                 decoder,
                 decoder_output_dropout,
                 # Hard attention decoder
                 hard_attn_decoder,
                 test_config,
                 action_embedding,
                 feat_embedding,
                 ):
        super(ULFCTP, self).__init__()

        self.vocab = vocab
        self.punctuation_ids = punctuation_ids
        self.use_pos = use_pos
        self.use_char_cnn = use_char_cnn
        self.use_ts_feats = use_ts_feats
        self.use_coverage = use_coverage
        self.use_bert = use_bert
        self.use_roberta = use_roberta
        #self.max_decode_length = max_decode_length
        self.max_action_length = max_action_length
        self.topk_size = topk_size
        self.beam_size = beam_size
        self.use_symbol_inputs = use_symbol_inputs

        self.transition_system_params = transition_system_params

        self.sanity_check_arcs = transition_system_params["type_checking_method"] == "sanity_checker"
        if transition_system_params['lexicon_file'].lower() == "none":
            print("No ULF lexicon...")
            self.ulf_lexicon = None
        else:
            print("Loading ULF lexicon...")
            self.ulf_lexicon = ULFLexicon(transition_system_params['lexicon_file'], transition_system_params['strict_lexicon'])

        self.bert_encoder = bert_encoder

        self.encoder_token_embedding = encoder_token_embedding
        self.encoder_pos_embedding = encoder_pos_embedding
        self.encoder_lemma_embedding = encoder_lemma_embedding
        self.encoder_char_embedding = encoder_char_embedding
        self.encoder_char_cnn = encoder_char_cnn
        self.encoder_embedding_dropout = encoder_embedding_dropout
        self.encoder = encoder
        self.encoder_output_dropout = encoder_output_dropout

        self.decoder_token_embedding = decoder_token_embedding
        self.decoder_pos_embedding = decoder_pos_embedding
        self.decoder_char_embedding = decoder_char_embedding
        self.decoder_char_cnn = decoder_char_cnn
        self.decoder_embedding_dropout = decoder_embedding_dropout
        self.decoder = decoder
        self.decoder_output_dropout = decoder_output_dropout

        self.hard_attn_decoder = hard_attn_decoder

        self.test_config = test_config

        self.action_embedding = action_embedding
        self.feat_embedding = feat_embedding

    def set_beam_size(self, beam_size):
        self.beam_size = beam_size

    def set_decoder_token_indexers(self, token_indexers):
        self.decoder_token_indexers = token_indexers
        self.character_tokenizer = CharacterTokenizer()

    def get_metrics(self, reset: bool = False, mimick_test: bool = False,
            epoch: int = 0, decode_only: bool = False):
        metrics = dict()
        if mimick_test and self.test_config:
            logger.info("mimick_test")
            metrics = self.mimick_test(epoch)
        if not decode_only:
            hard_attn_decoder_metrics = self.hard_attn_decoder.metrics.get_metric(reset)
            metrics.update(hard_attn_decoder_metrics)
        if 'EL_F1' not in metrics:
            metrics['EL_F1'] = -0.1
        if 'SEMBLEU_PREC' not in metrics:
            metrics['SEMBLEU_PREC'] = -0.1
        return metrics

    def get_mimick_prediction_path(self, epoch):
        prediction_file = '{}.epoch{}.beamsize{}.topk{}'.format(
                self.test_config['prediction_basefile'],
                epoch,
                self.beam_size,
                self.topk_size,
        )
        if self.sanity_check_arcs:
            prediction_file = prediction_file + ".sanity"
        if self.use_bert:
            prediction_file = prediction_file + ".bert"
        if self.use_roberta:
            prediction_file = prediction_file + ".roberta"
        prediction_dir = os.path.join(self.test_config['serialization_dir'], 'predictions')
        if not os.path.isdir(prediction_dir):
            os.mkdir(prediction_dir)
        prediction_path = os.path.join(prediction_dir, prediction_file)
        return prediction_path

    def mimick_test(self, epoch: int = 0):
        # This is always called during the dev stage of training, so no need to call self.eval()
        word_splitter = None
        if self.use_bert:
            word_splitter = self.test_config.get('word_splitter', None)
        dataset_reader = load_dataset_reader(
            self.test_config.get('data_type', 'ULF'),
            word_splitter=word_splitter,
            use_roberta=self.test_config.get('use_roberta', False)
        )
        dataset_reader.set_evaluation()
        # This needs to be imported since it's not packaged with stog.predictors.
        # Loading it registers it so that it can be loaded by name.
        from ulfctp.predictors import ULFCTPPredictor
        predictor = Predictor.by_name('ULFCTP')(self, dataset_reader)
        prediction_basepath = self.get_mimick_prediction_path(epoch)
        manager = _PredictManager(
            predictor,
            self.test_config['data'],
            prediction_basepath,
            self.test_config['batch_size'],
            False, # Print to console.
            True, # Has data reader.
            self.beam_size,
        )
        try:
            logger.info('Mimicking test...')
            manager.run()
        except Exception as e:
            logger.info('Exception threw out when running the manager.')
            logger.error(e, exc_info=True)
            #return {}
            raise e
        retval = {}
        try:
            logger.info('Computing the ELSmatch score...')
            el_result = subprocess.check_output([
                self.test_config['el_smatch_eval_script'],
                self.test_config['smatch_dir'],
                os.path.join(os.path.dirname(self.test_config['data']), "ulf.amr-format"),
                manager._output_file,
            ]).decode().split()
            logger.info("raw el_result: {}".format(el_result))
            el_result = list(map(float, el_result))
            retval.update(dict(
                EL_PREC=el_result[0]*100,
                EL_REC=el_result[1]*100,
                EL_F1=el_result[2]*100,
            ))
        except Exception as e:
            logger.info('Exception threw out when computing EL-smatch')
            logger.error(e, exc_info=True)
        try:
            logger.info('Computing the Raw Smatch score...')
            raw_result = subprocess.check_output([
                self.test_config['smatch_eval_script'],
                self.test_config['smatch_dir'],
                os.path.join(os.path.dirname(self.test_config['data']), "ulf.amr-format"),
                manager._output_file,
            ]).decode().split()
            logger.info("raw raw_result: {}".format(raw_result))
            raw_result = list(map(float, raw_result))
            retval.update(dict(
                RAW_PREC=raw_result[0]*100,
                RAW_REC=raw_result[1]*100,
                RAW_F1=raw_result[2]*100,
            ))
        except Exception as e:
            logger.info('Exception threw out when computing smatch.')
            logger.error(e, exc_info=True)
        try:
            logger.info('Computing the SemBleu score...')
            sembleu_result = subprocess.check_output([
                self.test_config['sembleu_eval_script'],
                self.test_config['smatch_dir'],
                os.path.join(os.path.dirname(self.test_config['data']), "ulf.amr-format"),
                manager._output_file,
            ]).decode().split()
            logger.info("raw sembleu_result: {}".format(sembleu_result))
            sembleu_result = list(map(float, sembleu_result))
            retval.update(dict(
                SEMBLEU_PREC=sembleu_result[0]*100
            ))
        except Exception as e:
            logger.info('Exception threw out when computing sembleu.')
            logger.error(e, exc_info=True)
        return retval


    def prepare_batch_input(self, batch):
        """Group batch parameters into inputs for the encoder, decoder, and actions.
        """
        # [batch, num_tokens]
        bert_token_inputs = batch.get('src_token_ids', None)
        if bert_token_inputs is not None:
            bert_token_inputs = bert_token_inputs.long()
        encoder_token_subword_index = batch.get('src_token_subword_index', None)
        if encoder_token_subword_index is not None:
            encoder_token_subword_index = encoder_token_subword_index.long()
        encoder_token_inputs = batch['src_tokens']['encoder_tokens']
        encoder_pos_tags = batch['src_pos_tags']
        # [batch, num_tokens, num_chars]
        encoder_char_inputs = batch['src_tokens']['encoder_characters']
        # [batch, num_tokens]
        encoder_mask = get_text_field_mask(batch['src_tokens'])
        encoder_lemma_inputs = batch['src_lemmas']

        encoder_inputs = dict(
            bert_token=bert_token_inputs,
            token_subword_index=encoder_token_subword_index,
            token=encoder_token_inputs,
            lemma=encoder_lemma_inputs,
            pos_tag=encoder_pos_tags,
            char=encoder_char_inputs,
            mask=encoder_mask
        )

        # Get token, char, pos, and mask for symbols
        # [batch, num_tokens]
        decoder_token_inputs = batch['tgt_tokens']['decoder_tokens'].contiguous()
        decoder_pos_tags = batch['tgt_pos_tags']
        # [batch, num_tokens, num_chars]
        decoder_char_inputs = batch['tgt_tokens']['decoder_characters'].contiguous()
        # [batch, num_decode_tokens]
        decoder_mask = get_text_field_mask(batch['tgt_tokens'])
        decoder_inputs = dict(
            token=decoder_token_inputs,
            pos_tag=decoder_pos_tags,
            char=decoder_char_inputs,
            mask=decoder_mask,
        )

        # Inputs relating the transition parser actions.
        # Strip off the useless end features, word_idx, and symbol_idx,
        # see the function `get_list_data` in ulfctp/data/dataset_readers/ulf_parsing/ulfamr.py
        # for details. It is used by ulfctp/data/dataset_readers/underspecified_episodic_logic.py
        if self.use_ts_feats:
            ts_feats = batch['ts_feats'][:,:-1][:,:self.max_action_length].contiguous()
        else:
            ts_feats = None
        action_idx = batch['action_tokens']['action_tokens'][:,:self.max_action_length+1].contiguous()
        word_idx = batch['action_word_tokens'][:,:-1][:,:self.max_action_length].contiguous()
        symbol_idx = batch['action_symbol_tokens'][:,:-1][:,:self.max_action_length].contiguous()
        action_mask = get_text_field_mask(batch['action_tokens'])[:,:-1][:,:self.max_action_length].contiguous()
        action_inputs = dict(
            ts_feats=ts_feats,
            action_idx=action_idx,
            word_idx=word_idx,
            symbol_idx=symbol_idx,
            mask=action_mask,
        )

        return encoder_inputs, decoder_inputs, action_inputs

    def forward(self, batch, for_training=False):
        encoder_inputs, decoder_inputs, action_inputs = self.prepare_batch_input(batch)
        encoder_outputs = self.encode(
            encoder_inputs['bert_token'],
            encoder_inputs['token_subword_index'],
            encoder_inputs['token'],
            encoder_inputs['lemma'],
            encoder_inputs['pos_tag'],
            encoder_inputs['char'],
            encoder_inputs['mask'],
        )
        # Decoder is run for dev/test also if we have symbols in the input.
        if for_training or self.use_symbol_inputs:
            decoder_outputs = self.get_symbol_embeddings(
                None,
                decoder_inputs['token'],
                decoder_inputs['pos_tag'],
                decoder_inputs['char'],
                decoder_inputs['mask'],
            )
        if for_training:
            hard_attn_decoder_outputs = self.hard_attn_decoder.forward_for_training(
                # Sentence encoder results. [batch, num_tokens, encoder_output_size]
                encoder_outputs['memory_bank'],
                # symbol encoder results. [batch, num_symbols, decoder_hidden_size]
                decoder_outputs['memory_bank'],
                # Action lengths.
                action_inputs['mask'],
                # Transition system features.
                action_inputs['ts_feats'],
                action_inputs['action_idx'],
                action_inputs['word_idx'],
                action_inputs['symbol_idx'],
                self.action_embedding,
                self.feat_embedding,
            )
            return dict(loss=hard_attn_decoder_outputs['loss'])
        else:
            return dict(
                encoder_memory_bank=encoder_outputs['memory_bank'],
                encoder_final_states=encoder_outputs['final_states'],
                decoder_outputs=decoder_outputs if self.use_symbol_inputs else None,
                encoder_inputs=encoder_inputs,
                decoder_inputs=decoder_inputs,
                action_inputs=action_inputs,
                raw_batch=batch,
            )

    def encode(self, bert_tokens, token_subword_index, tokens, lemmas, pos_tags, chars, mask):
        """Puts text input through the encoder.
        Returns the encoder outputs and final encoder states.
        """
        # [batch, num_tokens, embedding_size]
        encoder_inputs = []
        if self.use_bert:
            bert_mask = bert_tokens.ne(0)
            bert_embeddings, _ = self.bert_encoder(
                bert_tokens,
                attention_mask=bert_mask,
                output_all_encoded_layers=False,
                token_subword_index=token_subword_index,
            )
            if token_subword_index is None:
                bert_embeddings = bert_embeddings[:, 1:-1]
            # Add zero tokens corresponding to @end@, @none@, and @null@.
            zero_cols = torch.zeros(
                    bert_embeddings.shape[0],
                    3,
                    bert_embeddings.shape[2]).to(bert_embeddings.device)
            bert_embeddings = torch.cat([bert_embeddings, zero_cols], 1)
            encoder_inputs += [bert_embeddings]

        if self.use_roberta:
            # roberta_mask = bert_tokens.ne(0)
            # roberta_embeddings, _ = self.bert_encoder(
            #     bert_tokens,
            #     attention_mask=roberta_mask,
            #     known_tokens=tokens
            # )
            bert_tokens = torch.Tensor.numpy(bert_tokens.cpu())

            roberta_features = []
            num_tokens = 0
            for i in range(bert_tokens.shape[0]):
                char_list = [c for c in bert_tokens[i] if c!=0]
                sentence = ''.join(chr(c) for c in char_list)
                known_tokens = sentence.split(' ')
                aligned_features = self.bert_encoder.align(sentence, known_tokens, border_tokens=True)
                roberta_features.append(aligned_features)
                if num_tokens < list(aligned_features.size())[0]:
                    num_tokens = list(aligned_features.size())[0]

            padded_roberta_features = []
            for f in roberta_features:
                padded = torch.zeros(num_tokens, 768)
                padded[:list(f.size())[0],:] = f
                padded_roberta_features.append(padded)

            roberta_embeddings = torch.stack(padded_roberta_features)
            roberta_embeddings = roberta_embeddings.cuda()
            roberta_embeddings = roberta_embeddings[:, 1:-1]
            # Add zero tokens corresponding to @end@, @none@, and @null@.
            zero_cols = torch.zeros(
                    roberta_embeddings.shape[0],
                    3,
                    roberta_embeddings.shape[2]).to(roberta_embeddings.device)
            roberta_embeddings = torch.cat([roberta_embeddings, zero_cols], 1)
            encoder_inputs += [roberta_embeddings]

        token_embeddings = self.encoder_token_embedding(tokens)
        lemma_embeddings = self.encoder_lemma_embedding(lemmas)
        if self.use_pos:
            pos_tag_embeddings = self.encoder_pos_embedding(pos_tags)
            encoder_inputs += [token_embeddings, lemma_embeddings, pos_tag_embeddings]
        else:
            encoder_inputs += [token_embeddings, lemma_embeddings]

        if self.use_char_cnn:
            char_cnn_output = self._get_encoder_char_cnn_output(chars)
            encoder_inputs += [char_cnn_output]

        encoder_inputs = torch.cat(encoder_inputs, 2)
        encoder_inputs = self.encoder_embedding_dropout(encoder_inputs)
        # [batch, num_tokens, encoder_output_size]
        encoder_outputs = self.encoder(encoder_inputs, mask)
        encoder_outputs = self.encoder_output_dropout(encoder_outputs)

        # A tuple of (state, memory) with shape [num_layers, batch, encoder_output_size]
        encoder_final_states = self.encoder._states
        self.encoder.reset_states()

        return dict(
            memory_bank=encoder_outputs,
            final_states=encoder_final_states
        )

    def get_symbol_embeddings(
            self,
            state: Optional[Tensor],
            tokens: Tensor,
            pos_tags: Optional[Tensor],
            chars: Tensor,
            tgt_mask: Tensor,
            ):
        """Runs the incremental LSTM symbol decoder for the given sequences.

        symbol encoding. This is actually a symbol encoder, not a decoder,
        since the parser actions determine the new symbols and this just
        encodes them into embeddings.

        Parameters
        ----------
        state: Optional[Tensor]
            [batch_size, seq_len, hidden_dim] Previous state of the decoder,
            None for start.
        tokens:
            [batch_size, seq_len] symbol sequence token indices.
        pos_tags:
            [batch_size, seq_len] POS tag indices.
        chars:
            [batch_size, seq_len, max_chars] Character sequence indices.
        tgt_mask:
            [batch_size, seq_len] symbol masking tensor
        """
        input_is_empty = len(tokens.shape) == 1 and tokens.shape[0] == 0
        # [batch, num_tokens, embedding_size]
        token_embeddings = self.decoder_token_embedding(tokens)
        #pos_tag_embeddings = self.decoder_pos_embedding(pos_tags)
        if self.use_char_cnn and not input_is_empty:
            char_cnn_output = self._get_decoder_char_cnn_output(chars)
            decoder_inputs = torch.cat([
                token_embeddings, char_cnn_output], 2)
                #token_embeddings, pos_tag_embeddings, char_cnn_output], 2)
        else:
            decoder_inputs = torch.cat([
                token_embeddings], 2)
                #token_embeddings, pos_tag_embeddings], 2)
        decoder_inputs = self.decoder_embedding_dropout(decoder_inputs)
        decoder_outputs, decoder_final_states = \
                self.decoder(decoder_inputs, tgt_mask, state)
        final_outputs = self.decoder_output_dropout(decoder_outputs)
        self.decoder.reset_states()

        return dict(
            memory_bank=final_outputs,
            final_states=decoder_final_states,
            decoder_outputs=decoder_outputs,
        )

    def _get_encoder_char_cnn_output(self, chars):
        # [batch, num_tokens, num_chars, embedding_size]
        char_embeddings = self.encoder_char_embedding(chars)
        batch_size, num_tokens, num_chars, _ = char_embeddings.size()
        char_embeddings = char_embeddings.view(batch_size * num_tokens, num_chars, -1)
        char_cnn_output = self.encoder_char_cnn(char_embeddings, None)
        char_cnn_output = char_cnn_output.view(batch_size, num_tokens, -1)
        return char_cnn_output

    def _get_decoder_char_cnn_output(self, chars):
        # [batch, num_tokens, num_chars, embedding_size]
        char_embeddings = self.decoder_char_embedding(chars)
        batch_size, num_tokens, num_chars, _ = char_embeddings.size()
        char_embeddings = char_embeddings.view(batch_size * num_tokens, num_chars, -1)
        char_cnn_output = self.decoder_char_cnn(char_embeddings, None)
        char_cnn_output = char_cnn_output.view(batch_size, num_tokens, -1)
        return char_cnn_output

    def character_tensor_from_token_tensor(self,
            token_tensor,
            vocab,
            character_tokenizer,
            namespace=dict(tokens="decoder_token_ids", characters="decoder_token_characters"),
            ):
        token_str = [vocab.get_token_from_index(i, namespace["tokens"]) for i in token_tensor.view(-1).tolist()]
        max_char_len = max([len(token) for token in token_str])
        max_filter_size = max(self.decoder_char_cnn.conv_layer_0.kernel_size)
        max_char_len = max(max_char_len, max_filter_size)
        indices = []
        for token in token_str:
            token_indices = [vocab.get_token_index(vocab._padding_token) for _ in range(max_char_len)]
            for char_i, character in enumerate(character_tokenizer.tokenize(token)):
                index = vocab.get_token_index(character.text, namespace["characters"])
                token_indices[char_i] = index
            indices.append(token_indices)
        return torch.tensor(indices).view(token_tensor.size(0), token_tensor.size(1), -1).type_as(token_tensor)

    def decode(self, input_dict):
        """Actual ULF decoding. (not the "decoding" which refers to symbol
        generation).

        input_dict: Dictionary of the output of self.forward() when not training.
        """
        use_symbol_inputs = 'decoder_outputs' in input_dict and input_dict['decoder_outputs'] is not None
        # List of (hypothesis, output amr) pairs.
        # The output amr is the class defined under postprocessing/amr_graph.py
        beam_search_outputs = self.beam_search_with_cache_transition_system(
                input_dict['encoder_memory_bank'],
                input_dict['decoder_outputs']['memory_bank'] if use_symbol_inputs else None,
                input_dict['encoder_inputs']['mask'],
                input_dict['decoder_inputs']['mask'],
                input_dict['decoder_inputs'],
                input_dict,
                use_symbol_inputs,
        )
        return beam_search_outputs


    def beam_search_with_cache_transition_system(
            self, w_memory_bank, s_memory_bank, w_mask, s_mask, decoder_inputs, input_dict,
            use_symbol_inputs):
        """Runs beam search over the cache transition system.

        PyTorch version of NP2P_beam_decoder.run_beam_search. We reuse the
        non-tensorflow code.

        The beam search takes a trained model (`model`), vocabularies (`feat_vocab`,
        `action_vocab`), a few hyperparameters (`batch`, `cache_size`), general script
        flag options (`options`), and dictionary for storing categorized transition
        evaluation results.

        For actual beam search, `category_res` must not be set (i.e. be None).
        If `category_res` is not None, the search will always take the gold action
        and update `category_res` with counts of how often each action is
        top-scoring action by the model.
        """

        """
        Algorithm (refer to NP2P_beam_decoder...):

        Set up the cache transition system cts
        Set up the initial neural network inputs with first words and symbols
        last_action = none
        beam = [initial]
        while < max_actions and not beam.empty():
            hard_attn_decoder_state, cur_prob = beam.pop()
            next_actions, next_states, probs = NN.predict(hard_attn_decoder_state)
            for na, ns, prob in zip(next_actions, next_states, probs):
                if not cts.canApply(na, hard_attn_decoder_state):
                    continue
                beam.push((ns, prob + cur_prob))
            beam.sort()
            if len(beam) > beam_size:
                beam = beam[:beam_size]

        """
        # From parameters
        # w_memory_bank, s_memory_bank (encodings of word and symbol sequences)
        # w_mask, s_mask (masks for words and symbols)
        assert use_symbol_inputs == (s_memory_bank is not None), \
                "use_symbol_inputs: {}\ns_memory_bank: {}\n".format(use_symbol_inputs, s_memory_bank)

        # Set up Cache Transition
        ts_params = self.transition_system_params
        inseq_syms = None
        if 'inseq_symbol_file' in ts_params and ts_params['inseq_symbol_file'] is not None:
            inseq_syms = set(open(ts_params['inseq_symbol_file'], 'r').read().strip().split(' '))
        promote_syms = None
        if 'promote_symbol_file' in ts_params and ts_params['promote_symbol_file'] is not None:
            promote_syms = set(open(ts_params['promote_symbol_file'], 'r').read().strip().split(' '))
        cts = CacheTransition(
            ts_params['cache_size'],
            inseq_symbols=inseq_syms,
            promote_symbols=promote_syms,
            reserve_gen_method=ts_params['reserve_gen_method'],
            unreserve_gen_method=ts_params['unreserve_gen_method'],
            type_checking_method=ts_params['type_checking_method'],
            focus_method=ts_params['focus_method'],
            ulf_lexicon=self.ulf_lexicon,
            buffer_offset=ts_params['buffer_offset'],
        )

        # Set possible actions and arc choices from the counters.
        (
            cts.shiftpop_action_set,
            cts.push_action_set,
            cts.arcbinary_action_set,
            cts.arclabel_action_set,
            cts.gensym_action_set,
            cts.seqgen_action_set,
            cts.promote_sym_action_set,
            cts.promote_arc_action_set,
        ) = NP2P_data_stream.load_actions(
            ts_params['action_counter_dir'],
            reserve_gen_method=cts.reserve_gen_method,
            unreserve_gen_method=cts.unreserve_gen_method,
        )
        (
            cts.income_arcChoices,
            cts.outgo_arcChoices,
            cts.default_arcChoices,
        ) = NP2P_data_stream.load_arc_choices(ts_params['arc_choices_dir'])

        batch_size = w_memory_bank.shape[0]
        beam_search_results = []
        for batch_idx in range(batch_size):
            print("")
            print("batch_idx: {}".format(batch_idx))
            print("input: {}".format(input_dict['raw_batch']['meta_src_tokens'][batch_idx]))
            tgt_tokens = input_dict['raw_batch']['meta_tgt_tokens'][batch_idx]
            sid2wid = input_dict['raw_batch']['sid2wid'][batch_idx]
            # Remove tgt_tokens and alignment if symbols are generated from words.
            if cts.unreserve_gen_method == UnreserveGenMethod.WORD:
                tgt_tokens = [tgt_tokens[0], tgt_tokens[len(tgt_tokens) - 1]]
                sid2wid = {}
            # Reorganize single instance input for decoding.
            instance_input = dict(
                w_memory_bank = input_dict['encoder_memory_bank'][batch_idx],
                # encoder_final_states: tuple each with dimension [num_layers, beam_size, hidden_dim]
                w_final_states = (
                    input_dict['encoder_final_states'][0].transpose(0, 1)[batch_idx],
                    input_dict['encoder_final_states'][1].transpose(0, 1)[batch_idx],
                ),
                s_memory_bank = (
                    input_dict['decoder_outputs']['memory_bank'][batch_idx]
                    if use_symbol_inputs
                    else None
                ),
                w_mask = input_dict['encoder_inputs']['mask'][batch_idx],
                s_mask = (
                    input_dict['decoder_inputs']['mask'][batch_idx]
                    if use_symbol_inputs
                    else None
                ),
                src_tokens = input_dict['raw_batch']['meta_src_tokens'][batch_idx],
                src_lemmas = input_dict['raw_batch']['meta_src_lemmas'][batch_idx],
                src_pos_tags = input_dict['raw_batch']['meta_src_pos_tags'][batch_idx],
                src_ner_tags = input_dict['raw_batch']['meta_src_ner_tags'][batch_idx],
                tree = input_dict['raw_batch']['raw_dep'][batch_idx],
                tgt_tokens = tgt_tokens,
                sid2wid = sid2wid,
                use_symbol_inputs = use_symbol_inputs,
                sent_anno = input_dict['raw_batch']['np2p_sentence'][batch_idx],
            )
            hyps = self.single_instance_beam_search_cache_transition(
                    cts, instance_input,
                    action_repeat_limits=ts_params['action_repeat_limits'],
                    use_symbol_inputs=use_symbol_inputs,
            )
            hyp = hyps[0]
            output_ulfamr = generateAMR(
                hyp, cts, input_dict['raw_batch']['np2p_sentence'][batch_idx])
            ulfamr_str = output_ulfamr.to_amr_string()
            # Call common lisp to get the plain ULF form of the parsed AMR
            try:
                ulf_str = amr2ulf.amr2ulf(ulfamr_str)
            except:
                ulf_str = "(AMR2ULF_Exception)"
            beam_search_results.append(dict(
                hypothesis=hyp,
                output_ulfamr=output_ulfamr,
                output_ulfamr_string=ulfamr_str,
                output_ulf_string=ulf_str,
            ))
            self.print_ulfamr_decoding_results(hyp, cts, ulfamr_str, ulf_str)

        # Regroup results
        grouped_results = defaultdict(list)
        for inst_result in beam_search_results:
            for k, v in inst_result.items():
                grouped_results[k].append(v)
        return grouped_results

    def print_ulfamr_decoding_results(self, hyp, cts, ulfamr_str, ulf_str):
        curr_repr = "action seq: {}".format([
            hyp.actionSeqStr(
                lambda x: self.vocab.get_token_from_index(x, 'action_token_ids')
            ).replace("#",", ")
        ])
        split_action_strs = [curr_repr[i:i+120] for i in range(0, len(curr_repr), 120)]
        for sas in split_action_strs:
            print(sas)
        print(hyp.trans_state.toString(False, cts.unreserve_gen_method))
        print("AMR-ized ULF (internal parser representation)")
        print(ulfamr_str)
        print("Plain ULF")
        print(ulf_str)

        print()
        action_strs = hyp.actionStrs(
                lambda x: self.vocab.get_token_from_index(x, 'action_token_ids'))
        print("Action, focuses: {}".format(list(zip(action_strs[1:], hyp.foci))))
        print()


    def single_instance_beam_search_cache_transition(
            self, system, input_dict, category_res=None, action_repeat_limits={},
            use_symbol_inputs=True):
        """Runs beam search over a cache transition system for a single instance.

        For actual beam search, `category_res` must not be set (i.e. be None).
        If `category_res` is not None, the search will always take the gold action
        and update `category_res` with counts of how often each action is
        top-scoring action by the model.
        """

        def get_hyp_symbol_embs(hyp, cur_device):
            """Gets the symbol sequence embeddings from a hypothesis.

            This performs additional inference for any new tokens, and appends
            it to the existing embeddings which is returned.

            Returns
            -------
            s_embs: Tensor
                (hyp symbol len, symbol_hidden_dim), Encoded symbol embeddings.
            final_states: Tensor
                (symbol_hidden_dim), Final hidden states of the decoder for
                picking up on the next decoding step.
            """
            # Current memory bank state
            # [num symbols, hidden_dim] ???? check with assert below
            s_memory_bank = hyp.s_memory_bank
            s_final_state = hyp.state_embeddings.symbol_decoder
            symbol_seq = [START_SYMBOL] + hyp.trans_state.hypothesis.getSymbolSeq()
            if s_memory_bank is None:
                # Get it all started.
                new_symbol = START_SYMBOL
            elif s_memory_bank.shape[0] == len(symbol_seq) - 1:
                # We're missing an embedding.
                new_symbol = symbol_seq[-1]
            elif s_memory_bank.shape[0] == len(symbol_seq):
                # We have all the embeddings.
                new_symbol = None
            else:
                errmsg = (
                    "There is more than a one symbol discrepancy between the "
                    "symbol embeddings and the symbol sequence during "
                    "decoding."
                )
                raise ValueError(errmsg)

            if new_symbol is not None:
                # [1, 1] = [batch_size, num_tokens]
                decoder_tokens = torch.tensor(
                        [self.vocab.get_token_index(new_symbol, 'decoder_token_ids')],
                        dtype=torch.long,
                ).to(cur_device).reshape((1, 1))
                # [1, 1, max_chars] = [batch_size, num_tokens, max_chars]
                if self.use_char_cnn:
                    decoder_chars = self.character_tensor_from_token_tensor(
                            decoder_tokens, self.vocab, self.character_tokenizer).contiguous()
                else:
                    decoder_chars = None
                # Mask of all ones
                one_mask = torch.ones(1).to(cur_device).reshape((1, 1))
                s_decoder_outputs = self.get_symbol_embeddings(
                        s_final_state,
                        decoder_tokens.contiguous(),
                        None, # POS not used now
                        decoder_chars,
                        one_mask,
                )
                if s_memory_bank is None:
                    s_memory_bank = s_decoder_outputs['memory_bank'][0]
                else:
                    s_memory_bank = torch.cat([
                        s_memory_bank.clone(),
                        s_decoder_outputs['memory_bank'][0]
                    ])
                s_final_state = s_decoder_outputs['final_states']
            return s_memory_bank, s_final_state

        def extract_hyp_info(hyp, beam_input, gold_search):
            """
            Extracts information from a hypothesis for performing a decoding step
            and adds it to beam_input.

            If `gold_search` is True, feature extraction necessary for actual
            prediction is skipped.

            Assumes that beam_input has list or tensor entries for
                'ts_feats', 'word_idx', 'symbol_idx'
            """
            feat_idxs = None
            wid = None
            sid = None
            s_final_state = None
            # Reference tensor so that new tensors are on the same device.
            cur_device = input_dict['w_memory_bank'].device

            if (not gold_search and
                    hyp.trans_state.phase == FeatureType.SHIFTPOP and
                    system.unreserve_gen_method == UnreserveGenMethod.NONE):
                hyp.readOffUnalignWords()  # Ignore unaligned words.

            if not gold_search:
                if self.use_ts_feats:
                    feats = hyp.extractFeatures()
                    feat_idxs = [self.vocab.get_token_index(f, 'action_feature_tags') for f in feats]
                    beam_input['ts_feats'].append(torch.tensor(feat_idxs).to(cur_device))
                else:
                    beam_input['ts_feats'].append(None)
                word_focus, symbol_focus = system.focus_manager.get(hyp.trans_state)
                beam_input['word_focus'].append(word_focus)
                beam_input['symbol_focus'].append(symbol_focus)

            if not gold_search and not use_symbol_inputs:
                # Generate symbol inputs for the current hypothesis symbol sequence.
                s_embs, s_final_state = get_hyp_symbol_embs(hyp, cur_device)
                beam_input['symbol_hiddens'].append(s_embs)

            beam_input['symbol_decoder_state'].append(s_final_state)
            beam_input['hard_attn_decoder_state'].append(hyp.state_embeddings.hard_attn_decoder)
            beam_input['prev_action'].append(torch.tensor(hyp.latest_action()).to(cur_device))

        def get_beam_hard_attn_decoder_inputs(hyps, gold_search):
            """Takes a list of hypotheses, extracts their hypothesis info and
            reorganizes them for batched decoding.
            """
            cur_size = len(hyps) # current number of hypothesis in the beam
            # Get features for beam.
            beam_input = defaultdict(list)
            for h in hyps:
                extract_hyp_info(h, beam_input, gold_search)

            # Organize states into a single tensor for hard attention decoder.
            decoder_states = {}
            for state_key in ['hard_attn_decoder_state', 'symbol_decoder_state']:
                statelst = beam_input[state_key]
                if len(statelst) == 1 and statelst[0] is None:
                    statetens = None
                else:
                    # Build the batches for the hidden and final lstm states.
                    statetens = (
                        torch.stack([hs for hs, _ in statelst]).transpose(0, 1),
                        torch.stack([fs for _, fs in statelst]).transpose(0, 1)
                    )
                decoder_states[state_key] = statetens

            input_word_memory = torch.stack([
                input_hiddens[focus]
                for focus in beam_input['word_focus']
            ]).unsqueeze(1)
            if self.use_ts_feats:
                input_features = torch.stack([
                    self.feat_embedding(feat_indices.type(torch.cuda.LongTensor))
                    for feat_indices
                    in beam_input['ts_feats']
                ]).view(cur_size, 1, -1)
            else:
                input_features = None
            input_prev_actions = torch.stack([
                self.action_embedding(prev_action.type(torch.cuda.LongTensor))
                for prev_action
                in beam_input['prev_action']
            ]).unsqueeze(1)
            input_mask = mask[:cur_size]
            input_symbol_memory = None
            # Get focus-dependent symbol embeddings.
            if use_symbol_inputs:
                input_symbol_memory = torch.stack([
                    symbol_hiddens[focus]
                    for focus in beam_input['symbol_focus']
                ]).unsqueeze(1)
            else:
                # If symbols are generated by the model, the hidden states are dependent on the
                # hypothesis.
                input_symbol_memory = torch.stack([
                    hyp_symbol_hiddens[focus]
                    for focus, hyp_symbol_hiddens
                    in zip(beam_input['symbol_focus'], beam_input['symbol_hiddens'])
                ]).unsqueeze(1)

            hard_attn_kwargs = dict(
                state=decoder_states['hard_attn_decoder_state'],
                w=input_word_memory,
                s=input_symbol_memory,
                f=input_features,
                a=input_prev_actions,
                mask=input_mask,
            )
            return (hard_attn_kwargs,
                    decoder_states['symbol_decoder_state'],
                    beam_input['symbol_hiddens'])

        def generate_next_hypotheses(hyps, new_states, input_word_memory,
                input_symbol_memory, symbol_seq_memory, topk_actions, topk_log_probs):
            """
            Extend current beam of hypotheses into a new list of hypotheses
            using the topk predictions and the cache transition system.

            Parameters
            ----------
            hyps: list of decoding hypotheses
            new_states: DecoderStateEmbs
                 a tuple of states for each sequence decoder. Each one is a
                 tuple of tensor of new states after taking a single step from
                 hyps
                 [(num_layers, beam_size, hidden_dim),
                  (num_layers, beam_size, hidden_dim)]
            input_word_memory: tensor of word embeddings associated with the
                word focus of the current beam.
            input_symbol_memory: tensor of symbol embeddings associate with the
                symbol focus of the current beam.
            symbol_seq_memory: List[Tensor]
                Symbol memory bank to be propagated in the hypothesis.
            topk_actions: tensor of topk action ids for the current beam
                (beam_size, topk)
            topk_log_probs: tensor of log probabilities associated with the
                topk_actions tensor.
                (beam_size, topk)
            """
            if gold_search:
                # Assume only one instance.
                inst = input_dict['raw_batch'].instances[0]
                feat_id = inst.features_idx[steps][0]
                action_id = inst.actions_idx[steps]

            # Transpose new_states so that each element of new_states is beam_size first.
            transposed = []
            for nstate in new_states:
                transposed.append((
                    nstate[0].transpose(0, 1),
                    nstate[1].transpose(0, 1),
                ))
            tnew_states = DecoderStateEmbs(*transposed)

            all_hyps = []
            # i : beam example index
            # j : topk next step index
            for i in range(len(hyps)):
                h = hyps[i]
                tnew_state = DecoderStateEmbs(
                    *[(h[i], c[i]) for h, c in tnew_states]
                )
                cur_input_word = input_word_memory[i]
                cur_input_symbol = input_symbol_memory[i]
                jiters = (
                    self.topk_size
                    if gold_search
                    else min(self.topk_size, topk_actions.shape[1])
                )
                j = 0
                while j < jiters:
                    cur_action_id = topk_actions[i, j].item()
                    cur_action_log_prob = topk_log_probs[i, j].item()
                    new_hyp = h.extend(
                        system, # Transition system
                        topk_actions[i, j].item(),
                        topk_log_probs[i, j].item(),
                        tnew_state,
                        lambda x: self.vocab.get_token_from_index(x, 'action_token_ids'),
                        symbol_seq_memory[i],
                        print_info=False,
                        sent_anno=input_dict['sent_anno'],
                    )
                    j += 1
                    if (new_hyp and new_hyp.check_repeat_limits(action_repeat_limits)):
                        if gold_search:
                            category_res[feat_id][0] += 1.0
                            # Update `category_res` with action info.
                            category_res[feat_id][0] += 1.0
                            if cur_action_id == action_id:
                                category_res[feat_id][1] += 1.0
                            # Replace `new_hyp` with result of gold action.
                            new_hyp = h.extend(
                                system,
                                action_id,
                                1.0,
                                hard_attn_decoder_state,
                                lambda x: self.vocab.get_token_from_index(x, 'action_token_ids'),
                                symbol_seq_memory[i],
                                gold=True,
                            )
                            assert new_hyp is not None
                            # Exit search early, we found what we need.
                            j = jiters
                        all_hyps.append(new_hyp)
            return all_hyps

        gold_search = category_res is not None

        # Initial NN state.
        # The HardAttnDecoder takes care of initialization when given None.
        sent_stop_id = self.vocab.get_token_index('@end@', 'action_token_ids')

        cur_device = input_dict['w_memory_bank'].device
        input_hiddens = input_dict['w_memory_bank']
        input_mask = input_dict['w_mask']
        symbol_hiddens = input_dict['s_memory_bank']

        sent_length = len(input_dict['src_tokens'][:-3]) # remove end and none tokenc
        initial_config = CacheConfiguration(system.cache_size, sent_length)

        # All these attributes are shared by every hypothesis
        initial_config.wordSeq = input_dict['src_tokens'][:-3] # remove end and none tokens
        initial_config.lemSeq = input_dict['src_lemmas'][:-3]
        initial_config.posSeq = input_dict['src_pos_tags'][:-3]
        initial_config.nerSeq = input_dict['src_ner_tags'][:-3]
        initial_config.symbolSeq = input_dict['tgt_tokens'][1:-1] # remove start and end tokens
        initial_config.symbolAlign = input_dict['sid2wid']
        initial_config.tree = input_dict['tree']
        initial_config.buildWordToSymbol()
        initial_config.buildSymbolToTypes()
        start_action_id = self.vocab.get_token_index(START_SYMBOL, namespace='action_token_ids')
        initial_actionseq = [start_action_id]

        initial_hypo = Hypothesis(
            initial_actionseq,
            [0.0],
            DecoderStateEmbs(None, None),  # initial state
            initial_config,
            sanity_check_arcs=self.sanity_check_arcs,
        )
        hyps = [initial_hypo]
        #======Beam Search Decoding=======#
        # `results`: stores finished hypotheses (those that have emitted the </s> action)
        finished_results = []
        steps = 0
        beam_size = self.beam_size
        max_actions = self.max_action_length
        if gold_search:
            # Set gold action length as max.
            max_actions = len(batch.instances[0].actions_idx)
        feat_print_count = 0
        # Initialize mask for a single action with beam_size examples.
        mask = input_hiddens.new_ones((beam_size, 1))
        while steps < max_actions and len(finished_results) < beam_size:
            hard_attn_kwargs, symbol_decoder_states, symbol_seq_memory = \
                    get_beam_hard_attn_decoder_inputs(hyps, gold_search)
            _, final_states, _, log_probs = self.hard_attn_decoder(**hard_attn_kwargs)
            # [num_hyps, 1, action_vocab_size] -> [num_hyps, action_vocab_size]
            log_probs = log_probs.squeeze(1)

            # Get top-k actions and associated log-probs.
            # topk_log_probs, topk_actions (beam_size, topk_size)
            topk_log_probs, topk_actions = log_probs.topk(min(self.topk_size, log_probs.shape[1]))
            act_tokens = []
            for aid in topk_actions[0].tolist():
                act_tokens.append(self.vocab.get_token_from_index(aid, 'action_token_ids'))

            # Extend each hypothesis with the topk actions.
            decoder_state_embs = DecoderStateEmbs(final_states, symbol_decoder_states)
            input_word_memory = hard_attn_kwargs['w']
            input_symbol_memory = hard_attn_kwargs['s']
            all_hyps = generate_next_hypotheses(hyps, decoder_state_embs,
                    input_word_memory, input_symbol_memory, symbol_seq_memory,
                    topk_actions, topk_log_probs)
            if len(all_hyps) == 0:
                print("No hypothesis found at step {}, with {} finished results".format(
                    steps, len(finished_results)))
                break
            # Filter and collect any hypotheses that have produced the end action.
            # hyps will contain hypotheses for the next step
            hyps = []
            sorthyps = sort_hypotheses(all_hyps)
            for h in sort_hypotheses(all_hyps):
                # If this hypothesis is sufficiently long, put in finished_results. Otherwise discard.
                if (
                    h.latest_action() == sent_stop_id or
                    system.isTerminal(h.trans_state, with_gold=False)
                ):
                    if h.latest_action() == sent_stop_id:
                        print("hit the sent_stop_id?")
                        import ipdb; ipdb.set_trace() # BREAKPOINT
                    finished_results.append(h)
                    if gold_search:
                        print('!!!break here, we found \'em!')
                # hasn't reached stop action, so continue to extend this hypothesis
                else:
                    hyps.append(h)
                if len(hyps) == beam_size or len(finished_results) == beam_size:
                    break
            # Stop beam search, if there are no more hypotheses.
            if len(hyps) == 0:
                break
            steps += 1

        # At this point, either we've got beam_size results, or we've reached
        # maximum decoder steps if we don't have any complete results, add all
        # current hypotheses (incomplete summaries) to results
        if len(finished_results) == 0:
            finished_results = hyps
        # Return the hypothesis with highest average log prob
        hyps_sorted = sort_hypotheses(finished_results)
        return hyps_sorted


    @classmethod
    def from_params(cls, vocab, params):
        """Build the ULFCTP model from JSON parameter specifications.
        """
        logger.info('Building the ULFCTP Model...')

        # Sentence Encoder
        encoder_input_size = 0
        bert_encoder = None
        if params.get('use_bert', False):
            bert_encoder = Seq2SeqBertEncoder.from_pretrained(params['bert']['pretrained_model_dir'])
            encoder_input_size += params['bert']['hidden_size']
            for p in bert_encoder.parameters():
                p.requires_grad = False

        if params.get('use_roberta', False):
            # bert_encoder = Seq2SeqRobertaEncoder.from_pretrained(params['roberta']['pretrained_model_dir'])
            bert_encoder = RoBERTaBase()
            encoder_input_size += params['roberta']['hidden_size']
            # for p in bert_encoder.parameters():
            #     p.requires_grad = False

        encoder_token_embedding = Embedding.from_params(vocab, params['encoder_token_embedding'])
        encoder_input_size += params['encoder_token_embedding']['embedding_dim']
        if params['use_pos']:
            encoder_pos_embedding = Embedding.from_params(vocab, params['encoder_pos_embedding'])
            encoder_input_size += params['encoder_pos_embedding']['embedding_dim']
        else:
            encoder_pos_embedding = None
        encoder_lemma_embedding = Embedding.from_params(vocab, params['encoder_lemma_embedding'])
        encoder_input_size += params['encoder_lemma_embedding']['embedding_dim']

        if params['use_char_cnn']:
            encoder_char_embedding = Embedding.from_params(vocab, params['encoder_char_embedding'])
            encoder_char_cnn = CnnEncoder(
                embedding_dim=params['encoder_char_cnn']['embedding_dim'],
                num_filters=params['encoder_char_cnn']['num_filters'],
                ngram_filter_sizes=params['encoder_char_cnn']['ngram_filter_sizes'],
                conv_layer_activation=torch.tanh
            )
            encoder_input_size += params['encoder_char_cnn']['num_filters']
        else:
            encoder_char_embedding = None
            encoder_char_cnn = None

        encoder_embedding_dropout = InputVariationalDropout(p=params['encoder_token_embedding']['dropout'])

        params['encoder']['input_size'] = encoder_input_size
        encoder = PytorchSeq2SeqWrapper(
            module=StackedBidirectionalLstm.from_params(params['encoder']),
            stateful=True
        )
        encoder_output_dropout = InputVariationalDropout(p=params['encoder']['dropout'])

        # Symbol encoder (Decoder)
        decoder_input_size = 0
        decoder_input_size += params['decoder_token_embedding']['embedding_dim']
        decoder_token_embedding = Embedding.from_params(vocab, params['decoder_token_embedding'])
        if params['use_pos']:
            decoder_pos_embedding = Embedding.from_params(vocab, params['decoder_pos_embedding'])
        else:
            decoder_pos_embedding = None
        if params['use_char_cnn']:
            decoder_char_embedding = Embedding.from_params(vocab, params['decoder_char_embedding'])
            decoder_char_cnn = CnnEncoder(
                embedding_dim=params['decoder_char_cnn']['embedding_dim'],
                num_filters=params['decoder_char_cnn']['num_filters'],
                ngram_filter_sizes=params['decoder_char_cnn']['ngram_filter_sizes'],
                conv_layer_activation=torch.tanh
            )
            decoder_input_size += params['decoder_char_cnn']['num_filters']
        else:
            decoder_char_embedding = None
            decoder_char_cnn = None

        decoder_embedding_dropout = InputVariationalDropout(p=params['decoder_token_embedding']['dropout'])
        params['decoder']['input_size'] = decoder_input_size
        decoder = PytorchIncrSeq2SeqWrapper(
            module=StackedLstm.from_params(params['decoder']),
        )
        decoder_output_dropout = InputVariationalDropout(p=params['decoder']['dropout'])

        # Cache transition system embeddings
        action_embedding = Embedding.from_params(vocab, params['action_embedding'])

        # Use hard attention decoder.
        # encoder is bidirectional, so * 2
        encoder_hidden = params['encoder']['hidden_size'] * 2
        hard_attn_input_size = params['action_embedding']['embedding_dim'] + \
                encoder_hidden + \
                params['decoder']['hidden_size']
        if params['use_ts_feats']:
            feat_embedding = Embedding.from_params(vocab, params['feat_embedding'])
            num_action_feats = oracle.utils.action_feat_num(params['transition_system']['cache_size'])
            hard_attn_input_size += params['feat_embedding']['embedding_dim'] * num_action_feats
        else:
            feat_embedding = None
        params['hard_attn_decoder']['hard_attn_encoder']['input_size'] = hard_attn_input_size
        params['hard_attn_decoder']['action_vocab_size'] = action_embedding.num_embeddings
        params['hard_attn_decoder']['use_ts_feats'] = params['use_ts_feats']
        hard_attn_decoder = HardAttnDecoder.from_params(vocab, params['hard_attn_decoder'])

        # Vocab
        punctuation_ids = []
        oov_id = vocab.get_token_index(DEFAULT_OOV_TOKEN, 'decoder_token_ids')
        for s in ',.?!:;"\'-(){}[]':
            s_id = vocab.get_token_index(s, 'decoder_token_ids')
            if s_id != oov_id:
                punctuation_ids.append(s_id)

        logger.info('encoder_token: %d' % vocab.get_vocab_size('encoder_token_ids'))
        logger.info('encoder_chars: %d' % vocab.get_vocab_size('encoder_token_characters'))
        logger.info('decoder_token: %d' % vocab.get_vocab_size('decoder_token_ids'))
        logger.info('decoder_chars: %d' % vocab.get_vocab_size('decoder_token_characters'))

        return cls(
            vocab=vocab,
            punctuation_ids=punctuation_ids,
            use_pos=params['use_pos'],
            use_char_cnn=params['use_char_cnn'],
            use_ts_feats=params['use_ts_feats'],
            use_coverage=params['use_coverage'],
            use_bert=params.get('use_bert', False),
            use_roberta=params.get('use_roberta', False),
            #max_decode_length=params.get('max_decode_length', 50),
            max_action_length=params.get('max_action_length', 300),
            topk_size=params.get('topk_size', 100),
            use_symbol_inputs=params.get('use_symbol_inputs', False),
            beam_size=params.get('beam_size', 1),
            transition_system_params=params['transition_system'],
            bert_encoder=bert_encoder,
            encoder_token_embedding=encoder_token_embedding,
            encoder_pos_embedding=encoder_pos_embedding,
            encoder_lemma_embedding=encoder_lemma_embedding,
            encoder_char_embedding=encoder_char_embedding,
            encoder_char_cnn=encoder_char_cnn,
            encoder_embedding_dropout=encoder_embedding_dropout,
            encoder=encoder,
            encoder_output_dropout=encoder_output_dropout,
            decoder_token_embedding=decoder_token_embedding,
            decoder_pos_embedding=decoder_pos_embedding,
            decoder_char_cnn=decoder_char_cnn,
            decoder_char_embedding=decoder_char_embedding,
            decoder_embedding_dropout=decoder_embedding_dropout,
            decoder=decoder,
            decoder_output_dropout=decoder_output_dropout,
            hard_attn_decoder=hard_attn_decoder,
            test_config=params.get('mimick_test', None),
            action_embedding=action_embedding,
            feat_embedding=feat_embedding,
        )

