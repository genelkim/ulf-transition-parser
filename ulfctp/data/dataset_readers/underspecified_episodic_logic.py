from typing import Dict, List, Tuple
import logging
import os
import json
import numpy

from overrides import overrides
from stog.utils.file import cached_path
from stog.data.dataset_readers.dataset_reader import DatasetReader
from stog.data.fields import (
    TextField, SpanField, SequenceLabelField, ListField,
    MetadataField, Field, AdjacencyField, ArrayField,
)
from stog.data.instance import Instance
from stog.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from stog.data.tokenizers import Token
from stog.data.tokenizers.bert_tokenizer import AMRBertTokenizer
from stog.data.vocabulary import DEFAULT_PADDING_TOKEN, DEFAULT_OOV_TOKEN
from stog.utils.checks import ConfigurationError
from ulfctp.data.fields import SequenceLabelListField
from ulfctp.utils.string import (
    START_SYMBOL, END_SYMBOL, NONE_SYMBOL, NULL_SYMBOL, find_similar_token,
)
import NP2P_data_stream
# from transformers import RobertaTokenizer

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


def prepare_raw_input_dict(action_data, bos=None, eos=None, nis=None, null=None,
        bert_tokenizer=None, use_roberta = False, max_tgt_length=None):
    """Prepares a raw input dictionary from the raw action_data input by adding
    special tokens (start of sequence, end of sequence, not in sequence)
    appropriately, extracting BERT token info, and enforcing sequence length
    limits.

    bos: beginning of sequence token
    eos: end of sequence token
    nis: not in sequence token
    null: null token
    """
    # Target Tokens
    tgt_tokens = action_data.sentence.symbols
    if max_tgt_length:
        tgt_tokens = tgt_tokens[:max_tgt_length]
    tgt_tokens = [bos] + action_data.sentence.symbols + [eos]

    def add_source_side_tags_to_target_side(_src_tokens, _src_tags):
        assert len(_src_tags) == len(_src_tokens)
        tgt_tags = []
        for tgt_token in tgt_tokens:
            sim_token = find_similar_token(tgt_token, _src_tokens)
            if sim_token is not None:
                index = _src_tokens.index(sim_token)
                tag = _src_tags[index]
            else:
                tag = DEFAULT_OOV_TOKEN
            tgt_tags.append(tag)
        return tgt_tags

    def extend_action_input_features(raw_feats, raw_action2wid, raw_action2sid):
        # Preprocess action input features to fit into this data framework.
        # The features must correspond one-to-one on action tokens, which
        # includes a start and end symbol, add an empty entry at the end.
        # These already assume prediction of the full sequence + an end
        # token.
        #
        # The values don't really matter since they should be ignored by
        # the model.
        single_featset = raw_feats[0]
        empty_featset = ['NONE' for _ in single_featset]
        feats = raw_feats + [empty_featset]
        action2wid = raw_action2wid + [-1]
        action2sid = raw_action2sid + [-1]
        return feats, action2wid, action2sid

    # Source Copy
    src_tokens = action_data.sentence.tok
    src_lemmas = action_data.sentence.lemma
    src_token_ids = None
    src_token_subword_index = None
    src_pos_tags = action_data.sentence.pos
    src_ner_tags = action_data.sentence.ner
    if bert_tokenizer is not None:
        src_token_ids, src_token_subword_index = bert_tokenizer.tokenize(src_tokens, True)
    if use_roberta:
        src_token_ids = numpy.asarray([ord(c) for c in ' '.join(src_tokens)])
        # print(len(src_token_ids))
        # tokens = roberta_tokenizer.tokenize(' '.join(src_tokens), add_prefix_space=True)
        # tokens = [roberta_tokenizer.cls_token] + tokens + [roberta_tokenizer.sep_token]
        # src_token_ids = numpy.asarray(roberta_tokenizer.convert_tokens_to_ids(tokens))
    if eos:
        src_tokens = src_tokens + [eos]
        src_lemmas = src_lemmas + [eos]
        src_pos_tags = src_pos_tags + [eos]
        src_ner_tags = src_ner_tags + [eos]
    if nis:
        src_tokens = src_tokens + [nis]
        src_lemmas = src_lemmas + [nis]
        src_pos_tags = src_pos_tags + [nis]
        src_ner_tags = src_ner_tags + [nis]
    if null:
        src_tokens = src_tokens + [null]
        src_lemmas = src_lemmas + [null]
        src_pos_tags = src_pos_tags + [null]
        src_ner_tags = src_ner_tags + [null]
    tgt_pos_tags = add_source_side_tags_to_target_side(src_tokens, src_pos_tags)
    action_tokens = [bos] + action_data.actions_idx + [eos]
    ts_feats, action2wid, action2sid = extend_action_input_features(
            action_data.features_idx,
            action_data.action2wid,
            action_data.action2sid,
    )
    return {
        "tgt_tokens" : tgt_tokens,
        "tgt_pos_tags": tgt_pos_tags,
        "src_tokens" : src_tokens,
        "src_token_ids" : src_token_ids,
        "src_lemmas": src_lemmas,
        "src_token_subword_index" : src_token_subword_index,
        "src_pos_tags": src_pos_tags,
        "src_ner_tags": src_ner_tags,
        "action_tokens": action_tokens,
        "ts_feats": ts_feats,
        "action2wid": action2wid,
        "action2sid": action2sid,
    }
 

@DatasetReader.register("ulf_trees")
class UnderspecifiedEpisodicLogicDatasetReader(DatasetReader):
    '''
    Dataset reader for ULF data
    '''
    def __init__(self,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 word_splitter = None,
                 use_roberta = False,
                 lazy: bool = False,
                 skip_first_line: bool = True,
                 evaluation: bool = False,
                 pred_symbols: bool = False,
                 ) -> None:

        super().__init__(lazy=lazy)
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}
        if word_splitter is not None:
            self._word_splitter = AMRBertTokenizer.from_pretrained(
                word_splitter, do_lower_case=False)
        else:
            self._word_splitter = None

        self.use_roberta = use_roberta
        self._skip_first_line = skip_first_line
        self._evaluation = evaluation
        self._pred_symbols = pred_symbols

        self._number_bert_ids = 0
        self._number_bert_oov_ids = 0

    def report_coverage(self):
        if self._number_bert_ids != 0:
            logger.info('BERT OOV  rate: {0:.4f} ({1}/{2})'.format(
                self._number_bert_oov_ids / self._number_bert_ids,
                self._number_bert_oov_ids, self._number_bert_ids
            ))

    def set_evaluation(self):
        self._evaluation = True

    @overrides
    def _read(self, filepath):
        # if `filepath` is a URL, redirect to the cache
        filepath = cached_path(filepath)
        logger.info("Reading instances from lines in file at: %s", filepath)
        datadir = os.path.dirname(filepath)
        dataset = None
        dep_path = os.path.join(datadir, 'dep')
        token_path = os.path.join(datadir, 'token')
        if self._pred_symbols:
            pred_symbol_path = os.path.join(datadir, "symbol.pred")
            pred_symbol_align = os.path.join(datadir, "alignments.pred")
            pred_sid2wid = os.path.join(datadir, "atom2word.pred")
        else:
            pred_symbol_path = False
            pred_symbol_align = False
            pred_sid2wid = False

        dataset = NP2P_data_stream.read_all_GenerationDatasets(
            filepath,
            dep_path=dep_path,
            token_path=token_path,
            ulfdep=True,
            pred_symbol_path=pred_symbol_path,
            pred_symbol_align=pred_symbol_align,
            pred_sid2wid=pred_sid2wid,
            presplit_fields=True,
            outer_feat_delim="_&_",
        )

        for ulf_data in dataset:
            yield self.text_to_instance(ulf_data)
        self.report_coverage()

    @overrides
    def text_to_instance(self, action_data) -> Instance:
        """Generates a dataset Instance from the raw data generated from
        read_all_GenerationDatasets.
        """
        # pylint: disable=arguments-differ

        fields: Dict[str, Field] = {}

        max_tgt_length = None if self._evaluation else 100

        if self.use_roberta:
            list_data = prepare_raw_input_dict(action_data, 
                                           bos=START_SYMBOL,
                                           eos=END_SYMBOL, 
                                           nis=NONE_SYMBOL, 
                                           null=NULL_SYMBOL,
                                           use_roberta=True,
                                           max_tgt_length=max_tgt_length)
        else:
            list_data = prepare_raw_input_dict(action_data, 
                                            bos=START_SYMBOL,
                                           eos=END_SYMBOL, 
                                           nis=NONE_SYMBOL, 
                                           null=NULL_SYMBOL,
                                           bert_tokenizer=self._word_splitter,
                                           max_tgt_length=max_tgt_length)

        fields["src_tokens"] = TextField(
            tokens=[Token(x) for x in list_data["src_tokens"]],
            token_indexers={k: v for k, v in self._token_indexers.items() if 'encoder' in k}
        )

        if list_data['src_token_ids'] is not None:
            fields['src_token_ids'] = ArrayField(list_data['src_token_ids'])
            self._number_bert_ids += len(list_data['src_token_ids'])
            self._number_bert_oov_ids += len(
                [bert_id for bert_id in list_data['src_token_ids'] if bert_id == 100])

        if list_data['src_token_subword_index'] is not None:
            fields['src_token_subword_index'] = ArrayField(
                list_data['src_token_subword_index'])

        fields["tgt_tokens"] = TextField(
            tokens=[Token(x) for x in list_data["tgt_tokens"]],
            token_indexers={k: v for k, v in self._token_indexers.items() if 'decoder' in k}
        )

        fields['action_tokens'] = TextField(
            tokens=[Token(x) for x in list_data['action_tokens']],
            token_indexers={k: v for k, v in self._token_indexers.items() if 'action_tokens' in k}
        )

        fields['action_feature_tokens'] = TextField(
            tokens=[Token(x) for lst in action_data.features_idx for x in lst],
            token_indexers={k: v for k, v in self._token_indexers.items() if 'action_feature' in k}
        )

        fields["src_pos_tags"] = SequenceLabelField(
            labels=list_data["src_pos_tags"],
            sequence_field=fields["src_tokens"],
            label_namespace="pos_tags"
        )

        fields["tgt_pos_tags"] = SequenceLabelField(
            labels=list_data["tgt_pos_tags"],
            sequence_field=fields["tgt_tokens"],
            label_namespace="pos_tags"
        )

        # NB: these NER embeddings are currently unused, just using the plain string representations in meta_src_ner_tags to generate features.
        fields["src_ner_tags"] = SequenceLabelField(
            labels=list_data["src_ner_tags"],
            sequence_field=fields["src_tokens"],
            label_namespace="ner_tags"
        )

        fields['src_lemmas'] = SequenceLabelField(
            labels=list_data['src_lemmas'],
            sequence_field=fields['src_tokens'],
            label_namespace='src_lemma_tags'
        )

        fields['ts_feats'] = SequenceLabelListField(
            labels=list_data['ts_feats'],
            sequence_field=fields['action_tokens'],
            label_namespace='action_feature_tags',
            strip_sentence_symbols=False,
        )

        fields['action_word_tokens'] = SequenceLabelField(
            labels=list_data['action2wid'],
            sequence_field=fields['action_tokens'],
            label_namespace='widx_tags',
            strip_sentence_symbols=False,
        )

        fields['action_symbol_tokens'] = SequenceLabelField(
            labels=list_data['action2sid'],
            sequence_field=fields['action_tokens'],
            label_namespace='sidx_tags',
            strip_sentence_symbols=True,
        )

        # Dependency tree as a field.
        dep_indices, dep_labels = action_data.sentence.tree.getAdjacencyFieldInputs()
        # Add three root nodes for the @end@, @none@, and @null@ tags.
        seqlen = len(dep_indices)
        dep_indices = dep_indices + ((seqlen, seqlen),
                                     (seqlen+1, seqlen+1),
                                     (seqlen+2, seqlen+2))
        dep_labels = dep_labels + (action_data.sentence.tree._ROOT_LABEL,
                                   action_data.sentence.tree._ROOT_LABEL,
                                   action_data.sentence.tree._ROOT_LABEL)
        fields['dep'] = AdjacencyField(
            indices=dep_indices,
            labels=dep_labels,
            label_namespace='dependency_labels',
            sequence_field=fields['src_tokens'],
        )

        # Non-tensorized data needed for the cache transition system.
        fields["raw_amr_data"] = MetadataField(action_data)
        fields['np2p_sentence'] = MetadataField(action_data.sentence)
        fields['raw_dep'] = MetadataField(action_data.sentence.tree)
        fields['sid2wid'] = MetadataField(action_data.sid2wid)
        fields['meta_src_tokens'] = MetadataField(list_data['src_tokens'])
        fields['meta_src_lemmas'] = MetadataField(list_data['src_lemmas'])
        fields['meta_tgt_tokens'] = MetadataField(list_data['tgt_tokens'])
        fields['meta_src_pos_tags'] = MetadataField(list_data['src_pos_tags'])
        fields['meta_src_ner_tags'] = MetadataField(list_data['src_ner_tags'])

        if self._evaluation:
            # Metadata fields, good for debugging
            fields["src_tokens_str"] = MetadataField(
                list_data["src_tokens"]
            )

            fields["tgt_tokens_str"] = MetadataField(
                list_data.get("tgt_tokens", [])
            )

        return Instance(fields)
