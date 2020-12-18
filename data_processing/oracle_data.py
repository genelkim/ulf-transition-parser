import json


class OracleExample(object):
    def __init__(self, toks_=None, lems_=None, poss_=None, ners_=None, symbols_=None,
            map_infos_=None, feats_=None, actions_=None, foci_=None,
            sw_align_=None):
        self.toks, self.lems, self.poss, self.ners = toks_, lems_, poss_, ners_
        self.symbols, self.map_infos = symbols_, map_infos_
        self.feats, self.actions = feats_, actions_
        self.word_align = [w for w, _ in foci_]
        self.symbol_align = [s for _, s in foci_]
        self.symbol_to_word = sw_align_

    def toJSONobj(self):
        json_obj = {}
        # Basic Annotation Info
        json_obj["text"] = " ".join(self.toks)
        json_obj["annotation"] =  {
            "lemmas": " ".join(self.lems),
            "POSs": " ".join(self.poss),
            "NERs": " ".join(self.ners),
            "mapinfo": "_#_".join(self.map_infos),
        }
        # symbols
        json_obj["symbols"] = " ".join(self.symbols)
        json_obj["symbol_tokens"] = self.symbols
        # Actions
        json_obj["actionseq"] = " ".join(self.actions)
        json_obj["actions"] = self.actions
        # Alignments
        word_align_str = " ".join([str(idx) for idx in self.word_align])
        symbol_align_str = " ".join([str(idx) for idx in self.symbol_align])
        symbol_to_word_str = " ".join([str(idx) for idx in self.symbol_to_word])
        json_obj["alignment"] = {
            "word-align": word_align_str,
            "symbol-align": symbol_align_str,
            "symbol-to-word": symbol_to_word_str,
        }
        # Features
        json_obj["feats"] = "_&_".join(["_#_".join(feat) for feat in self.feats])
        return json_obj


class OracleData(object):
    def __init__(self):
        self.examples = []
        self.num_examples = 0

    def addExample(self, example):
        self.examples.append(example)
        self.num_examples += 1

    def toJSON(self, json_output):
        dataset = []
        for idx in range(self.num_examples):
            curr_json_example = self.examples[idx].toJSONobj()
            dataset.append(curr_json_example)

        with open(json_output, "w") as json_wf:
            json.dump(dataset, json_wf)
            json_wf.close()
