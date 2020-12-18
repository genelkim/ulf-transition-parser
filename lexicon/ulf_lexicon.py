import json
from collections import defaultdict

class ULFLexicon:

    def __init__(self, lexicon_file, strict_check=False):
        """
        strict: whether to require the token to be in the lexicon to allow an
            action. If False, words that don't appear in the lexicon can be
            generated in any manner.
        """
        # Build lexicon dictionary
        # The json file in the dictionary is assumed to be in the format of
        #   word -> pos -> ulf token
        json_data = json.loads(open(lexicon_file, 'r').read())
        self._lexicon_dict = defaultdict(dict, json_data)
        self._strict = strict_check

    def lookup(self, token, suffix=None):
        """Looks up the given token, suffix combination in the lexicon. If both
        are provided, it returns the ULF string. IF no suffix is provided, it
        returns the dictionary from suffix to ULF.

        Returns None if the combination does not exist in the lexicon.
        """
        token = token.upper()
        suffix_dict = self._lexicon_dict[token]
        if suffix is None:
            return suffix_dict
        else:
            suffix = suffix.upper()
            return suffix_dict[suffix] if suffix in suffix_dict else None

    def check_multiword_action(self, c, action):
        """For a multi word lexical item that is not in the lexicon to be
        allowed to generate without restrictions, no token-span must be in
        the lexicon.

        Returns whether this action is allowed by the lexicon.
        """
        if "SUFFIX:" not in action:
            return False
        for idx_offset in range(c.widx_width):
            for width in range(1, c.widx_width - idx_offset):
                span_str = c.getCurWord(idx_offset=idx_offset, width_override=width)
                if self.lookup(span_str):
                    return False # subspan found in lexicon.
        # No further restrictions.
        return True

    def check_action(self, c, action, strict_override=None):
        """Checks the current action, which must be a SUFFIX:* action, with the
        lexicon so see if it is allowed.
        """
        # If not a SUFFIX action, just fail.
        if "SUFFIX:" not in action:
            return False
        cur_tok = c.getCurWord()
        suffix = ":".join(action.split(":")[1:])

        #print("checking lexicon")
        #print("cur_tok: {}".format(cur_tok.upper()))
        #print("suffix: {}".format(suffix.upper()))
        if c.widx_width > 1:
            # If a multiword expression, special check.
            return self.check_multiword_action(c, action)
        elif self.lookup(cur_tok):
            # If the current token is in the lexicon, just check if it's in the lexicon.
            #print("action: {}".format(action))
            #print("suffix.upper(): {}".format(suffix.upper()))
            #print("entry: {}".format(self.lexicon[cur_tok.upper()]))
            #print("check: {}".format(suffix.upper() in self.lexicon[cur_tok.upper()]))
            return self.lookup(cur_tok, suffix)
        elif strict_override is not None:
            return not strict_override
        else:
            return not self._strict

    def check_name_action(self, c, action, strict_override=None):
        """Checks the current name generation action, which must be a SUFFIX:*
        action, with the lexicon so see if it is allowed.

        For now, names can't generate at all if the word is in the lexicon.
        """
        # If not a SUFFIX action, just fail.
        if "SUFFIX:" not in action:
            return False
        cur_tok = c.getCurWord()
        suffix = ":".join(action.split(":")[1:])

        if c.widx_width > 1:
            # If a multiword expression, special check.
            return self.check_multiword_action(c, action)
        elif self.lookup(cur_tok):
            return False
        elif strict_override is not None:
            return not strict_override
        else:
            return not self._strict


