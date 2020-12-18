import cl4py
import sys
import os
from lisp.amr2ulf import escape_lisp_string
from stog.utils.logging import init_logger

logger = init_logger()

class ULFlib(object):
    def __init__(self):
        lisp = cl4py.Lisp(quicklisp=True)
        cl = lisp.function('find-package')('CL')
        ql = cl.find_package('QL')
        ql.quickload('ULF-LIB')
        self.__compose_types = lisp.function('ulf-lib::left-right-compose-type-string!')
        self.__list_compose_types = lisp.function('ulf-lib::list-left-right-compose-type-string!')
        self.__list_compose_memo = dict()
        self.__ulf_type = lisp.function('ulf-lib::str-ulf-type-string?')

    def ulf_type(self, lispstr):
        """A wrapper for ulf-lib::str-ulf-type-string? with some special
        handling specific to the cache transition parser representation.

        Special rules:
        - COMPLEX is the type for COMPLEX, since it is a ULF-AMR specific symbol.
        - nil is changed to "..."
        """
        if lispstr == "COMPLEX":
            return "COMPLEX"
        else:
            try:
                escaped_str = escape_lisp_string(lispstr)
                raw = self.__ulf_type(escaped_str)
                if raw == ():
                    return "..."
                else:
                    return raw
            except Exception as e:
                logger.error("Exception while computing type")
                logger.error("lispstr: {}\nescaped: {}".format(lispstr, escaped_str))
                logger.error("{}".format(e))
                return "..."

    def compose_types(self, type1, type2):
        """A wrapper for ulf-lib::compose-type-string! with some special
        handling specific to the cache transition parser representation.

        Special rules:
        - COMPLEX takes whatever type its INSTANCE child has
        - "..." can't compose with anything
        """
        if type1 == "COMPLEX":
            # TODO: only allow this for INSTANCE of COMPLEX
            return type2
        if type2 == "COMPLEX":
            return type1
        elif type1 == "..." or type2 == "...":
            return "..."
        else:
            try:
                new_type = self.__compose_types(type1, type2)
                if new_type == ():
                    new_type = "..."
                return new_type
            except Exception as e:
                logger.error("Exception while composing type")
                logger.error("type1: {}\ntype2: {}".format(type1, type2))
                logger.error("{}".format(e))
                return "..."

    def directional_compose_types(self, type1, type2, direction, memoize=True):
        """A wrapper for ulf-lib::list-left-right-compose-type-string! with some
        special handling specific to the cache transition parser
        representation.

        Special rules:
        - COMPLEX takes whatever type its INSTANCE child has
        - "..." can't compose with anything
        """
        if direction not in ["left", "right"]:
            raise ValueError("Invalid value for composition direction: {}".format(direction))

        if type1 == "COMPLEX" and direction == "right":
            # TODO: only allow this for INSTANCE of COMPLEX
            return type2
        if type2 == "COMPLEX" and direction == "left":
            return type1
        elif type1 == "..." or type2 == "...":
            return "..."
        else:
            try:
                if memoize and (type1, type2) in self.__list_compose_memo:
                    new_type, compdir =\
                            self.__list_compose_memo[(type1, type2)]
                else:
                    new_type, compdir, _ =\
                            self.__list_compose_types(type1, type2)
                    if memoize:
                        self.__list_compose_memo[(type1, type2)] =\
                                (new_type, compdir)
                        revcompdir = "left" if compdir == "right" else "right"
                        self.__list_compose_memo[(type2, type1)] = \
                                (new_type, revcompdir)
                if new_type == () or direction != compdir:
                    new_type = "..."
                return new_type
            except Exception as e:
                logger.error("Exception while composing type")
                logger.error("type1: {}\ntype2: {}".format(type1, type2))
                logger.error("{}".format(e))
                return "..."


# Single instantiation of the Lisp interface.
ulf_lib = ULFlib()


class MemoizedULFTypes(object):
    """Memoized container for ulf_lib to avoid re-evaluating types.
    """
    def __init__(self):
        self._ulf_lib = ulf_lib
        self.memo = {}

    def get_type(self, expr):
        if expr in self.memo:
            return self.memo[expr]
        else:
            return self._ulf_lib.ulf_type(expr)

