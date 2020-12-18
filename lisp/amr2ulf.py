"""
An interface to the Common Lisp implementation for mapping AMR-ized ULFs back
to the raw ULF formats.
"""

import cl4py
import sys
import os

ESCAPE_REPLACEMENTS = {
    '{': '\{',
    '}': '\}',
    ',': '\,',
    #'.': '\.',
    '"': '\\"',
    ';': '\;',
    '\'': '\\\'',
}

def escape_lisp_string(lispstr):
    escaped = lispstr
    for fro, to in ESCAPE_REPLACEMENTS.items():
        escaped = escaped.replace(fro, to)
    return escaped

class AMR2ULF(object):
    def __init__(self):
        lisp = cl4py.Lisp(quicklisp=True)
        cl = lisp.function('find-package')('CL')
        ql = cl.find_package('QL')
        ql.quickload('ULF2AMR')
        self.__ulfctp_amr2ulf = lisp.function('ulf2amr:ulfctp-amr2ulf')

    def amr2ulf(self, ulfamr_string):
        """A wrapper for he common lisp function ulf2amr:ulfctp-amr2ulf.
        """
        return self.__ulfctp_amr2ulf(escape_lisp_string(ulfamr_string))

# Single instantiation of the Lisp interface since there's a large startup overhead.
amr2ulf = AMR2ULF()

