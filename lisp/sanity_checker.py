"""
An interface to the Common Lisp ULF sanity sanity checker. This is meant to be
a rough-and-ready version of the type checking mechanism.
"""

import cl4py
import sys
import os
from .amr2ulf import amr2ulf

class ULFSanityChecker(object):
    def __init__(self):
        lisp = cl4py.Lisp(quicklisp=True)
        cl = lisp.function('find-package')('CL')
        ql = cl.find_package('QL')
        ql.quickload('ULF-SANITY-CHECKER')
        self.__exists_bad_pattern = lisp.function('ulf-sanity-checker::exists-bad-pattern?')

    def exists_bad_pattern(self, ulfamr_string):
        """A wrapper for performing the type checking.
        """
        # we only need to interface with Lisp at one point.
        #ulfamr_string = '(' + ulfamr_string + ')'
        ulfstr = amr2ulf.amr2ulf(ulfamr_string)
        rawres = self.__exists_bad_pattern(ulfstr)
        if rawres == ():
            return False
        elif rawres == True:
            return True
        else:
            raise Exception("Unknown exists-bad-pattern return value: {}".format(rawres))

# Keep one global instance to minimize overhead.
ulf_sanity_checker = ULFSanityChecker()

