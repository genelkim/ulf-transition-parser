;; ULF Sanity Checker

(in-package :cl-user)

(defpackage :ulf-sanity-checker
  (:use :cl :ttt :cl-util :ulf-lib)
  (:shadowing-import-from :alexandria)
  (:export sanity-check))

;; Inherit all the ulf-lib external symbols.
;; ulf-lib contains ULF segment matching functions (e.g. noun?) which are used
;; in the TTT patterns.
(use-package :ulf-lib)

