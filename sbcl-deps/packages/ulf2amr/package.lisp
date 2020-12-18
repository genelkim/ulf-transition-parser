(in-package :cl-user)

(defpackage :ulf2amr
  (:use :cl :ttt :cl-util :ulf-lib :lisp-unit)
  (:shadowing-import-from :cl-ppcre)
  (:shadowing-import-from :alexandria)
  (:shadowing-import-from :cl-strings)
  (:export ulf2amr
           amr2ulf))

(in-package :ulf2amr)

(defparameter *ulf2amr-debug* nil)
(use-package :ulf-lib)

