#! /p/lisp/acl/linux/latest/alisp -#!
;; AUTHOR: Gene Kim  <gkim21@cs.rochester.edu>
;; Works for Allegro Lisp, other Lisps may need a different first #! sequence
;; Should eventually make this portable across Lisp implementations.

(when (not (>= (length (sys:command-line-arguments)) 1))
  (format t "USAGE: ulf-preprocess-script.cl formula~%")
  (format t "   o  formula in quotes ~%")
  (exit))
;(require 'asdf)
(load "~/quicklisp/asdf.lisp")
(load "~/quicklisp/setup.lisp")
;(load "~/research/mylisplibs/cl-util/load")
;(ql:quickload :util :silent t)
;(ql:quickload :ttt :silent t)
;(ql:quickload :cl-strings :silent t)
(ql:quickload :util)
(ql:quickload :ttt)
(ql:quickload :cl-strings)
;(asdf:operate 'asdf:load-op 'util)
;(load "io")
;(load "string")
(load "~/research/elcc/annotation_tool/ulf2amr-2018/ulf-preprocess.lisp")

(let* ((use-stdin nil)
       (formulahandle (if use-stdin *standard-input*
                     (open (nth 1 (sys:command-line-arguments)))))
       ;(lines (util:read-file-lines formulahandle))
       (lines (util:read-file-lines2 (nth 1 (sys:command-line-arguments))))
       (preproc-lines (mapcar #'ulf-preread-process 
                              (remove-if #'null lines)))
       (formulas (mapcar #'(lambda (x) 
                             (if (or (is-comment? x)
                                     (is-empty-line? x))
                                         x
                                         (util:read-all-from-string x)))
                         preproc-lines)))
  ;; Make sure newlines aren't introduced by increasing the right margin.
  (setq *print-right-margin* 10000000)
  (dolist (f formulas)
    (if (stringp f)
      (format t "~a~%" f)
      (format t "~s~%" (mapcar #'ulf-preprocess f)))))

