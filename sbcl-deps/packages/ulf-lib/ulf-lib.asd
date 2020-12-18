;; ULF Inferface and Manipulation Library.

(asdf:defsystem :ulf-lib
  :depends-on (:ttt :cl-strings :cl-ppcre :cl-util :lisp-unit)
  :components ((:file "package")
               (:file "suffix")
               (:file "semtype")
               (:file "ttt-lexical-patterns")
               (:file "ttt-phrasal-patterns")
               (:file "gen-phrasal-patterns")
               (:file "search")
               (:file "macro")
               (:file "preprocess")
               (:file "composition")
               ))

