;; ULF to AMR

(asdf:defsystem :ulf2amr
  :name "ulf2amr"
  :version "0.0.2"
  :author "XXXX"
  :depends-on (:ttt :cl-strings :cl-util :cl-ppcre :alexandria :ulf-lib :lisp-unit)
  :components ((:file "package")
               (:file "penman")
               (:file "ulf-preprocess")
               (:file "ulfamr-postprocess")
               (:file "ulf2amr")
               (:file "amr2ulf")
               (:file "ulfctp-interface"))) 

