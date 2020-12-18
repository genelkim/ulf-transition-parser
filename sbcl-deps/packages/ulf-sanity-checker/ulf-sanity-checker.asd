;; ULF Sanity Checker

(asdf:defsystem :ulf-sanity-checker
  :name "ulf-sanity-checker"
  :version "0.3.0"
  :author "XXXX"
  :depends-on (:ttt :cl-util :ulf-lib :alexandria)
  :components ((:file "package")
               (:file "ttt-bad-patterns")
               (:file "ulf-sanity-checker")
               (:file "ulf-ctp-interface")))

