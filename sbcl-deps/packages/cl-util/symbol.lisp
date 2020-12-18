
(in-package :util)

(defun split-into-atoms (atm); tested
;````````````````````````````
; Useful for applying TTT to processing symbols as list of separate
; symbolic characters. Caveat: Watch out for {!,~,^,*,+,?} as part of 'atm'
; as they are specially interpreted by TTT.
  (mapcar #'intern
          (mapcar #'(lambda (ch) (format nil "~a" ch))
                  (coerce (mkstr atm) 'list))))

(defun fuse-into-atom (atm-list &key (pkg *package*)); tested
;``````````````````````````````
; Make a single atom out of the list of atoms
; e.g., (fuse-into-atom '(this - and - that)) --> THIS-AND-THAT
;; TODO: make this more robust to catch any locked package, not just COMMON-LISP.
  (intern (apply #'concatenate 'string
                 (mapcar #'(lambda (x)
                             (declare (type (or character symbol) x))
                             (string x))
                         atm-list))
          (if (eq pkg (find-package "COMMON-LISP"))
            *package*
            pkg)))

;; A parameter for default output/calling package for the interning macros.
(defparameter *intern-caller-pkg* nil)

;; Macro for pre- and post- interning symbols.
;; bgnval: initial symbol or s-expr of symbols
;; midval: name of variable storing bgnvar after it is interned into the ulf-lib package
;; callpkg: optional argument for the package that the output should be interned to
;;          if nil, it defaults to the value *ulf-lib-caller-pkg*
;; body: body of the code (using midval)
;; outval: immediate output of the body
(defmacro inout-intern ((bgnval midval inpkg &key (callpkg nil)) &body body)
  `(let* ((,midval (intern-symbols-recursive ,bgnval ,inpkg))
          (outval (progn ,@body)))
     (cond
       (,callpkg (intern-symbols-recursive outval ,callpkg))
       (*intern-caller-pkg* (intern-symbols-recursive outval *intern-caller-pkg*))
       (t outval))))
;; Same as inout-intern macro but only performs the pre- interning portion.
;; Interns the incoming symbols and stores it in midval before evaluating
;; the body.
(defmacro in-intern ((bgnval midval inpkg) &body body)
  `(let* ((,midval (intern-symbols-recursive ,bgnval ,inpkg)))
     ,@body))

(defun preslash-unsafe-chars (char-string)
  "Prefix '\' to unsafe characters # ` ' : ; , . \ | in 'aString'."
  (let ((chars (coerce char-string 'list)) result)
       (dolist (ch chars)
           (cond ((alphanumericp ch) (push ch result))
                 ((member ch '(#\( #\) #\")) (push ch result)); unbalanced "
                 ((member ch
                   '(#\# #\` #\' #\: #\; #\, #\. #\\ #\|) )
                  (push #\\ result) (push ch result) )
                 (T (push ch result)) ))
        (coerce (reverse result) 'string)))

