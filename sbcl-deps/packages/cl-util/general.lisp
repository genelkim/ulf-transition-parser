
(in-package :util)

(defmacro define-constant (name value &optional doc)
  "ANSI compliant, robust version of defconstant."
	`(defconstant ,name (if (boundp ',name) (symbol-value ',name) ,value)
								,@(when doc (list doc))))

(defun add-nickname (package nickname)
  "Adds a package nickname."
  (let ((pkg (if (stringp package)
               (find-package package)
               package)))
    (check-type pkg package)
    (check-type nickname string)
    (rename-package pkg (package-name pkg)
                    (adjoin nickname (package-nicknames pkg)
                            :test #'string=))))

(defun safe-intern (strsym &optional pkg)
  "Safe intern.
  strsym can be a string or a symbol and it interns it."
  (let ((fnpkg (if (eq pkg (find-package "COMMON-LISP"))
                 *package*
                 pkg)))
    (cond
      ((stringp strsym) (intern strsym fnpkg))
      ((symbolp strsym) (intern (symbol-name strsym) fnpkg))
      ((numberp strsym) strsym)
      (t (error
           (format nil
                   "The input to safe-intern is not a supported data type.~%Value: ~s~%Type: ~s~%"
                   strsym (type-of strsym)))))))

;;; Functions for determining lisp implementation.

(defvar *lisp-implementation-to-shorthand*
  '(("SBCL" . sbcl)
    ("International Allegro CL Free Express Edition" . acl)
    ("International Allegro CL Enterprise Edition" . acl)
    ("CMU Common Lisp" . cmucl)))
(defun lisp-impl ()
  "Returns a symbol of the lisp implementation. 
  This uses the implementation shorthands rather than the idiosyncratic names
  returned from #'CL:LISP-IMPLEMENTATION-TYPE."
  (let ((impl-str (lisp-implementation-type)))
    (cdr (assoc impl-str *lisp-implementation-to-shorthand* :test #'string=))))
;;; Functions for specific implementations.
(defun sbcl-impl? ()
  (eql (lisp-impl) 'sbcl))
(defun acl-impl? ()
  (eql (lisp-impl) 'acl))
(defun cmucl-impl? ()
  (eql (lisp-impl) 'cmucl))

(defun safe-symbol-eval (sym pkg-name)
  "Evaluate symbol w.r.t. a package safely.
  Evaluates a symbol with respect to the given package iff the package is
  available and the symbol is found in that package."
  (let ((pkg (find-package pkg-name)))
    (if pkg 
      (eval (find-symbol (symbol-name sym) pkg))
      nil)))

(defun argv ()
  "Gives the argv depending on the distribution."
  (or
    #+SBCL sb-ext:*posix-argv*
    #+LISPWORKS system:*line-arguments-list*
    #+CMU extensions:*command-line-words*
    #+ALLEGRO (sys:command-line-arguments)
    #+CLISP *args*
    nil))

