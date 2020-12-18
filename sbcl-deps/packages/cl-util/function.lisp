
(in-package :util)

(defun nilfn (x)
  "Function that just returns nil."
  (declare (ignore x))
  nil)

(defun tfn (x)
  "Function that just returns t."
  (declare (ignore x))
  t)

(defun compose (&rest args)
  "Composes any number of one argument functions.
  Modified from https://rosettacode.org/wiki/Function_composition#Common_Lisp"
	(lambda (x)
    (reduce #'(lambda (acc fn) (funcall fn acc))
            (reverse args)
            :initial-value x)))

(defun grouped-arglist (fn)
  "Returns an alist of the function argument list grouped by argument type.
   e.g. (grouped-arglist #'reduce)
        -> ((&required . (function sequence))
            (&key . (key from-end start end initial-value))
            (&optional . nil)
            (&rest . nil))
        (grouped-arglist #'mapcar)
        -> ((&required . (function list))
            (&key . nil)
            (&optional . nil)
            (&rest . (more-lists)))"
  (let* ((arglst (conium:arglist fn))
         (curtype '&required)
         (argtypes '(&required &key &optional &rest))
         (all-groups (pairlis argtypes '(nil nil nil nil)))
         curlst)
    (loop
      for arg in arglst
      do (cond 
           ;; Hit a new argument type, so last argument type is complete. Store
           ;; the arglist of the last type and start a new one.
           ((member arg argtypes)
            (setf (cdr (assoc curtype all-groups)) curlst)
            (setf curtype arg)
            (setf curlst nil))
           ;; Otherwise add arg to list. If there's a default value, ignore it.
           (t 
             (push (if (consp arg) (car arg) arg) curlst))))
    (unless (null curlst)
      (setf (cdr (assoc curtype all-groups)) curlst))
    all-groups))

(defun argnum (fn &optional (argtypes '(&required &key &optional &rest)))
  "Returns the number of required arguments in the given function.
  Required arguments don't include &key &optional or &rest arguments."
  (let ((grp-arglst (grouped-arglist fn)))
    (apply #'+
           (mapcar #'(lambda (argtype)
                       (length (cdr (assoc argtype grp-arglst))))
                   argtypes))))

;;; Getters for the number of arguments of each type. 
(defun required-argnum (fn) (argnum fn '(&required)))
(defun key-argnum (fn) (argnum fn '(&key)))
(defun optional-argnum (fn) (argnum fn '(&optional)))

