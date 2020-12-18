
(in-package :util)

;; Give cl-ppcre a nickname.
;(defpackage cl-ppcre (:nicknames re))
(add-nickname "CL-PPCRE" "RE")

;; List of characters that constitute whitespace in the trim functions.
(defparameter *trim-whitespace-chars*
  '(#\Space #\Newline #\Backspace #\Tab
    #\Linefeed #\Page #\Return #\Rubout))

;; Trims a string of whitespace.  Optionally you can provide the character list
;; that are trimmed, though in that case you might as well use string-trim.
(defun trim (str &optional trimlist-arg)
  (declare (type simple-string str)
           (type list trimlist-arg))
  (let ((trimlist (if trimlist-arg trimlist-arg
                    *trim-whitespace-chars*)))
    (declare (type list trimlist))
  (string-trim trimlist str)))

;; Same as trim, but only for the left side.
(defun left-trim (str &optional trimlist-arg)
  (declare (type simple-string str)
           (type list trimlist-arg))
  (let ((trimlist (if trimlist-arg trimlist-arg
                    *trim-whitespace-chars*)))
    (declare (type list trimlist))
  (string-left-trim trimlist str)))

;; Same as trim, but only for the right side.
(defun right-trim (str &optional trimlist-arg)
  (declare (type simple-string str)
           (type list trimlist-arg))
  (let ((trimlist (if trimlist-arg trimlist-arg
                    *trim-whitespace-chars*)))
    (declare (type list trimlist))
  (string-right-trim trimlist str)))

;; Converts a symbol to the corresponding string.
(defun sym2str (sym &key (preserve-package nil))
  (declare (type symbol sym)
           (type t preserve-package))
  (if preserve-package
    (format nil "~s" sym)
    (format nil "~s" (intern-symbols-recursive sym *package*))))

;; Converts an atom to the corresponding string (a generalization of sym2str).
(defun atom2str (atm &key (preserve-package nil))
  (declare (type atom atm)
           (type t preserve-package))
  (if preserve-package
    (format nil "~s" atm)
    (format nil "~s" (intern-symbols-recursive atm *package*))))

;; Converts a list to a string with a given delimiter between elements.
;; Elements are represented with the usual string representation.
(defun list-to-string (lst delim &optional (remove-newlines nil))
  (when (null lst)
    (return-from list-to-string ""))
  (reduce
    #'(lambda (x y)
        (concatenate 'string x delim y))
    (mapcar #'(lambda (a)
                (if remove-newlines
                  (cl-ppcre:regex-replace-all (format nil "~s" a) "\\n" " ")
                  (format nil "~s" a)))
            lst)))

;; Levenshtein edit distance algorithm.
;; From https://rosettacode.org/wiki/Levenshtein_distance#Common_Lisp
;; I changed the inner 'defun' to 'labels' so we silence some warnings.
(defun levenshtein (a b)
  (declare (optimize (safety 0)) ; assume no overflow
           (type simple-string a b))
  (let* ((la  (length a))
         (lb  (length b))
         (rec (make-array (list (the fixnum
                                     (1+ (the fixnum la)))
                                (the fixnum
                                     (1+ (the fixnum lb))))
                          :initial-element nil)))
    (declare (type fixnum la lb)
             (type (simple-array (or t character) *) rec))
    (labels
      ((leven (x y)
              (declare (type fixnum x y)
                       (ftype (function (fixnum fixnum) fixnum) leven))
              (cond
                ((zerop x) y)
                ((zerop y) x)
                ((aref rec x y) (aref rec x y))
                (t (setf (aref rec x y)
                         (the fixnum
                              (+ (if (char= (char a (- la x)) (char b (- lb y))) 0 1)
                                 (min (leven (1- x) y)
                                      (leven x (1- y))
                                      (leven (1- x) (1- y))))))))))
       (leven la lb))))

