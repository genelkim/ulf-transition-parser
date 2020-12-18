
(in-package :util)

;; Give cl-ppcre a nickname.
;;(defpackage cl-ppcre (:nicknames re))
(add-nickname "CL-PPCRE" "RE")

(define-constant eof (gensym))

(defmacro do-lines (var path &rest body)
  (alexandria:with-gensyms (p str)
    `(let ((,p (probe-file ,path)))
       (when ,p
         (with-infile ,str ,p
           (do ((,var (read-line ,str nil ,eof)
                      (read-line ,str nil ,eof)))
               ((eql ,var ,eof))
             ,@body))))))


(defmacro do-lines-slurp (var path &rest body)
  (alexandria:with-gensyms (p str)
    `(let ((,p (slurp ,path)))
       (with-input-from-string (,str ,p)
         (do ((,var (read-line ,str nil ,eof)
                    (read-line ,str nil ,eof)))
             ((eql ,var ,eof))
           ,@body)))))


(defmacro with-infile (var fname &rest body)
  (alexandria:with-gensyms (f)
    `(let ((,f ,fname))
       (with-open-file (,var ,f :direction :input)
         ,@body))))


(defmacro with-outfile (var fname &rest body)
  (alexandria:with-gensyms (f)
    `(let ((,f ,fname))
       (with-open-file (,var ,f :direction :output
                                :if-exists :supersede)
         ,@body))))


(defmacro bind (&rest args)
  `(multiple-value-bind ,@args))


(defmacro in-case-error (expr err)
  (alexandria:with-gensyms (val cond)
    `(bind (,val ,cond) (ignore-errors ,expr)
         (if (typep ,cond 'error)
               (progn
                 ,err
                 (error ,cond))
             ,val))))


(defun slurp (file)
  (with-open-file (stream file)
    (let ((seq (make-string (file-length stream))))
      (read-sequence seq stream)
      seq)))


(defun intern-symbols-recursive (tree package)
  ;(list tree package))
  (cond ((null tree) nil)
        ((or (numberp tree) (stringp tree) (functionp tree))
         tree)
        ((keywordp tree) tree) ; don't intern keywords
        ((atom tree) (intern (symbol-name tree) package))
        ((atom (cdr tree)) ; A dotted pair
         (cons (intern-symbols-recursive (car tree) package)
               (intern-symbols-recursive (cdr tree) package)))
        (t (mapcar #'(lambda (x)
                       (intern-symbols-recursive x package))
                   tree))))


;; (NN dog (s)) -> (NN DOG {S})
;; (NN dog ()) -> (NN DOG {})
(defun replace-lemma-parens (string)
;; TODO: replace with cl-ppcre
  (re:regex-replace-all "\\(([^\\s\\(\\)]*)\\)"
                      string
                      "{\\1}"))


;; (NN DOG {S}) -> (|NN| DOG {S})
(defun symbolize-node-tags (string)
;; TODO: replace with cl-ppcre
  (re:regex-replace-all "\\(([^\\s\\)\\(]+)\\s+"
                      string
                      "(|\\1| "))


;; (|NN| DOG {S}) -> (|NN| |DOG {S}|)
(defun symbolize-lemmas (string)
;; TODO: replace with cl-ppcre
  (re:regex-replace-all "\\s+([^\\)\\(]+)\\)"
                      string
                      " |\\1|)"))


(defun tree-from-string (string &optional (package *package*))
  "Takes a string representing a tree, returns a lisp object.
   Assumes tree is represented as embedded parenthesis.
   Disallowed symbols such as ',' will be escaped.
   No matter input, output will be uppercase."
  (intern-symbols-recursive
   (read-from-string
    (escape-all-symbols
     (symbolize-lemmas
      (symbolize-node-tags
       ;;(replace-lemma-parens
       (string-upcase string)))));)
   package))


(defun escape-all-symbols (string)
  (re:regex-replace-all "([^\\s\\)\\(]+)"
                        (re:regex-replace-all "\\|" string "")
                        "|\\1|"))


(defun mintersection (x y &key (predicate '<)
                     (when-equal #'(lambda (a b)
                                     (declare (ignore b))
                                     a)))
  "Returns an intersection, as a vector.

   By default assumes x and y are lists or vectors of ints sorted least to
   greatest."

  (let ((a (if (vectorp x) x (apply #'vector x)))
        (b (if (vectorp y) y (apply #'vector y))))
    (loop
       with A-i = 0
       with B-i = 0
       with A-end = (length A)
       with B-end = (length B)
       with result = (make-array (min A-end B-end)
                              :adjustable t
                              :fill-pointer 0)
       finally (return result)
       while (and (< A-i A-end) (< B-i B-end))
       for A-val = (aref A A-i)
       for B-val = (aref B B-i)
       do
         (if (and (not (funcall predicate A-val B-val))
                 (not (funcall predicate B-val A-val)))
             ;; then must be equal
             (let ((res (funcall when-equal A-val B-val)))
               (when res (vector-push-extend res result))
               (incf A-i) (incf B-i))
             ;; else
             (if (funcall predicate A-val B-val)
                ;; then
                (incf A-i)
                ;; else
                (incf B-i))))))

;; (defun maintersection (x y)
;;   (let ((z (merge 'vector x y #'<)))
;;     (loop
;;        with end = (length z)
;;        for i from 0 below (- end 1)
;;        when (eq (aref z i) (aref z (+ 1 i))) collect (aref z i))))

(defun get-line (file offset)
  (with-open-file (input file)
    (file-position input offset)
    (read-line input)))

(defun extract-sentence (tree-string &optional (is-BNC t))
  ;; when is-BNC is true, replace the markup with something approximate.
  (let ((raw
         (apply 'concatenate
               (cons 'string
                     (re:all-matches-as-strings "\\s[^\\s\\(\\)]+" tree-string)))))
    (when is-BNC
      (loop for (pattern replacement) in
         ;; based on: http://www.natcorp.ox.ac.uk/docs/userManual/codes.xml.ID=codes
           '(("-LRB-"       "(")
             ("-RRB-"       ")")
             ("&amp;"       "&")
             ("&ast;"       "*")
             ("&bquo;"       "`")
             ("&dollar;""$")
             ("&pound;""$UK")
             ("&equo;"       "'")
             ("&hellip;""...")
             ("&ins;"       "''")
             ("&lsqb;"       "[")
             ("&mdash;"       "-")
             ("&ndash;"       "-")
             ("&quot;"       "'")
             ("&rsqb;"       "]")
             ("&times;"       "x"))
         do
           (setf raw (re:regex-replace-all pattern raw replacement))))
    raw))


;; From 'On Lisp', but it's the same as McCarthy's 1978 code, only using
;; 'labels'.
;(defun flatten (x)
;  (labels ((rec (x acc)
;             (cond ((null x) acc)
;                   ((atom x) (cons x acc))
;                   (t (rec (car x) (rec (cdr x) acc))))))
;    (rec x nil)))


(defun contains-underscore (atm)
  (member #\_ (coerce (string atm) 'list)))


(defun split-at-char (atm c)
  "Split the given literal atom into a part ending in an underscore
   and a part after the underscore; If there is no underscore, return
   nil; o/w return a list (metavar test) consisting of two atoms
   corresponding to the respective parts above; if there was nothing
   after the underscore, use test = nil."
  (prog (chars l metavar test)
        (setq chars (coerce (string atm) 'list))
        (setq test (member c chars))
        (if (null test)
            (return nil))
        (setq l (1- (length test))) ; length of 'test' atom
        (if (zerop l)
            (setq test nil)
          (setq test (intern (coerce (cdr test) 'string))) )
        (setq metavar (intern (coerce (butlast chars l) 'string)))
        (return (list metavar test))))


(defun split-at-underscore (atm)
  (split-at-char atm #\_))


;; Member but returning t/nil. Works for lists or my hash-sets (that is, a
;; hash where the value is meaningless and all we care about is whether
;; an element is present).
(defun memberp (x listy)
  (if (hash-table-p listy)
      (gethash x listy)
      (not (null (member x listy)))))


(defun subst-in-symb (symb old new)
  (intern (re:regex-replace-all (string old) (string symb) (string new))))


;; Source: Paul Graham, 'On Lisp'
(defun prune (test tree)
  (labels ((rec (tree acc)
             (cond ((null tree) (nreverse acc))
                   ((consp (car tree))
                    (rec (cdr tree)
                         (cons (rec (car tree) nil) acc)))
                   (t (rec (cdr tree)
                           (if (funcall test (car tree))
                               acc
                               (cons (car tree) acc)))))))
    (rec tree nil)))


(defun safe-car (x) (if (listp x) (car x)))

(defun safe-first (x) (if (listp x) (car x)))

(defun safe-second (x) (if (listp x) (second x)))

(defun safe-third (x) (if (listp x) (third x)))

(defun safe-fourth (x) (if (listp x) (fourth x)))

(defun safe-fifth (x) (if (listp x) (fifth x)))

(defun safe-cdr (x) (if (listp x) (cdr x)))

(defun safe-cddr (x) (if (listp x) (cddr x)))

(defun safe-cdddr (x) (if (listp x) (cdddr x)))

;; As in Python's re.sub.
;; TODO: replace this with cl-ppcre
(defun sub (pat repl str)
  (re:regex-replace-all pat str repl))


;; From Paul Graham's "On Lisp":
;; Given one or more arguments, e.g. symbols or numbers, join them into
;; a single symbol.
;;   (symb 'hello 'there)
;;   HELLOTHERE
(defun symb (&rest args)
  (values (intern (apply #'mkstr args))))


;; From Paul Graham's "On Lisp":
;; Given one or more arguments, e.g. symbols or numbers, join them into
;; a single string.
;;   (mkstr 'hello 'there)
;;   "HELLOTHERE"
(defun mkstr (&rest args)
  (with-output-to-string (s)
    (dolist (a args)
      (if (keywordp a)
          (prin1 a s))
      (princ a s))))

