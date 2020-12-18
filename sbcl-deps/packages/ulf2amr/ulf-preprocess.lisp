;; Sanity checker for ULF annotations.
;; This checks for possible errors using simple pattern matching methods,
;; recursing as much as it can.  This is NOT a full-fledged syntax checker.

(in-package :ulf2amr)

;; This function takes a string and add parentheses on the left and right to
;; make the parentheses match for the purposes of reading in s-expression.
;; This means that parentheses within complex symbols and withing strings
;; will be ignored.
;;  e.g. "HI)" -> "(HI)"
;;       "( this \"(string1\" ))" -> "(( this \"(string1\" ))"
;;       ")" -> "()"
;; Some miscellaneous spaces may get introduced between symbols and brackets.
;; These spaces will not affect the resulting s-expression when read into Lisp.
;;
;; NB: This function assumes that the only reason why a read would fail is
;; that the parens aren't matching.  Note that this function will crash if
;; quotes or symbol markers (e.g. |) don't match.
(defun make-string-paren-match (str)
  (labels
    ;; Heavy-lifting helper function.
    ;; Acc keeps track of previous completely interpreted strings, which may
    ;; need to be wrapped in parentheses.
    ((helper (str acc lpcount)
       (multiple-value-bind (sexpr endidx)
         ;; Allegro common lisp gives a speical error when there's an extra right paren.
         ;(handler-case (read-from-string str)
         ;  (end-of-file (c) (values str -1))
         ;  (excl::extra-right-paren-error (c) (values str -2)))
         (handler-case (read-from-string str)
           (end-of-file (c) (values str -1)))
         ;; Body of multiple-value-bind.
         (cond
           ;; If we got an end of file error and it's not empty, add a paren at
           ;; the end of the current string.
           ((and (= endidx -1) (not (equal "" (util:trim str))))
            (helper (concatenate 'string str ")") acc lpcount))
           ;; If we got a extra-right-paren-error, this means the paren was at
           ;; the beginning (otherwise, read-from-string just reports that we
           ;; didn't read the whole string).  So put the first character --
           ;; which should be a right-paren -- into acc, increment lpcount, and
           ;; recurse.
           ((= endidx -2)
            (let* ((trimmed (util:trim str))
                   (firstletter (subseq trimmed 0 1))
                   (restletters (subseq trimmed 1)))
              (assert (equal firstletter ")") (firstletter) (format nil "firstletter ~s" firstletter))
              (helper restletters (cons firstletter acc) (1+ lpcount))))
           ;; If we read the whole thing, we're done, return all of acc, str,
           ;; and left parens with space separation
           ((or (>= endidx (length str)) (equal "" (util:trim str)))
            (util:trim (cl-strings:join (cons (cl-strings:repeat "(" lpcount)
                                         (reverse (cons str acc)))
                                   :separator " ")))
           ;; If we stopped somewhere so include the current segment into acc
           ;; and recurse into the rest of the string.
           (t (helper (subseq str endidx)
                      (cons (subseq str 0 endidx)
                            acc)
                      lpcount)))))
     ) ; end of labels definitions.
    ;; Main body.
    (helper str nil 0)))


;; TODO: filter when choosing the type, but include in the formula (at the end).
(defparameter *filtered* (list '|,| '|;| '|"| '|'|
                               (list 'quotestart-i '|"|)
                               (list 'quotestart-i '|'|)))

;; Condition to check if an element is a filitered sentence-level operator.
;; Basically all sentence-level operators that are written as phrasal in the
;; surface form.
(defun phrasal-sent-op? (e)
  (or
    (adv-e? e)
    (adv-s? e)
    (adv-f? e)
    (member e '(not not.adv-e not.adv-s))))


(defparameter *embedding-op*
  '(to ka ke tht that quotestart-o quote-o more-than more-x-than adv-e adv-s adv-f))

; Lifts sentence-level operators to the sentence before the last embedding.
;
; Recurse:
;  - If atom return '()
;  - If the current operator is an embedding operator,
;     take the results from each argument recursion and add the operators.
;  - Other lists recurse and return the sent-ops
; The recursion is done with an in-order traversal so that the sentence level
; operators are applied in that order.
(defun lift-sent-ops (f)
  (labels
    (
     ;; Apply the sentence operators to formula.
     ;; If f is a sentence, then just wrap.
     ;; If f is a predicate, then add lambda expression.
     (apply-sent-ops
       (f ops)
       (cond
         ((null ops) f)
         ((or (sent? f) (tensed-sent? f))
          (reduce #'list ops :from-end t :initial-value f))
         ((pred? f)
          ;; TODO: generate unique var.  For now, just make it really unlikely
          ;; to get the same num.
          (let ((var (genvar (format nil "x~%" (random 1000000)))))
            (list '=l var
                  (reduce #'list ops :from-end t
                          :initial-value
                          (list var f)))))
         (t (format t "Not yet handled this case in apply-sent-ops!~%f: ~s~%ops ~s~%~%" f ops))))
     ;; Recurse into a list of arguments and apply the sent-ops to them.
     ;; If sent ops appear in the arguments, apply to the next non-sent-op.
     ;; If sent op is the last, apply it to the arg before.
     (arg-sent-ops
       (args sops)
       (cond
         ((and (null args)
               (not (null sops)))
          (list 'arg-sent-ops-sent-end sops))
         ((and (null args) (null sops)) nil)
          ;; If a sent-op, recurse after appending this to sops.
         ((and (listp args)
               (phrasal-sent-op? (car args)))
          (arg-sent-ops (cdr args)
                        (append sops (list (car args)))))
          ;; Otherwise... first recurse into remaining args.
          ;; If there are sent ops afterwards, apply them then sops.
          ;; If not, apply sops and add to recres.
         ((listp args)
          (let* ((recarg (helper (car args)))
                 (recres (arg-sent-ops (cdr args) cursops)))
            (cond
              ((and (listp recres)
                    (> (length recres) 0)
                    (equal (first recres) 'arg-sent-ops-sent-end))
               (list
                 (apply-sent-ops recarg
                                 (append sops
                                         (second recres)))))
              (t (cons (apply-sent-ops recarg ops) recres)))))
         (t (format t "arg-sent-ops CASE NOT HANDLED~%args: ~s~%sops: ~s~%~%" args sops))))
     ;; Recursive helper that does all the work, but returns different values
     ;; than the top-level function.
     (helper
       (f)
       (cond
         ((atom f) (list f '()))
         ((member (first f) *embedding-op*)
          ;; Recurse into each argument of the embedding operator and apply
          ;; sentence operators.
          (list (cons (car f)
                      (mapcar #'(lambda (x) (apply #'apply-sent-ops (helper x)))
                              (cdr f)))
                nil))
         ((listp f)
          ;; - Recurse into everything
          ;; - Check top-level sentence ops result in no additional sentence ops.
          ;; - Return all recursed sentence op in order.
          (let* ((recres (mapcar #'helper f))
                (top-sent-ops (remove-if-not #'(lambda (x) (phrasal-sent-op? (first x)))
                                         recres))
                (non-sent-ops (remove-if #'phrasal-sent-op?
                                         (mapcar #'first recres)))
                (all-sent-ops
                  (append
                    (mapcar #'(lambda (x) (if (phrasal-sent-op? (first x))
                                            (list (first x))
                                            (second x)))
                            recres))))
            (cond
               ;; If one of the sentence ops results in additional sent ops
               ;; raise error.
              ((not (null (remove-if #'(lambda (x) (null (second x)))
                                     top-sent-ops)))
               (format t "lift-sent-ops_helper Sent Op creates more sent-ops~%")
               (format t "sent-ops: ~s~%" top-sent-ops)
               (format t "helper input: ~s~%" f))
              (t (list non-sent-ops all-sent-ops)))))
         (t (format t "Unknown case for lift-sent-ops helper~%")
            (format t "f ~s~%" f))))
     ) ; end labels definitions.

  ;; Function body.
  (let ((helpres (helper f)))
    (apply-sent-ops (first helpres) (second helpres)))))

;; Naive function for determining if a string is a comment.
;; Simply checks if the first character, after trimming, is a semi-colon.
(defun is-comment? (str)
  (and (> (length (util:trim str)) 0)
       (equal (subseq (util:trim str) 0 1) ";")))

;; Returns whether this string is empty or consists only of whitespace.
(defun is-empty-line? (str)
  (= (length (util:trim str)) 0))

;; Run all preprocessing functions that need to occur on strings before they
;; are read into s-expressions.  Assumes that the given input is on a single
;; line.
(defun ulf-preread-process (line)
  (let ((fns (list #'make-string-paren-match)))
    (cond
      ;; If nil just return it.
      ((null line)
       nil)
      ;; If this line is commented, just return the line.
      ((is-comment? line)
       line)
      ;; Otherwise process.
      (t (reduce #'(lambda (acc new) (funcall new acc))
                 fns :initial-value line)))))

;; Run all the preprocessing functions.
;; TODO: complete the process.
(defun ulf-preprocess (f)
  (setq preproc-fns nil)
  (reduce #'(lambda (acc new) (funcall new acc))
          preproc-fns :initial-value f))

;; Merge object quotes.
;; Lift sentence-level operators that are phrasal in surface form:
;;  not, adv-e, adv-s, adv-f
;; Replace aliases.
;; Returns (preprocessed-f, sent-ops)
;; TODO: update quote handling to more recent version.
(defun ulf-preprocess-old (f)
  (labels
    (
     ; Merge object quotes.
     (merge-quoteo
       (fs)
       (if (atom fs) fs
       (let
         ((mfs (mapcar #'merge-quoteo fs)))
         (cond
           ((and (equal (list 'quotestart-o '|"|)
                        (first mfs))
                 (equal '|"| (last mfs)))
            (cons (list 'quoteo '|"|)
                    (cdr (reverse (cdr (reverse mfs))))))
           ((and (equal (list 'quotestart-o '|'|)
                        (first mfs))
                 (equal '|'| (last mfs)))
            (cons (list 'quoteo '|'|)
                  (cdr (reverse (cdr (reverse mfs))))))
           ((or
                 (and
                   (equal (list 'quotestart-o '|'|)
                          (first mfs))
                   (equal '|"| (last mfs)))
                (and
                  (equal (list 'quotestart-o '|"|)
                         (first mfs))
                  (equal '|'| (last mfs))))
               (format t "Object quote with mismatched quote types."))
          (t mfs)))))
     (remove-extra-parens
       (f)
       (cond
         ((atom f) f)
         ((= (length f) 1) (remove-extra-parens (car f)))
         (t (mapcar #'remove-extra-parens f))))
     ); end of labels definitions.
    (let* ((merged (merge-quoteo f))
           (lifted (lift-sent-ops merged))
           (parenrm (remove-extra-parens lifted)))
      (format t "merged ~s~%" merged)
      (format t "lifted ~s~%" lifted)
      (format t "parenrm ~s~%" parenrm)
      parenrm)))

