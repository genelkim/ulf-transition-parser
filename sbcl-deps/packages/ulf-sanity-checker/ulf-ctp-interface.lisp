;;; Functions to interface the ulf-sanity-checker with ULF cache transition
;;; parser (ULFCTP). The sanity checking is reduced to a binary check of
;;; whether a bad pattern exists or not.

(in-package :ulf-sanity-checker)

;; ULFCTP specific bad pattern tests. Doesn't include ones that might need to be
;; ignored, or only indicate suspicious patterns
(defparameter *ulfctp-bad-pattern-tests*
  (list
    #'bad-det?
    #'bad-prep?
    #'bad-tensed-sent-op?
    #'bad-sent-op?
    #'bad-verb-reifier?
    #'bad-noun-reifier?
    #'bad-plur?
    #'bad-aux?
    #'bad-advformer?
    #'bad-detformer?
    #'bad-np-preds?
    #'bad-n-preds?
    #'bad-sent-punct?
    #'bad-double-tense?
    #'no-periods-or-commas?
    #'old-ps-ann?
    #'bad-possessive?
    #'bad-pu?
    #'bad-flat-mod?
    #'bad-equal?
    #'conservative-bad-sent-reifier?
    #'bad-noun-pp?
    #'bad-verb-sent?
    #'bad-pasv?
    #'bad-verb-args?
    #'bad-name-decomp?
    #'bad-sent-term?
    #'bad-adv-a-arg?
    #'old-adj-mod?
    ))

;
; Some ULFCTP specific patterns for macros.
; These allow for intermediate stages of completion.
;
(defparameter *ttt-ulfctp-bad-sub*
  '(!1
     (sub _!2 _!3 _+)
     (_+ sub _*)
     (sub _!4 (!5 ~ contains-sub-var?))
     ))
(defparameter *ttt-ulfctp-bad-rep*
  '(!1
     (rep _!2 _!3 _+)
     (_+ rep _*)
     (rep (!4 ~ contains-rep-var?) _!5)))
(defparameter *ttt-ulfctp-bad-qt-attr*
  '(!1
     (_+ qt-attr _*)
     (qt-attr _! _+)
     (qt-attr (!2 ~ contains-qt-attr-var?))))
(defun ulfctp-bad-sub? (x) (ttt:match-expr *ttt-ulfctp-bad-sub* x))
(defun ulfctp-bad-rep? (x) (ttt:match-expr *ttt-ulfctp-bad-rep* x))
(defun ulfctp-bad-qt-attr? (x) (ttt:match-expr *ttt-ulfctp-bad-qt-attr* x))

(defparameter *ulfctp-raw-bad-pattern-tests*
  (list
    #'bad-single-bracket?
    #'ulfctp-bad-sub?
    #'ulfctp-bad-rep?
    #'ulfctp-bad-qt-attr?
    #'bad-rel-sent?
    #'bad-voc?
    ))


(defun check-for-single-bad-pattern (f pattern-tests)
  "Recursively check for bad patterns on formula.
  pattern-tests is a list of functions, each of which returns true if it
  notices a specific bad pattern. The functions are evaluated and if t,
  check-for-single-bad-pattern returns t. Otherwise, it checks the next
  pattern.
  "
  (labels
    (;; Evaluates segment 'x' on all 'pattern-tests'.
     (bad-pattern-eval (x)
       (loop for test-fn in pattern-tests do
             (when (apply test-fn (list x))
               ; Return early if we found one.
               (return-from bad-pattern-eval t)))
       ; None found, return nil.
       nil)); end of labels definitions.
    ;; Main body.
    (cond
      ; No need to check if f is atomic.
      ((atom f) nil)
      ((bad-pattern-eval f) t)
      (t
        (loop for sub-f in f do
               (when (bad-pattern-eval sub-f)
                 (return-from check-for-single-bad-pattern t)))
        nil))))

(defun exists-bad-pattern? (fstr)
  "Checks whether there exists a bad pattern from the sanity check for the
  given formula for the purposes of filtering parsing results.
  "
  (let ((f (read-from-string fstr))
        preprocd)
    (when (check-for-single-bad-pattern (util:hide-ttt-ops f)
                                        *ulfctp-raw-bad-pattern-tests*)
      (return-from exists-bad-pattern? t))
    (setf preprocd (util:intern-symbols-recursive (preprocess f) *package*))
    (loop for preproc-elem in preprocd do
          (when (check-for-single-bad-pattern preproc-elem
                                              *ulfctp-bad-pattern-tests*)
            (return-from exists-bad-pattern? t)))
    ;; Otherwise return nil.
    nil))

