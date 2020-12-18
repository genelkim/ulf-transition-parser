;; Functions to help with regex, such as variants on cl-ppcre functions.

(in-package :util)

;; Give cl-ppcre a nickname.
(add-nickname "CL-PPCRE" "RE")

;; Same as cl-ppcre:all-matches, but handles overlapping matches.
(defun overlap-regex-matches (regex target-string &key
                                       (start 0) (end (length target-string)))
  (labels
    ((helper (curstart acc)
       (multiple-value-bind (ms me) (re:scan regex target-string
                                             :start curstart :end end)
         (if (null ms)
           ;; Base case.
           (reverse acc)
           ;; Recursive case.
           (helper (1+ ms) (cons me (cons ms acc)))))))
    (helper start nil)))


;; Same as cl-ppcre:all-matches-as-strings, but handles overlapping matches.
(defun overlap-regex-matches-as-strings (regex target-string &key
                                               (start 0) (end (length target-string)))
  (let*
    ((matches (overlap-regex-matches regex target-string :start start :end end))
     (rawres (reduce
               #'(lambda (acc cur)
                   (let ((strlst (car acc))
                         (start (cdr acc)))
                     (if start
                       ;; cur must be the end, so we find the substring.
                       (cons
                         (cons (subseq target-string start cur) strlst)
                         nil)
                       ;; cur must be start, so add that to acc
                       (cons strlst cur))))
               matches
               :initial-value '(nil . nil))))
    (reverse (car rawres))))

;; Maps an alist with regex keys to an alist with cl-ppcre scanners as keys.
(defun regex-alist-to-scanner-alist (ralist &key (case-insensitive-mode t))
  (mapcar
    #'(lambda (contr-pair)
        (cons (re:create-scanner (car contr-pair)
                                 :case-insensitive-mode case-insensitive-mode)
              (cdr contr-pair)))
    ralist))

