
(in-package :util)


(defun insert (x lst i)
  "Inserts X to LST at position I.
  Destructive."
  (if (= i 0)
    (push x lst)
    (push x (cdr (nthcdr (1- i) lst))))
  lst)

(defun slice (lst start end)
  "Returns a slice of :ST with given indices.
  START and END are inclusive and exclusive, respectively."
  (declare (type fixnum start end)
           (type list lst))
  (labels
    ((helper
       (cur index acc)
       (declare (type fixnum index)
                (type list cur acc))
       (cond
         ;; Ran out of list.
         ((null cur) acc)
         ;; Index past end.
         ((>= index end) acc)
         ;; Recursive case, in range.
         ((and (>= index start)
               (< index end))
          (helper (cdr cur) (1+ index) (cons (car cur) acc)))
         ;; Recursive case before start.
         (t (helper (cdr cur) (1+ index) acc)))))
    (reverse (helper lst 0 '()))))

(defun remove-nth (n lst)
  "Returns LST without the N-th element."
  (declare (type fixnum n)
           (type list lst))
  (append (subseq lst 0 n) (nthcdr (1+ n) lst)))

(defun split-by-cond (lst cndn)
  "Returns LST with CNDN filtered out followed by LST with only CNDN."
  (declare (type list lst))
  (list (remove-if cndn lst)
        (remove-if-not cndn lst)))

(defun interleave (lst1 lst2)
  "Returns a list where the items alternate between the items of LST1 and LST2."
  (labels
    ;; Helper function, builds the interleaving in reverse.
    ;; Reduces the number of base and recursive cases by swapping l1 and l2
    ;; between recursive calls.
    ((helper (l1 l2 acc)
       (cond
         ((null l1) (append (reverse l2) acc))
         (t (helper l2 (cdr l1) (cons (car l1) acc))))))
    (reverse (helper lst1 lst2 nil))))

(defun pair-up-list (lst)
  "Returns a list where every two consecutive items in LST are paired together.
  i.e. (a b c d) -> ((a b) (c d))
  Assumes that the list is of even length and doesn't contain nil elements."
  (reverse 
    (car 
      ;; ACC is a pair where the first value is the accumulation of the return
      ;; value and the second value is the first value of the next pair or NIL
      ;; if a fresh pair.
      (reduce #'(lambda (acc cur)
                  (let ((acclst (first acc))
                        (firstval (second acc)))
                    (if firstval
                      (list (cons (list firstval cur) acclst)
                            nil)
                      (list acclst cur))))
              lst
              :initial-value (list nil nil)))))

(defun powerset (s)
  "Computes a powerset of set S.
  From https://rosettacode.org/wiki/Power_set#Common_Lisp"
  (if s (mapcan #'(lambda (x) (list (cons (car s) x) x))
                (powerset (cdr s)))
      '(())))

(defun permute (list)
  "Returns a list of all permutations of LIST.
  From https://rosettacode.org/wiki/Permutations#Common_Lisp"
  (if list
    ;; Recursive case.
    (mapcan #'(lambda (x)
                (mapcar #'(lambda (y) (cons x y))
                        (permute (remove x list))))
            list)
    ;; Base case.
    '(()))) 

(defun label-with-num (lst)
  "Labels LST with numbers.
  Similar to 'enumerate' in Python.
  (label-with-num '(a b c)) -> '((0 a) (1 b) (2 c))"
  (labels
    ((helper (acc cur)
       (let ((lst+ (first acc))
             (curidx (second acc)))
         (list (cons (list curidx cur) lst+) (1+ curidx)))))
    ;; strip off the counter and reverse.
    (reverse (first (reduce #'helper lst :initial-value (list nil 0))))))

