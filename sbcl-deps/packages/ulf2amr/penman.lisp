;; Lisp functions for handling s-expressions representing AMRs.

(in-package :ulf2amr)

(defun shallow-penman-format? (expr)
  "Checks that the given s-expression is a valid penman format at the top
  level.  This means that it is a list of length 3 + 2n, n>=0 and the second
  element is the '/ symbol or it is an atomic element.
  "
  (or (atom expr)
      (and (listp expr)
           (>= (length expr) 3)
           (= 0 (rem (- (length expr) 3) 2)))))

(defun penman-format? (expr)
  "Checks that the given s-expression is a valid penman format, recursively.
  "
  (and (shallow-penman-format? expr)
       (or (atom expr)
           (every #'penman-format? expr))))


;; '(x / run.v :arg0 ...) -> 'run.v
;; 'x -> nil
(defun get-label (expr)
  (if (listp expr)
    (third expr)
    nil))

;; '(x / run.v) -> 'x
;; 'x -> 'x
(defun get-var (expr)
  (if (listp expr)
    (first expr)
    expr))

;; Returns the arc label and child pairs.
;; '(x / run.v :arg0 A :arg1 B :instance C)
;; -> '((:arg0 A) (:arg1 B) (:instance C))
(defun get-arc-pairs (expr)
  (util:pair-up-list (subseq expr 3)))

;; Returns two lists of arc label and child pairs. One with a given arc label
;; and one without.
;(defun split-pairs-by-label (arc-pairs label)
;  (list 
;    (remove-if-not #'(lambda (pair) (eq label (first pair)))
;                   arc-pairs)
;    (remove-if #'(lambda (pair (eq label (first pair))))
;               arc-pairs)))


;; Returns whether two penman format trees are equal.
;; NB: since these are trees, we don't need to handle re-entrancy which 
;; requires keeping track of penman variable references.
(defun penman-tree-eq (tree1 tree2)
  (labels
    ((arc-pair-eq (ap1 ap2)
       ;; Returns whether two list of arc pairs are equal.
       ;; For each pair in ap1, find all in ap2 with the same arc label and
       ;; check if any are equal. Then recurse with the remainder.
       (when (and (null ap1) (null ap2)) (return-from arc-pair-eq t))
       (when (or (null ap1) (null ap2)) (return-from arc-pair-eq nil))
       (when (not (= (length ap1) (length ap2))) (return-from arc-pair-eq nil))
       (let ((cur-arclabel (first (first ap1)))
             (cur-body (second (first ap1)))
             same-labels cur-same)
         (setf same-labels
               (remove-if-not #'(lambda (arcpair) (equal cur-arclabel (first arcpair)))
                              ap2))
         ;; Get the element in same-labels that is equal to the current ap1 arc.
         (setf cur-same
               (reduce #'(lambda (acc cur)
                           (or acc (if (penman-tree-eq cur-body (second cur)) cur)))
                       same-labels :initial-value nil))
         ;; Recurse into rest of arc pairs if a match was found here.
         (and cur-same
              (arc-pair-eq (cdr ap1) 
                           (remove cur-same ap2 :test #'equal :count 1)))))
     ) ; end of labels definitions.
  ;; Jointly recurse in the two trees and make sure the node and arc labels
  ;; match.
  (cond
    ((and (null tree1) (null tree2)) t)
    ((or (null tree1) (null tree2)) nil)
    ((or (atom tree1) (atom tree2))
     (error "Input to penman-tree-eq must be penman formatted trees. A graph was given."))
    (t
      (let ((node1 (get-label tree1))
            (node2 (get-label tree2))
            (arc-pairs1 (get-arc-pairs tree1))
            (arc-pairs2 (get-arc-pairs tree2)))
        (and (equal node1 node2)
             (arc-pair-eq arc-pairs1 arc-pairs2)))))))

;; Decodes a penman format s-expression to triples.
;; '(x / run.v :arg0 (y / him.pro))
;; -> '((x instance run.v)
;;      (y instance him.pro)
;;      (x :arg0 y))
;; Order is not guaranteed in any way.
(defun decode-to-triples (penman)
  (labels
    (;; Recursive helper function for tail-recursion.
     (rec-helper (acc cur)
       (cond
         ((null cur) acc)
         ((atom cur) acc)
         (t
           (let ((inst-triple (list (get-var cur) 'instance (get-label cur)))
                 (arc-pairs (get-arc-pairs cur))
                 (cur-var (first cur)))
             (reduce #'(lambda (inner-acc inner-cur)
                         (let ((arclabel (first inner-cur))
                               (arcchild-var (get-var (second inner-cur))))
                           ;; Recurse into the arc child after adding the current
                           ;; arc relation.
                           (rec-helper (cons (list cur-var arclabel arcchild-var) 
                                             inner-acc) 
                                       (second inner-cur))))
                     arc-pairs :initial-value (cons inst-triple acc))))))
     ) ; end of labels definitions.
    ;; Main body.
    (rec-helper nil penman)))

