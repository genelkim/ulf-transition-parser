;; Utility functions over general s-expressions.

(in-package :util)

;; Recurses through the s-expression, f, and extracts out categories that
;; satisfy catfn.  Subexpressions that satisfy ign-cnd-fn are ignored, so
;; elements satisfying catfn directly within them are ignored.
;;
;; Returns:
;;  A two element list where the first element is f with catfn subexpressions
;;  removed, and the second is a list of catfn subexpressions.
;;
;; Example:
;;  (extract-category '(a (2 b) 3 c)
;;                    #'numberp
;;                    #'(lambda (x) (<= (length x) 2)))
;;  -> '((a (2 b) c) (3))
;;
(defun extract-category (f catfn ign-cnd-fn)
  (if (atom f) (list f '())
    (let* ((split
             (if (funcall ign-cnd-fn f)
               (list f nil)
               (util:split-by-cond f catfn)))
           (no-sent-ops (first split))
           (sent-ops (second split))
           (recursed (mapcar #'(lambda (x)
                                 (extract-category x catfn ign-cnd-fn))
                             no-sent-ops)))
      (list (mapcar #'first recursed)
            (apply #'append (cons sent-ops (mapcar #'second recursed)))))))


;; Returns subexpressions in a tree that are the same as the given symbol and
;; the given equality test, similar to how subst works.
(defun tree-find (tree sym &key (test #'eql))
  (second (extract-category tree
            #'(lambda (x) (funcall test sym x))
            #'nilfn)))


;; Returns subexpressions in tree that satisfy cndfn, similar to how subst-if
;; substitutes expression that satisfy a condition.
(defun tree-find-if (tree cndfn)
  (second (extract-category tree cndfn
            #'nilfn)))

