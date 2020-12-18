
(in-package :util)

;; Input data format:
;;   List of <boolean>
;; Returns the precision (<number true>/<total number>).
(defun precision (data &key (make-float t))
  (declare (type list data))
  (let* ((num (length (remove-if-not #'(lambda (x) x) data)))
         (den (length data))
         (rat
          (if (= 0 den) nil
            (/ num den))))
    (values
      (if (and make-float rat) (float rat) rat)
      rat
      num
      den)))


;; Input data format:
;;   List of (category . <boolean>)
;; Returns:
;;   List of (category . precision)
(defun group-precisions (data)
  (let (grouped)
    (setq grouped
          ;; Accumulate the association list by the key.
          (reduce #'(lambda (acc new)
                      (let ((cat (car new))
                            (val (cdr new)))
                        (if (not (assoc cat acc :test #'equal))
                          (setq acc (acons cat nil acc)))
                        (rplacd (assoc cat acc :test #'equal)
                                (cons val (cdr (assoc cat acc :test #'equal))))
                        acc))
                      data :initial-value nil))
    (mapcar #'(lambda (x)
                (cons (car x)
                      (multiple-value-list (precision (cdr x)))))
            grouped)))

;; Input data format:
;;   (category . <boolean>)
;; Returns the macro precision -- precision over the full set.
(defun macro-precision (data)
  (precision (mapcar #'cdr data)))

;; Input data format:
;;   List of (category . <boolean>)
;; Returns the micro precision -- average precision over the categories.
(defun micro-precision (data)
  (cl-mathmean (mapcar #'second (group-precisions data))))

;; Cartesian product of n lists.
;;  ((a b) (1 2) (x y))
;;  -> ((a 1 x) (a 1 y) (a 2 x) (a 2 y) (b 1 x) (b 1 y) (b 2 x) (b 2 y))
(defun cartesian-product (choicelst)
  (reduce
    #'(lambda (acc choices)
        (apply #'append
          (mapcar
            #'(lambda (choice)
                (mapcar #'(lambda (prev) (cons choice prev))
                        acc))
            choices)))
    (reverse choicelst)
    :initial-value '(nil)))

