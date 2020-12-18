; Code to interface the ulf2amr code to the ULFCTP parser.
; The ULFCTP parser generates the arcs in reverse, so first unreverse them
; then call the appropriate amr2ulf method.

(in-package :ulf2amr)

(defun remove-single-arg-lists (ulfamr)
  "Flattens out single argument lists since they sometimes get introduced in
  lazy preprocessing.
  "
  (cond
    ((atom ulfamr) ulfamr)
    ((= 1 (length ulfamr)) (remove-single-arg-lists (first ulfamr)))
    (t (mapcar #'remove-single-arg-lists ulfamr))))


(defun reverse-ulfamr-arcs (ulfamr)
  "Recursively reverses the order of outgoing arcs in the given ULFAMR.
  e.g. (x / make.v :arg1 (y / happy.a) :arg0 (z / him.pro))
    -> (x / make.v :arg0 (z / him.pro) :arg1 (y / happy.a))
  "
  (cond
    ((null ulfamr) ulfamr)
    (t (append
         (list (get-var ulfamr)
               '/
               (get-label ulfamr))
         (apply #'append
                (mapcar #'(lambda (pair)
                            (list (first pair) ; arclabel
                                  (reverse-ulfamr-arcs (second pair)))) ; child
                        (reverse (get-arc-pairs ulfamr))))))))


(defun ulfctp-amr2ulf (str &key (method 'arcmap))
  "Takes a string representation of the AMR-ized ULF (possibly with 
  fragments) and maps them back to the basic ULF representation. Then
  stringifies them before returning the value. Fragments are separated
  with a newline.
  "
  (let* 
    ((all-segments (util:read-all-from-string str))
     (pre-segments (mapcar #'remove-single-arg-lists all-segments))
     all-ulfs)
    (when (not (every #'penman-format? pre-segments))
     (error "One of these is not a valid penman expression! ~s~%" pre-segments))
    (setf all-ulfs (mapcar #'(lambda (ulfamr)
                               (amr2ulf (reverse-ulfamr-arcs ulfamr) :method method))
                           pre-segments))
    (util:list-to-string all-ulfs '(#\Newline))))

