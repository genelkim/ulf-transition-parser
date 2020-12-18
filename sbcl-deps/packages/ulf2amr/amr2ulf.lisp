;;; The reverse of ulf2amr.lisp.
;;;
;;; Takes an AMR-ized ULF and converts it back to the original ULF.

(in-package :ulf2amr)


(defun arcmap-amr2ulf (amr)
  "Maps an 'arcmap style AMRized ULF back to original ULF style."
  (funcall
    (compose
      #'remove-single-bracket
      #'arcmap-drop-arcs)
    amr))

(defun arcmap-drop-arcs (amr)
  "Drops arcs and removes COMPLEX labels in accordance with 'arcmap style AMRization.
  Removes arcs and COMPLEX labels while keeping all bracketing and retaining the
  input argument order.
   e.g.
      (x / A :arg0 (y / B) :arg1 (z / C) :arg2 (a / D))
      -> (A B C D)
  "
  (let ((label (get-label amr))
        (arc-pairs (get-arc-pairs amr)))
    (cond
      ((and (null arc-pairs) (eq label 'COMPLEX)) nil)
      ((null arc-pairs) label)
      (t (let ((children (mapcar #'(lambda (x) (arcmap-drop-arcs (second x)))
                                 arc-pairs)))
           (if (eq label 'COMPLEX)
             children
             (cons label children)))))))


(defun get-arc-num (arc)
  (let ((arcstr (symbol-name arc)))
    (if (equal "ARG" (subseq arcstr 0 3))
      (parse-integer (subseq arcstr 3))
      nil)))

(defun split-pairs-by-label (arc-pairs label)
  (util:split-by-cond
    arc-pairs
    #'(lambda (pair) (eq label (first pair)))))


(defun argstruct-reorder-arc-pairs (arc-pairs)
; Reorders arc-pairs to retain as much ordering as possible, but forcing all arg0
; to come before arg1, all arg1 before arg2, etc.
; Get earliest index of argX, where X is the highest arg.
; Move all argY, Y < X before that index.
; Repeat for decrementing X.
  (let* ((arc-names (mapcar #'first arc-pairs))
         (max-arg (apply #'max (cons -1 (remove-if #'null (mapcar #'get-arc-num arc-names)))))
         (argmax (intern (format nil "ARG~s" max-arg) :keyword)))
    (when (= max-arg -1)
      (return-from argstruct-reorder-arc-pairs arc-pairs))
    (labels
      ((move-before-max (curpairs m)
         (let ((argm (intern (format nil "ARG~s" m) :keyword))
               mpos maxpos)
           (setf mpos (position-if #'(lambda (x) (eq argm (first x)))
                                   curpairs :from-end t))
           (setf maxpos (position-if #'(lambda (x) (eq argmax (first x)))
                                     curpairs))
           (cond
             ; successful, try again!
             ((and (not (null mpos)) (not (null maxpos)) (< max-arg m))
              (move-before-max
                (util:insert (nth mpos curpairs)
                             (util:remove-nth mpos curpairs)
                             maxpos)
                m))
             ; nothing to be done for m, so go to m-1 if possible.
             ((> m 0) (move-before-max curpairs (1- m)))
             ; just return curpairs if nothing is to be done at all.
             (t curpairs)))))
      ;; top-level call.
      (move-before-max arc-pairs (1- max-arg)))))


(defun argstruct-drop-arcs (amr)
; Removes arcs and COMPLEX labels while keeping all bracketing and enforcing
; arg0/arg1/arg2 ordering.
; e.g.
;   (x / A :arg1 (y / B) :arg2 (z / C) :arg0 (a / D))
;   -> (A D B C)
;   TODO: punctuation before...
  (let ((label (get-label amr))
        (arc-pairs (get-arc-pairs amr)))
    (cond
      ((and (null arc-pairs) (eq label 'COMPLEX)) nil)
      ((null arc-pairs) label)
      (t (let ((children (mapcar #'(lambda (x) (argstruct-drop-arcs (second x)))
                                 (argstruct-reorder-arc-pairs arc-pairs))))
           (if (eq label 'COMPLEX)
             children
             (cons label children)))))))

; PLUR
; (v2 / friend-of.n
;     :plur (v1 / plur)
;     :arg0 (v3 / him.pro))
; vvvvvv
; (v1 / plur
;     :arg0 (v2 / friend-of.n
;               :arg0 (v3 / him.pro)))
(defun reverse-plur-post-ttt (amr)
  (let* ((top-label (get-label amr))
         (top-var (get-var amr))
         (arc-pairs (get-arc-pairs amr))
         (split-pairs (split-pairs-by-label arc-pairs ':plur))
         (plur-pairs (second split-pairs))
         (noplur-pairs (first split-pairs)))
    (cond
      ((null plur-pairs) amr)
      (t
        (let* ((plur-node (second (first plur-pairs)))
               (plur-var (get-var plur-node))
               (plur-label (get-label plur-node)))
          (assert (= (length plur-pairs) 1) (plur-pairs)
                  "plur-pairs: ~s" plur-pairs)
          (assert (null (get-arc-pairs plur-node)))
          (list plur-var '/ plur-label
                ':arg0
                (append (list top-var '/ top-label)
                        (apply #'append noplur-pairs))))))))

; ASPECT
; (v2 / see.v
;     :aspect (v1 / prog)
;     :tense (v3 / past)
;     :arg1 (v4 / him.pro))
; vvvv
; (v1 / prog
;     :arg0 (v2 / see.v
;               :arg1 (v4 / him.pro))
;     :tense (v3 / past))
(defun reverse-aspect-post-ttt (amr)
  (let* ((top-label (get-label amr))
         (top-var (get-var amr))
         (arc-pairs (get-arc-pairs amr))
         (split-pairs (split-pairs-by-label arc-pairs ':aspect))
         (asp-pairs (second split-pairs))
         (noasp-pairs (first split-pairs))
         (tense-split-pairs (split-pairs-by-label noasp-pairs ':tense))
         (tense-flatarcs (apply #'append (second tense-split-pairs)))
         (notense-flatarcs (apply #'append (first tense-split-pairs))))
    (cond
      ((null asp-pairs) amr)
      ;; Single aspect (perf or prog).
      ((= (length asp-pairs) 1)
        (let* ((asp-node (second (first asp-pairs)))
               (asp-var (get-var asp-node))
               (asp-label (get-label asp-node))
               res)
          (assert (= (length asp-pairs) 1) (asp-pairs) "asp-pairs: ~s~%" asp-pairs)
          (assert (null (get-arc-pairs asp-node)))
          (append
            (list asp-var '/ asp-label
                  ':arg0
                  (append (list top-var '/ top-label) notense-flatarcs))
            tense-flatarcs)))
      ;; Both perf and prog are present. Perf comes before prog.
      ((= (length asp-pairs) 2)
       (let* ((perf-node (first (remove-if-not
                                  #'(lambda (node) (eq (get-label node) 'perf))
                                  (mapcar #'second asp-pairs))))
              (prog-node (first (remove-if-not
                                  #'(lambda (node) (eq (get-label node) 'prog))
                                  (mapcar #'second asp-pairs)))))
         (append
           (list (get-var perf-node) '/ 'perf
                 ':arg0
                 (list (get-var prog-node) '/ 'prog
                       ':arg0
                       (append (list top-var '/ top-label) notense-flatarcs)))
           tense-flatarcs)))
      (t (error "Too many aspect relations! amr: ~s~%" amr)))))

; AUX
; (v2 / see.v
;     :aux (v1 / can.aux-v)
;     :tense (v3 / past)
;     :arg1 (v4 / him.pro))
; vvvv
; (v1 / can.aux-v
;     :arg0 (v2 / see.v
;               :arg1 (v4 / him.pro))
;     :tense (v3 / past))
(defun reverse-aux-post-ttt (amr)
  (let* ((top-label (get-label amr))
         (top-var (get-var amr))
         (arc-pairs (get-arc-pairs amr))
         (split-pairs (split-pairs-by-label arc-pairs ':aux))
         (aux-pairs (second split-pairs))
         (noaux-pairs (first split-pairs))
         (tense-split-pairs (split-pairs-by-label noaux-pairs ':tense))
         (tense-pairs (second tense-split-pairs))
         (notense-pairs (first tense-split-pairs)))
    (cond
      ((null aux-pairs) amr)
      (t
        (let* ((aux-node (second (first aux-pairs)))
               (aux-var (get-var aux-node))
               (aux-label (get-label aux-node))
               res)
          (assert (= (length aux-pairs) 1))
          (assert (null (get-arc-pairs aux-node)))
          (setf res
                (list aux-var '/ aux-label
                      ':arg0
                      (append (list top-var '/ top-label)
                              (apply #'append notense-pairs))))
          (append res
                  (apply #'append tense-pairs)))))))

; Tense
; (v3 / see.v
;     :tense (v2 / past)
;     :arg1 (v3 / him.pro))
; vvvvv
; (complex / complex
;     :instance (v2 / past
;                   :arg0 (v3 / see.v))
;     :arg1 (v3 / him.pro))
(defun reverse-tense-post-ttt (amr)
  (let* ((top-label (get-label amr))
         (top-var (get-var amr))
         (arc-pairs (get-arc-pairs amr))
         (split-pairs (split-pairs-by-label arc-pairs ':tense))
         (tense-pairs (second split-pairs))
         (notense-pairs (first split-pairs))
         tense-node tense-var tense-label inst-child)
    ;; Return early if there are not tense arcs.
    (when (null tense-pairs)
      (return-from reverse-tense-post-ttt amr))
    ;; Extract tense info.
    (setf tense-node (second (first tense-pairs)))
    (setf tense-var (get-var tense-node))
    (setf tense-label (get-label tense-node))
    (assert (= (length tense-pairs) 1))
    (assert (null (get-arc-pairs tense-node)))
    ;; Build tense-lifted version.
    (if (not (eq top-label 'complex))
      ;; Top label is not COMPLEX, just add tense to it.
      (append
        (list 'aacomplexaa '/ 'complex
              ':instance (list tense-var '/ tense-label
                               ':arg0
                               (list top-var '/ top-label)))
        (apply #'append notense-pairs))
      ;; Top label is complex, so add tense to the child of :INSTANCE arc.
      (progn
        (setf inst-child
              (second (first (remove-if-not #'(lambda (pair)
                                                (eq (first pair) ':instance))
                                            arc-pairs))))
        (append
          (list 'bbcomplexbb '/ 'complex
                ':instance (list tense-var '/ tense-label
                                 ':arg0 inst-child))
          (apply #'append
                 (remove-if #'(lambda (pair) (member (first pair)
                                                     '(:instance :tense)))
                            arc-pairs)))))))

; PASV
; (v2 / find.v
;     :pasv (v1 / pasv)
;     :arg1 (v3 / him.pro))
; vvvvvv
; (COMPLEX / complex
;     :instance (v1 / pasv
;                   :arg0 (v2 / find.v))
;     :arg1 (v3 / him.pro)))
(defun reverse-pasv-post-ttt (amr)
  (let* ((top-label (get-label amr))
         (top-var (get-var amr))
         (arc-pairs (get-arc-pairs amr))
         (split-pairs (split-pairs-by-label arc-pairs ':pasv))
         (pasv-pairs (second split-pairs))
         (nopasv-pairs (first split-pairs)))
    (cond
      ((null pasv-pairs) amr)
      (t
        (let* ((pasv-node (second (first pasv-pairs)))
               (pasv-var (get-var pasv-node))
               (pasv-label (get-label pasv-node)))
          (assert (= (length pasv-pairs) 1))
          (assert (null (get-arc-pairs pasv-node)))
          (append
            (list 'yycomplexyy '/ 'complex
                  ':instance (list pasv-var '/ pasv-label
                                   ':arg0
                                   (list top-var '/ top-label)))
            (apply #'append nopasv-pairs)))))))

(defun reverse-argstruct-post-ttt (ulfamr)
  "Reverses the 'argstruct method TTT post-processing on ULFAMR, an AMRized ULF."
  (let ((single-step
          (util:compose #'reverse-tense-post-ttt
                        #'reverse-aspect-post-ttt
                        #'reverse-aux-post-ttt
                        #'reverse-pasv-post-ttt
                        #'reverse-plur-post-ttt)))
    (cond
      ((null ulfamr) ulfamr)
      ((atom ulfamr) ulfamr)
      (t (funcall single-step
                  (mapcar #'reverse-argstruct-post-ttt ulfamr))))))


(defun reverse-lexop-flat-arcmap-post-ttt (ulfamr)
  "Reverses the 'lexop-flat-arcmap method post-processing on ULFAMR, an AMRized ULF."
  (let ((single-step
          (util:compose #'reverse-tense-post-ttt
                        #'reverse-aspect-post-ttt
                        #'reverse-pasv-post-ttt
                        #'reverse-plur-post-ttt)))
    (cond
      ((null ulfamr) ulfamr)
      ((atom ulfamr) ulfamr)
      (t (funcall single-step
                  (mapcar #'reverse-lexop-flat-arcmap-post-ttt ulfamr))))))

(defun complex-chain-end (amr)
;; Returns the symbol at the end of the 'complex-:instance chain.
  (let* ((top-label (get-label amr))
         (arc-pairs (get-arc-pairs amr))
         (instance-pair (find-if #'(lambda (x) (eq ':instance (first x))) arc-pairs)))
    (cond
      ((eq top-label 'complex) (complex-chain-end (second instance-pair)))
      (t top-label))))

(defun verb-amr? (amr)
  ; Top label could be a verb, or COMPLEX whose "instance"-chain results in
  ; a verb, aux, aspect, pasv, or tense operator.
  (let* ((top-label (complex-chain-end amr)))
    (or (lex-verb? top-label)
        (aux? top-label)
        (lex-tense? top-label)
        (eq top-label 'pasv))))

(defun coord-amr? (amr)
  ; Top label could be a verb, or COMPLEX whose "instance"-chain results in
  ; a coordinator operator.
  (let* ((top-label (complex-chain-end amr)))
    (lex-coord? top-label)))

(defun lift-arg0 (amr)
; Lifts the :arg0 arcs of verbs and coordinators.
; e.g.
; (v1 / see.v
;   :arg1 (v2 / him.pro)
;   :arg0 (v3 / i.pro))
; VVVVVV
; (v3 / i.pro
;     :arg0 (v1 / see.v
;               :arg1 him.pro))
  (let* ((top-label (get-label amr))
         (top-var (get-var amr))
         (arc-pairs (get-arc-pairs amr))
         (arg0-pair (find-if #'(lambda (x) (eq ':arg0 (first x))) arc-pairs)))
    (cond
      ; No arg0, just return
      ((null arg0-pair) amr)
      ; Not a verb or coordinator.
      ; Top label could be a verb, or COMPLEX whose "instance"-chain results in
      ; a verb, aux, aspect, or tense operator.
      ((and (not (verb-amr? amr)) (not (coord-amr? amr))) amr)
      ; The arg0 is itself a verb and top-level verb
      ; -- just return (this will be a child of a pasv/aspect/aux)
      ((and (verb-amr? amr) (verb-amr? (second arg0-pair))) amr)
      (t
        ;; Perform lifting, only take the first arg0. Leave the rest.
        (let* ((arg0-node (second arg0-pair))
               (other-pairs
                 (util:remove-nth
                   (position-if #'(lambda (x) (eq ':arg0 (first x)))
                                arc-pairs)
                   arc-pairs)))
          (list 'zzcomplexzz '/ 'complex
                ':instance arg0-node
                ':arg0
                (append (list top-var '/ top-label)
                        (apply #'append other-pairs))))))))

(defun lift-all-arg0 (amr)
  (cond
    ((null amr) amr)
    ((atom amr) amr)
    (t (lift-arg0 (mapcar #'lift-all-arg0 amr)))))

(defun remove-single-bracket (ulf)
  (cond
    ((atom ulf) ulf)
    ((and (listp ulf) (= (length ulf) 1))
     (remove-single-bracket (first ulf)))
    (t (mapcar #'remove-single-bracket ulf))))


(defun argstruct-amr2ulf (amr)
; First, move the arg0 arc of verbs to prefix position.
; Then, reverse the argstruct post-processing (tense, aspect, plur, etc.).
; Finally, drop arc labels while enforcing arg0/arg1/arg2 ordering to get ULF.
  (funcall
    (util:compose
      #'remove-single-bracket
      #'argstruct-drop-arcs
      #'reverse-argstruct-post-ttt
      #'lift-all-arg0)
    amr))


(defun lexop-flat-arcmap-amr2ulf (amr)
  "Maps a 'lexop-flat-arcmap style AMRized ULF, back to original ULF style.
  This is done in two steps:
  1. reverse the 'lexop-flat-arcmap type post-processing
  2. drop all arcs and retain brackets
  "
  (funcall
    (util:compose
      #'remove-single-bracket
      #'arcmap-drop-arcs
      #'reverse-lexop-flat-arcmap-post-ttt)
    amr))

(defun unstringify-amrulf (amr)
  "Maps strings in the AMRULF into symbols, since Lisp can handle these properly.
  NB: The strings are designed to fit into Penman format which can only introduce
  spaces into symbols through strings."
  (cond
    ((stringp amr)
     ;; Remove pipes from string if there.
     (if (and (= (position #\| amr) 0)
              (= (position #\| amr :from-end t) (1- (length amr))))
       (intern (subseq amr 1 (1- (length amr))))
       (intern amr)))
    ((atom amr) amr)
    (t (mapcar #'unstringify-amrulf amr))))


(defun amr2ulf (amr &key (method 'arcmap))
;~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
; Converts AMR-syntax ULF formula back to original ULF format.
;
; method is a symbol indicating the method that the AMR was generated.
;   'arcmap is a contituency to arc translation. e.g.
;     (A B C D) -> (x / A :arg0 (y / B) :arg1 (z / C) :arg2 (a / D))
;     where A may be expanded into COMPLEX+:instance if it's not an atom.
;     The amr2ulf here is simple. It simply removes the :arc relations and
;     complex while keeping bracketing.
;   'argstruct converts ULFs into the most EL-smantically coherent form, where
;     the operator comes first, and lexical markings are given special relations.
;     e.g. (i.pro (pres know.v))
;       -> (x / know.v :arg0 (y / i.pro) :tense (z / pres))
;     the amr2ulf is a bit more complicate here, since the :arg0 subject must be lifted
;     appropriately for verbs and the tenses must be rescoped correctly.
  (funcall
    (util:compose
      ;; Choose general amr2ulf function based on AMRization method.
      (case method
        (arcmap #'arcmap-amr2ulf)
        (argstruct #'argstruct-amr2ulf)
        (lexop-flat-arcmap #'lexop-flat-arcmap-amr2ulf)
        (otherwise (error "Unknown amr2ulf method: ~s~%" method)))
      ;; First map strings to symbols.
      #'unstringify-amrulf)
    amr))

