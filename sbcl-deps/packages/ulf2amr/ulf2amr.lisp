;~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
; Converts a formula in Unscoped Episodic Logical Form (ULF) to
; Abstract Meaning Representation (AMR) syntax.  Does not convert robustly
; to conserve semantics, simply convert the syntax so that we can use the
; smatch script to evaluate similarity of ULFs using EL-smatch.
; ;
; This implementation goes depth first, so that the types can be propagated
; upward.

(in-package :ulf2amr)

;; TODO: make a simpler version that doesn't add mod.

;; Returns true if one of xs is a member of lst,
;; otherwise nil.
(defun one-of-member (xs lst)
  (or (remove-if #'null (mapcar #'(lambda (x) (member x lst)) xs))))

;; Generate variable.
;; Simply iterates up integers, until the given base symbol with the
;; integer appended is not in the symbol list.
;; Returns the new variable, integer appended, and the updated symbol
;; list in a tuple.
(defun generate-variable
  (base start symbol-set)
  (let ((var (read-from-string (format nil "~s~s" base start))))
    ;(format t "base ~s~%start ~s~%new symbol ~s~%" base start var)
    (if (member var symbol-set)
      ;; Already in symbol set, so recurse with incremented integer.
      ; TODO: return a fully incremented value returned from recursive call.
      (generate-variable base (+ start 1) symbol-set)
      ;; Not in symbol set, add it and return.
      (list var (+ 1 start) (cons var symbol-set)))))


(defun orig-lf-components (type-elems lf-elems node-type ulf el-vars all-vars start)
;````````````````````````````````
; Generates the lf-components for the original output format.
; This is the original rendition of argstruct lf-components which has many bugs.
; This is left in the code for experimenting with old versions.
;
; Parameters:
;   type-elems - types of the current list of elements
;   lf-elems - lfs of the current list of elements
;   node-type - symbol for the type of the current node operation (e.g. pred1, conj, lambda, etc.)
;   ulf - current ULF state
;   el-vars - all existing semantic variables
;   all-vars - all existing syntactic variables
;   start - current variable starting index
;
; Returns:
; a list of the components of the current logical expression
;   1. the operator expression
;   2. the list of arguments in order
;   3. the resulting constituent type
;   4. 'ordered or 'unordered (referring to the arguments)
;   5. the new starting index for variables
  (cond
    ;; Quantifier. : no quantifiers in ULF
    ;((member node-type '(quant bad-quant))
    ; (progn
    ;   (list (first lf-elems)
    ;         (subseq lf-elems 1)
    ;         'pronoun
    ;         'ordered start)))
    ;; Determiner.
    ((and (= (list-length type-elems) 2)
          (equal (first type-elems) 'determiner)
          (member (second type-elems) '(predicate noun)))
     (list (first lf-elems) (cdr lf-elems) 'pronoun 'ordered start))

    ;; SUB
    ;((and (= (list-length type-elems) 3)
    ;      (equal (first type-elems) 'sub))
    ; (list (first lf-elems) (cdr lf-elems)
    ;       ;; Type depends on how the substitution composes.
    ;       ;; TODO: run substitutation and analyze the child.
    ;       ...

    ;; N+PREDS
    ((and (> (list-length type-elems) 1)
          (equal (first type-elems) 'n+preds)
          (member (second type-elems) '(predicate noun)))
     (list (first lf-elems) (cdr lf-elems) 'predicate 'ordered start))

    ;; NP+PREDS
    ((and (> (list-length type-elems) 1)
          (equal (first type-elems) 'np+preds)
          (member (second type-elems) '(pronoun skolem)))
     (list (first lf-elems) (cdr lf-elems) 'pronoun 'ordered start))


    ;; Coordinating conjunctions.
    ((equal node-type 'conj)
     (progn
       (if *ulf2amr-debug*
         (progn
           (format t "in conj node type~%")
           (format t "lf-elems ~s~%" lf-elems)
           (format t "one-of-member ~s~%~%" (one-of-member '(or or.cc) (mapcar #'(lambda (x) (if (consp x) (third x) x)) lf-elems)))))

       (if (one-of-member '(or or.cc) (mapcar #'(lambda (x) (if (consp x) (third x) x)) lf-elems))
         ;; disjunction
         (let ((filtered
                 (remove-if
                   #'(lambda (x)
                       (and (consp x)
                            (< 2 (list-length x))
                            (member (third x) '(or or.cc))))
                   lf-elems)))
           (if *ulf2amr-debug*
             (format t "finished disjunction filtering~%~%~%"))
           (list 'or filtered 'sentence 'unordered start))
         ;; conjunction
         (let ((filtered
                 (remove-if
                   #'(lambda (x)
                       (and (consp x)
                            (< 2 (list-length x))
                            (member (third x) '(and and.cc))))
                   lf-elems)))
           (list 'and filtered 'sentence 'unordered start)))))

    ;; Lambda.
    ((equal node-type 'lambda)
     (progn
       (list (first lf-elems) (subseq lf-elems 1)
             'predicate 'ordered start)))

    ;; Gensym.
    ;; Break down gensym by type.
    ((equal node-type 'gensym)
     (let ((exact-node-type
             (case (first type-elems)
               (gen-wff 'sentence)
               ((gen-pred-vp gen-pred-pp gen-pred-any) 'predicate)
               (gen-modif 'modifier)
               (gen-term 'term)
               (gen-det 'sentence)
               (gen-conj 'sentence)
               (gen-generic 'unknown))))
       (list (first lf-elems) (subseq lf-elems 1)
             exact-node-type 'ordered start)))

    ;; Colon-keyword.
    ;; Special handling for lambda, otherwise recurse with cdr
    ;; as the single argument.
    ;;
    ;; Also special handling of :r.  Instead of cdr, use the next
    ;; element.  :r indicates attachment ambiguity rather than
    ;; lf substructure ambiguity.
    ;;
    ;; The AMR representation was already constructed for each
    ;; lf element separately, so recreate it with it all as 1
    ;; argument.
    ((equal node-type 'colon-key)
     (cond
       ;; Lambda
       ((equal (first ulf) '=l)
        (progn
          (list (first lf-elems) (subseq lf-elems 1) 'predicate 'ordered start)))

       ;; Role ambiguity (:r)
       ((equal (first ulf) '=r)
        (progn
          (if (not (equal 2 (list-length ulf)))
            (if *ulf2amr-debug*
              (format t "ROLE AMBIGUITY OF UNKNOWN LENGTH ~s~%~%~%" ulf)))
         (let* ((arg (helper (second ulf) el-vars all-vars start))
                (arg-lf (first arg))
                (arg-type (second arg)))
           (setf start (third arg))
           ;; Predicate
           (list (first lf-elems) (list arg-lf)
                 arg-type 'ordered start))))

       ;; Default
       (t
         (progn
           (if *ulf2amr-debug*
             (format t "Colon-keyword recursing (cdr ulf): ~s~%~%~%" ulf))
           (let* ((arg (helper (cdr ulf) el-vars all-vars start))
                  (arg-lf (first arg))
                  (arg-type (second arg)))
             (setf start (third arg))
             ;; Predicate
             (list (first lf-elems) (list arg-lf)
                   arg-type 'ordered start))))))

    ;; Default case.
    ;; Differentiate between predicate modification, curried
    ;; predicate and basic predicate application.
    (t
      (cond
        ;; Adverbival predicate modifier:
        ;;  (adv-a ...), (attr ...)
        ((and (equal (list-length type-elems) 2)
              (equal (first type-elems) 'adv-pred-mod))
         (list (first lf-elems) (cdr lf-elems) 'adverb 'ordered start))

        ;; Adverbial sentence modifier:
        ;;  (adv-e, adv-p, adv-f, ...)
        ((and (equal (list-length type-elems) 2)
              (equal (first type-elems) 'adv-sent-mod))
         (list (first lf-elems) (cdr lf-elems) 'sent-op 'ordered start))

        ;; Verb Predicate modification.
        ;;   (quickly.adv run.v)
        ;;   (quickly.adv (...))
        ((and (equal (list-length type-elems) 2)
              (equal (first type-elems) 'adverb)
              (or (equal (second type-elems) 'predicate)
                  (member (second type-elems) '(verb adjective noun))))
         (list (first lf-elems) (cdr lf-elems) 'predicate 'ordered start))


        ;; Noun Predicate modification.
        ;; Adjective modification of noun predicate:
        ;;   (pickling.a juice.n)
        ;; or pluralization of noun prediacte.
        ;;   (plur thing.n)
        ((and (equal (list-length type-elems) 2)
              (or (equal (first type-elems) 'adjective)
                  (equal (first type-elems) 'pred-mod))
              (equal (second type-elems) 'noun))
         (list (first lf-elems)
               (subseq lf-elems 1)
               'predicate 'ordered start))

        ;; Transitive predicate modifiers
        ;; (pasv x.v), ....
        ((and (equal (list-length type-elems) 2)
              (equal (first type-elems) 'pred-mod))
         (list (first lf-elems)
               (subseq lf-elems 1)
               (second type-elems) 'ordered start))

        ;; Prefix predicate.
        ((and (> (list-length type-elems) 1)
              (member (first type-elems) '(predicate verb)))
         (list (first lf-elems) (cdr lf-elems) 'predicate 'ordered start))


        ;; Modifier creation.
        ((and (equal (list-length type-elems) 2)
              (equal (first type-elems) 'mod-creator)
              (member (second type-elems) '(verb noun adjective predicate)))
         (list (first lf-elems) (cdr lf-elems) 'pred-mod 'ordered start))

        ;; Sentence operation.
        ;;   (pres (John see Mary))
        ;;   ((adv-e (in-loc forest)) ...)
        ((and (equal (list-length type-elems) 2)
              (equal type-elems '(sent-op sentence)))
         (list (first lf-elems) (cdr lf-elems) 'sentence 'ordered start))

        ;; Sentence nominalization.
        ((and (equal (list-length type-elems) 2)
              (equal type-elems '(sent-nom sentence)))
         (list (first lf-elems) (cdr lf-elems) 'pronoun 'ordered start))

        ;; Tense.
        ((and (equal (list-length type-elems) 2)
              (equal (first type-elems) 'tense))
         (list (first lf-elems) (cdr lf-elems) (second type-elems) 'ordered start))

        ;; quote-o
        ((and (equal (list-length type-elems) 2)
              (equal (first type-elems) 'quote-o))
         (list (first lf-elems) (cdr lf-elems) 'pronoun 'ordered start))

        ;; bracket-op
        ((and (> (list-length type-elems) 0)
              (equal (first type-elems) 'bracket-op))
         (list (first lf-elems) (Cdr lf-elems) 'pronoun 'ordered start))

        ;; Predicate nominalization.
        ;; Some words may be predicates, but not marked yet,
        ;; so we can be flexible with predicate nominalizations.
        ;; Allow anything that is length 2 with a predicate
        ;; nominalization to start.
        ((and (equal (list-length type-elems) 2)
              (equal (first type-elems) 'pred-nom))
         (list (first lf-elems) (cdr lf-elems) 'pronoun 'ordered start))

        ;; Preposition arguments.
        ((and (> (list-length type-elems) 1)
              (equal (first type-elems) 'parg))
         (list (first lf-elems) (cdr lf-elems) 'pronoun 'ordered start))

        ;; Preposition.
        ((and (equal (list-length type-elems) 2)
              (equal (first type-elems) 'preposition)
              (member (second type-elems) '(pronoun noun var)))
         (list (first lf-elems)
               (cdr lf-elems)
               'predicate 'ordered start))

        ;; Sent to pred.
        ((and (equal (list-length type-elems) 2)
              (equal (first type-elems) 'sent-to-pred)
              (equal (second type-elems) 'sentence))
         (list (first lf-elems) (cdr lf-elems) 'predicate 'ordered start))

        ;; Other sent to pred (bad name).
        ((and (equal (list-length type-elems) 3)
              (equal (first type-elems) 'sent-to-pred))
         (list (first lf-elems) (cdr lf-elems) 'sentence 'ordered start))

        ;; Set forming operator.
        ((and (< 1 (list-length type-elems))
              (equal (first type-elems) 'set-form)
              (reduce #'(lambda (x y) (and x (equal y 'pronoun)))
                      (cdr type-elems)))
         (list (first lf-elems) (cdr lf-elems) 'pronoun 'unordered start))

        ;; SUCH-AS operator.
        ;; Maps a pronoun and predicate into a new predicate.
        ((and (equal 3 (list-length type-elems))
              (equal (first type-elems) 'such-as))
         (list (first lf-elems) (cdr lf-elems) 'predicate 'ordered start))

        ;;;;;;;;;;;;;;;;;;;;;;;;;;;
        ;;; Below are failure mode analyses (type don't correctly
        ;;; compose but we can guess based on restrictive types).
        ;;;;;;;;;;;;;;;;;;;;;;;;;;;

        ;; Determiner/NP+PREDS failure modes
        ((member (first type-elems) '(determiner np+preds))
         (list (first lf-elems) (cdr lf-elems) 'pronoun 'ordered start))

        ;; N+PREDS failure mode
        ((member (first type-elems) '(n+preds))
         (list (first lf-elems) (cdr lf-elems) 'noun 'ordered start))

        ;; SUB failure mode
        ;((member (first type-elems) '(sub))
        ; (list (first lf-elems) (cdr lf-elems)
               ;; TODO recursive analysis.

        ;; Preposition.
        ;; Preposition + pronoun -> predicate
        ((and (= (list-length type-elems) 2)
              (equal (first type-elems) 'preposition)
              (equal (second type-elems) 'pronoun))
         (list (first lf-elems) (cdr lf-elems) 'predicate 'ordered start))

        ;; Comparison
        ((and (equal (list-length type-elems) 3)
              (equal (first type-elems) 'comp-op))
         (list (first lf-elems) (cdr lf-elems) 'sentence 'ordered start))

        ;; Generic multi-argument predicates.
        ;; Predicate can be an ambiguous role verb.
        ((and (< 2 (list-length type-elems))
              (or (equal (second type-elems) 'predicate)
                  (equal (second type-elems) 'verb))
              (not (equal (first type-elems) 'predicate)))
         (list (second lf-elems)
               (append (subseq lf-elems 0 1)
                       (subseq lf-elems 2))
               'sentence 'ordered start))

         ;; Generic single argument predicates.
         ;; Predicate can be an ambiguous role verb,
         ;; adjective, or noun.
        ((and (equal 2 (list-length type-elems))
              (or (equal (second type-elems) 'predicate)
                  (equal (second type-elems) 'verb)
                  (equal (second type-elems) 'adjective))
              (not (member (first type-elems) '(predicate sent-op))))
         (list (second lf-elems)
               (subseq lf-elems 0 1)
               'sentence 'ordered start))

        ;; Single argument noun predicates.
        ;; This can only operate on variables or other nouns.
        ((and (equal 2 (list-length type-elems))
              (equal (second type-elems) 'noun)
              (member (first type-elems) '(noun var)))
         (list (second lf-elems)
               (subseq lf-elems 0 1)
               'sentence 'ordered start))

        ;; Partial predicate application.
        ((and (equal 2 (list-length type-elems))
              (or (equal (first type-elems) 'predicate)
                  (equal (first type-elems) 'verb)
                  (equal (first type-elems) 'adjective)
                  (equal (first type-elems) 'noun))
              (not (member (second type-elems) '(predicate sent-op))))
         (list (first lf-elems)
               (subseq lf-elems 1)
               'predicate 'ordered start))

        ;; Adverb on a sentence.
        ;; This is not valid ULF, but handled for robustness.
        ;; (ususally.adv <sentence>)
        ((and (equal (list-length type-elems) 2)
              (equal (first type-elems) 'adverb)
              (equal (second type-elems) 'sentence))
         (list (first lf-elems)
               (cdr lf-elems)
               'sentence 'ordered start))

        ;; Sentence operation when there is no sentence.
        ;;   (pres eat.v)
        ((and (equal (list-length type-elems) 2)
              (equal (first type-elems) 'sent-op))
         (list (first lf-elems) (cdr lf-elems) 'sentence 'ordered start))

        ;; Identity functions.
        ((and (= (list-length type-elems) 2)
              (equal (first type-elems) 'identity))
         (list (first lf-elems) (cdr lf-elems)
               (second type-elems) 'ordered start))

        ;; Punctuation.
        ;; Two things, one being punctuation.  Assumes the other is a
        ;; sentence.
        ((and (= (list-length type-elems) 2)
              (setf operator-pos (position 'punct type-elems)))
         (list (nth operator-pos lf-elems)
               (util:remove-nth operator-pos lf-elems)
               'sentence 'ordered start))

        ;; If the second argument is unknown, but the first argument
        ;; is known.  Use the known argument as the type.
        ((and (equal (list-length type-elems) 2)
              (equal (second type-elems) 'unknown)
              (not (member (first type-elems) '(unknown var))))
         (list (first lf-elems)
               (subseq lf-elems 1)
               'predicate 'ordered start))

        ;; Unknown second argument at this point defaults to
        ;; a multi-argument predicate.
        ((and (< 1 (list-length type-elems))
              (equal (second type-elems) 'unknown)
              (not (equal (first type-elems) 'predicate)))
         (list (second lf-elems)
               (append (subseq lf-elems 0 1)
                       (subseq lf-elems 2))
               'sentence 'ordered start))


        ;; Logical Fragments.
        ;; Assume sentence.
        ((equal (first type-elems) 'fragment)
         (list (first lf-elems)
               (subseq lf-elems 1)
               'sentence 'ordered start))

        ;; Bunch of variables with a non-variable.
        ;; Set the non-variable as the type.
        ((and (member 'var type-elems)
              (< 0 (list-length
                     (remove-if
                       #'(lambda (x) (equal x 'var))
                       type-elems))))
         (let ((idx (position-if #'(lambda (x) (not (equal x 'var)))
                                 type-elems)))
           (list (nth idx lf-elems)
                 (append (subseq lf-elems 0 idx)
                         (subseq lf-elems (+ idx 1)))
                 'sentence 'ordered start)))

        ;; Generic single argument predicates (probably),
        ;; where the argument is of unknown type.  Defaulting
        ;; to a predicate.
        ;; Predicate can be an ambiguous role verb,
        ;; adjective, or noun.
        ((and (equal 2 (list-length type-elems))
              (or (equal (second type-elems) 'predicate)
                  (equal (second type-elems) 'verb)
                  (equal (second type-elems) 'noun)
                  (equal (second type-elems) 'adjective))
              (not (equal (first type-elems) 'predicate)))
         (list (second lf-elems)
               (subseq lf-elems 0 1)
               'sentence 'ordered start))

        ;; Unknown argument structure.
        ;; Print message and simply use the first value as the type.
        (t (progn
             (if *ulf2amr-debug*
               (format t "UNKNOWN LF structure!~%Types ~s~%LFs ~s~%~%~%" type-elems lf-elems))
             (list (first lf-elems)
                   (cdr lf-elems)
                   'sentence 'ordered start)
             ))
        )) ; End of default branch of node-type condition.
      )) ; End of node-type condition.


(defun arcmap-lf-components (type-elems lf-elems node-type ulf el-vars all-vars start)
;~~~~~~~~~~~~~~~~~~~~
; Bracketing-only informed argument marking.
  (list (first lf-elems) (cdr lf-elems) 'ignore 'ordered start))

(defun argstruct-lf-components (type-elems lf-elems node-type ulf el-vars all-vars start)
;````````````````````````````````
; Generates the lf-components for argument structure format.
;
; Parameters:
;   type-elems - types of the current list of elements
;   lf-elems - lfs of the current list of elements
;   node-type - symbol for the type of the current node operation (e.g. pred1, conj, lambda, etc.)
;   ulf - current ULF state
;   el-vars - all existing semantic variables
;   all-vars - all existing syntactic variables
;   start - current variable starting index
;
; Returns:
; a list of the components of the current logical expression
;   1. the operator expression
;   2. the list of arc-argument pairs
;   3. the resulting constituent type
;   4. 'ordered or 'unordered (referring to the arguments)
;   5. the new starting index for variables
  (let ((arg-labels (gen-arg-labels (1- (list-length lf-elems))))
        (mod-labels (loop for n from 0 below (1- (list-length lf-elems)) by 1
                          collect 'mod)))
    (cond
      ;; Determiner.
      ((and (= (list-length type-elems) 2)
            (equal (first type-elems) 'determiner)
            (member (second type-elems) '(predicate noun)))
       (list (first lf-elems)
             (list (list ':restrictor (second lf-elems)))
             'pronoun 'ordered start))

      ;; Auxiliaries.
      ((and (= (list-length type-elems) 2)
            (equal (first type-elems) 'aux)
            (member (second type-elems) '(verb predicate)))
       (list (first lf-elems)
             (list (list ':arg0 (second lf-elems)))
             'verb 'ordered start))
      ;; SUB
      ;((and (= (list-length type-elems) 3)
      ;      (equal (first type-elems) 'sub))
      ; (list (first lf-elems) (cdr lf-elems)
      ;       ;; Type depends on how the substitution composes.
      ;       ;; TODO: run substitutation and analyze the child.
      ;       ...

      ;; N+PREDS
      ((and (> (list-length type-elems) 1)
            (equal (first type-elems) 'n+preds)
            (member (second type-elems) '(predicate noun)))
       (list (first lf-elems)
             (mapcar #'list arg-labels (cdr lf-elems))
             'predicate 'ordered start))

      ;; NP+PREDS
      ((and (> (list-length type-elems) 1)
            (equal (first type-elems) 'np+preds)
            (member (second type-elems) '(pronoun skolem)))
       (list (first lf-elems)
             (mapcar #'list arg-labels (cdr lf-elems))
             'pronoun 'ordered start))

      ;; Coordinating conjunctions.
      ((equal node-type 'conj)
       (progn
         (when *ulf2amr-debug*
           (format t "in conj node type~%")
           (format t "lf-elems ~s~%" lf-elems))
         (let ((coordinator
                 (third
                   (find-if #'(lambda (x) (and (consp x)
                                               (< 2 (length x))
                                               (ulf:lex-coord? (third x))))
                            lf-elems)))
               (filtered
                 (remove-if #'(lambda (x) (and (consp x)
                                               (< 2 (length x))
                                               (ulf:lex-coord? (third x))))
                            lf-elems)))
           (when *ulf2amr-debug*
             (format t "finished coordinator filtering.~%~%~%"))
           (list coordinator
                 (mapcar #'list (subseq arg-labels 0 (length filtered)) filtered)
                 'sentence 'unordered start))))

      ;; Lambda.
      ((equal node-type 'lambda)
       (list (first lf-elems)
             (mapcar #'list arg-labels (subseq lf-elems 1))
             'predicate 'ordered start))

    ;; Default case.
    ;; Differentiate between predicate modification, curried
    ;; predicate and basic predicate application.
    (t
      (cond
        ;; Adverbival predicate modifier:
        ;;  (adv-a ...), (attr ...)
        ((and (equal (list-length type-elems) 2)
              (equal (first type-elems) 'adv-pred-mod))
         (list (first lf-elems)
               (mapcar #'list arg-labels (cdr lf-elems))
               'adverb 'ordered start))

        ;; Adverbial sentence modifier:
        ;;  (adv-e, adv-p, adv-f, ...)
        ((and (equal (list-length type-elems) 2)
              (equal (first type-elems) 'adv-sent-mod))
         (list (first lf-elems)
               (mapcar #'list arg-labels (cdr lf-elems))
               'sent-op 'ordered start))

        ;; Verb Predicate modification.
        ;;   (quickly.adv run.v)
        ;;   (quickly.adv (...))
        ((and (equal (list-length type-elems) 2)
              (equal (first type-elems) 'adverb)
              (or (equal (second type-elems) 'predicate)
                  (member (second type-elems) '(verb adjective noun))))
         (list (first lf-elems)
               (mapcar #'list arg-labels (cdr lf-elems))
               'predicate 'ordered start))

        ;; Reverse Verb Predicate modification.
        ;;   (run.v quickly.adv)
        ;;   ((...) quickly.adv)
        ((and (equal (list-length type-elems) 2)
              (equal (second type-elems) 'adverb)
              (or (equal (first type-elems) 'predicate)
                  (member (first type-elems) '(verb adjective noun))))
         (list (second lf-elems)
               (mapcar #'list arg-labels (list (first lf-elems)))
               'predicate 'ordered start))
        
        ;; Perfect and progressive aspect.
        ((and (= (length type-elems) 2)
              (eql (first type-elems) 'perfprog)
              (member (second type-elems) '(predicate verb)))
         (list (first lf-elems)
               (mapcar #'list arg-labels (cdr lf-elems))
               (second type-elems) 'ordered start))

        ;; Noun Predicate modification.
        ;; Adjective modification of noun predicate:
        ;;   (pickling.a juice.n)
        ;; or pluralization of noun prediacte.
        ;;   (plur thing.n)
        ((and (equal (list-length type-elems) 2)
              (or (equal (first type-elems) 'adjective)
                  (equal (first type-elems) 'pred-mod))
              (equal (second type-elems) 'noun))
         (list (first lf-elems)
               (mapcar #'list arg-labels (subseq lf-elems 1))
               'predicate 'ordered start))

        ;; Transitive predicate modifiers
        ;; (pasv x.v), ....
        ((and (equal (list-length type-elems) 2)
              (equal (first type-elems) 'pred-mod))
         (list (first lf-elems)
               (mapcar #'list arg-labels (subseq lf-elems 1))
               (second type-elems) 'ordered start))

        ;; Prefix verb (assume subject arg is missing so start at :arg1)
        ((and (> (list-length type-elems) 1)
              (member (first type-elems) '(verb)))
         (list (first lf-elems)
               (mapcar #'list
                       (cdr (gen-arg-labels (length lf-elems)))
                       (subseq lf-elems 1))
               'predicate 'ordered start))

        ;; Modifier creation.
        ((and (equal (list-length type-elems) 2)
              (equal (first type-elems) 'mod-creator)
              (member (second type-elems) '(verb noun adjective predicate)))
         (list (first lf-elems)
               (mapcar #'list arg-labels (cdr lf-elems))
               'pred-mod 'ordered start))

        ;; Sentence operation.
        ;;   (pres (John see Mary))
        ;;   ((adv-e (in-loc forest)) ...)
        ((and (equal (list-length type-elems) 2)
              (equal type-elems '(sent-op sentence)))
         (list (first lf-elems)
               (mapcar #'list arg-labels (cdr lf-elems))
               'sentence 'ordered start))

        ;; Sentence nominalization.
        ((and (equal (list-length type-elems) 2)
              (equal type-elems '(sent-nom sentence)))
         (list (first lf-elems)
               (mapcar #'list arg-labels (cdr lf-elems))
               'pronoun 'ordered start))

        ;; Tense.
        ;; Since tense requires moving the order of operators, use TTT to post-process. Besides, this
        ;; only requires atom recognition.
        ((and (equal (list-length type-elems) 2)
              (equal (first type-elems) 'tense))
         (list (first lf-elems)
               (mapcar #'list arg-labels (cdr lf-elems))
               (second type-elems) 'ordered start))

        ;; quote-o
        ((and (equal (list-length type-elems) 2)
              (equal (first type-elems) 'quote-o))
         (list (first lf-elems)
               (mapcar #'list arg-labels (cdr lf-elems))
               'pronoun 'ordered start))

        ;; bracket-op
        ((and (> (list-length type-elems) 0)
              (equal (first type-elems) 'bracket-op))
         (list (first lf-elems)
               (mapcar #'list arg-labels (cdr lf-elems))
               'pronoun 'ordered start))

        ;; Predicate nominalization.
        ;; Some words may be predicates, but not marked yet,
        ;; so we can be flexible with predicate nominalizations.
        ;; Allow anything that is length 2 with a predicate
        ;; nominalization to start.
        ((and (equal (list-length type-elems) 2)
              (equal (first type-elems) 'pred-nom))
         (list (first lf-elems)
               (mapcar #'list arg-labels (cdr lf-elems))
               'pronoun 'ordered start))

        ;; Preposition arguments.
        ((and (> (list-length type-elems) 1)
              (equal (first type-elems) 'parg))
         (list (first lf-elems)
               (mapcar #'list arg-labels (cdr lf-elems))
               'pronoun 'ordered start))

        ;; Preposition.
        ((and (equal (list-length type-elems) 2)
              (equal (first type-elems) 'preposition)
              (member (second type-elems) '(pronoun noun var)))
         (list (first lf-elems)
               (mapcar #'list arg-labels (cdr lf-elems))
               'predicate 'ordered start))

        ;; Sent to pred.
        ((and (equal (list-length type-elems) 2)
              (equal (first type-elems) 'sent-to-pred)
              (equal (second type-elems) 'sentence))
         (list (first lf-elems)
               (mapcar #'list arg-labels (cdr lf-elems))
               'predicate 'ordered start))

        ;; Other sent to pred (bad name).
        ((and (equal (list-length type-elems) 3)
              (equal (first type-elems) 'sent-to-pred))
         (list (first lf-elems)
               (mapcar #'list arg-labels (cdr lf-elems))
               'sentence 'ordered start))

        ;; Set forming operator.
        ((and (< 1 (list-length type-elems))
              (equal (first type-elems) 'set-form)
              (reduce #'(lambda (x y) (and x (equal y 'pronoun)))
                      (cdr type-elems)))
         (list (first lf-elems)
               (mapcar #'list arg-labels (cdr lf-elems))
               'pronoun 'unordered start))

        ;; SUCH-AS operator.
        ;; Maps a pronoun and predicate into a new predicate.
        ((and (equal 3 (list-length type-elems))
              (equal (first type-elems) 'such-as))
         (list (first lf-elems)
               (mapcar #'list arg-labels (cdr lf-elems))
               'predicate 'ordered start))

        ;;;;;;;;;;;;;;;;;;;;;;;;;;;
        ;;; Below are failure mode analyses (type don't correctly
        ;;; compose but we can guess based on restrictive types).
        ;;;;;;;;;;;;;;;;;;;;;;;;;;;

        ;; Determiner/NP+PREDS failure modes
        ((member (first type-elems) '(determiner np+preds))
         (list (first lf-elems) (mapcar #'list arg-labels (cdr lf-elems)) 'pronoun 'ordered start))

        ;; N+PREDS failure mode
        ((member (first type-elems) '(n+preds))
         (list (first lf-elems) (mapcar #'list arg-labels (cdr lf-elems)) 'noun 'ordered start))

        ;; SUB failure mode
        ;((member (first type-elems) '(sub))
        ; (list (first lf-elems) (mapcar #'list arg-labels (cdr lf-elems))
               ;; TODO recursive analysis.

        ;; Preposition.
        ;; Preposition + pronoun -> predicate
        ((and (= (list-length type-elems) 2)
              (equal (first type-elems) 'preposition)
              (equal (second type-elems) 'pronoun))
         (list (first lf-elems) (mapcar #'list arg-labels (cdr lf-elems)) 'predicate 'ordered start))

        ;; Comparison
        ((and (equal (list-length type-elems) 3)
              (equal (first type-elems) 'comp-op))
         (list (first lf-elems) (mapcar #'list arg-labels (cdr lf-elems)) 'sentence 'ordered start))

        ;; Generic multi-argument predicates.
        ;; Predicate can be an ambiguous role verb.
        ((and (< 2 (list-length type-elems))
              (or (equal (second type-elems) 'predicate)
                  (equal (second type-elems) 'verb))
              (not (equal (first type-elems) 'predicate)))
         (list (second lf-elems)
               (mapcar #'list
                       arg-labels
                       (append (subseq lf-elems 0 1)
                               (subseq lf-elems 2)))
               'sentence 'ordered start))

         ;; Generic single argument predicates.
         ;; Predicate can be an ambiguous role verb,
         ;; adjective, or noun.
        ((and (equal 2 (list-length type-elems))
              (or (equal (second type-elems) 'predicate)
                  (equal (second type-elems) 'verb)
                  (equal (second type-elems) 'adjective))
              (not (member (first type-elems) '(predicate sent-op))))
         (list (second lf-elems)
               (mapcar #'list arg-labels (subseq lf-elems 0 1))
               'sentence 'ordered start))

        ;; Single argument noun predicates.
        ;; This can only operate on variables or pronouns.
        ((and (equal 2 (list-length type-elems))
              (equal (second type-elems) 'noun)
              (member (first type-elems) '(pronoun var)))
         (list (second lf-elems)
               (mapcar #'list arg-labels (subseq lf-elems 0 1))
               'sentence 'ordered start))

        ;; Prefixed predicate application.
        ;; Assume the subject argument is missing (start at :arg1)
        ((and (< 1 (list-length type-elems))
              (or (equal (first type-elems) 'predicate)
                  (equal (first type-elems) 'verb)
                  (equal (first type-elems) 'adjective)
                  (equal (first type-elems) 'noun))
              (not (member (second type-elems) '(predicate sent-op))))
         (list (first lf-elems)
               (mapcar #'list
                       (cdr (gen-arg-labels (length lf-elems)))
                       (subseq lf-elems 1))
               'predicate 'ordered start))

        ;; Adverb on a sentence.
        ;; This is not valid ULF, but handled for robustness.
        ;; (ususally.adv <sentence>)
        ((and (equal (list-length type-elems) 2)
              (equal (first type-elems) 'adverb)
              (equal (second type-elems) 'sentence))
         (list (first lf-elems)
               (mapcar #'list arg-labels (cdr lf-elems))
               'sentence 'ordered start))

        ;; Sentence operation when there is no sentence.
        ;;   (pres eat.v)
        ((and (equal (list-length type-elems) 2)
              (equal (first type-elems) 'sent-op))
         (list (first lf-elems) (mapcar #'list arg-labels (cdr lf-elems)) 'sentence 'ordered start))

        ;; Identity functions.
        ((and (= (list-length type-elems) 2)
              (equal (first type-elems) 'identity))
         (list (first lf-elems) (mapcar #'list arg-labels (cdr lf-elems))
               (second type-elems) 'ordered start))

        ;; Punctuation.
        ;; Two things, one being punctuation.  Assumes the other is a
        ;; sentence.
        ((and (= (list-length type-elems) 2)
              (setf operator-pos (position 'punct type-elems)))
         (list (nth operator-pos lf-elems)
               (mapcar #'list
                       arg-labels
                       (util:remove-nth operator-pos lf-elems))
               'sentence 'ordered start))

        ;; If the second argument is unknown, but the first argument
        ;; is known. Use the known argument as the type.
        ((and (equal (list-length type-elems) 2)
              (equal (second type-elems) 'unknown)
              (not (member (first type-elems) '(unknown var))))
         (list (first lf-elems)
               (mapcar #'list arg-labels (subseq lf-elems 1))
               'predicate 'ordered start))

        ;; Unknown second argument at this point defaults to
        ;; a multi-argument predicate.
        ((and (< 1 (list-length type-elems))
              (equal (second type-elems) 'unknown)
              (not (equal (first type-elems) 'predicate)))
         (list (second lf-elems)
               (mapcar #'list
                       arg-labels
                       (append (subseq lf-elems 0 1)
                               (subseq lf-elems 2)))
               'sentence 'ordered start))

        ;; Logical Fragments.
        ;; Assume sentence.
        ((equal (first type-elems) 'fragment)
         (list (first lf-elems)
               (mapcar #'list arg-labels (subseq lf-elems 1))
               'sentence 'ordered start))

        ;; Bunch of variables with a non-variable.
        ;; Set the non-variable as the type.
        ((and (member 'var type-elems)
              (< 0 (list-length
                     (remove-if
                       #'(lambda (x) (equal x 'var))
                       type-elems))))
         (let ((idx (position-if #'(lambda (x) (not (equal x 'var)))
                                 type-elems)))
           (list (nth idx lf-elems)
                 (mapcar #'list
                         arg-labels
                         (append (subseq lf-elems 0 idx)
                                 (subseq lf-elems (+ idx 1))))
                 'sentence 'ordered start)))

        ;; Generic single argument predicates (probably),
        ;; where the argument is of unknown type.  Defaulting
        ;; to a predicate.
        ;; Predicate can be an ambiguous role verb,
        ;; adjective, or noun.
        ((and (equal 2 (list-length type-elems))
              (or (equal (second type-elems) 'predicate)
                  (equal (second type-elems) 'verb)
                  (equal (second type-elems) 'noun)
                  (equal (second type-elems) 'adjective))
              (not (equal (first type-elems) 'predicate)))
         (list (second lf-elems)
               (mapcar #'list arg-labels (subseq lf-elems 0 1))
               'sentence 'ordered start))

        ;; Unknown argument structure.
        ;; Print message and simply use the first value as the type.
        (t (progn
             (if *ulf2amr-debug*
               (format t "UNKNOWN LF structure!~%Types ~s~%LFs ~s~%~%~%" type-elems lf-elems))
             (list (first lf-elems)
                   (mapcar #'list arg-labels (cdr lf-elems))
                   'sentence 'ordered start)
             ))
        )) ; End of default branch of node-type condition.
      ) ; End of top level cond
    ) ; End of top level let
  ) ; End of argstruct-lf-components


;; Generates an amr element with the given parameters.
;; Must not be a leaf node.
;;   elem-var : variable to assign variable
;;   elem-type : type of the amr element
;;   args : list of arguments
;;   ordered : whether the argument order matters
;;             (i.e. use :arg or :mod)
(defun amr-elem (elem-var elem-type args &optional (ordered t))
  (let ((arg-labels (if ordered
                      (gen-arg-labels (list-length args))
                      (loop for n from 0 below (list-length args) by 1
                            collect ':mod))))
    (argstruct-amr-elem elem-var elem-type (mapcar #'list arg-labels args))))


;; Generates an amr element with the given parameters.
;; Must not be a leaf node.
;;   elem-var : variable to assign variable
;;   elem-type : type of the amr element
;;   arc-args : list of arc-argument pairs
;;   ordered : whether the argument order matters
;;             (i.e. use :arg or :mod)
(defun argstruct-amr-elem (elem-var elem-type arc-args &optional ordered)
  (declare (ignore ordered)) ; included to keep number of args same as amr-elem.
  (cond
    ;; Element type is NULL (indicative of mapping error).
    ;; Handle like simple element types, but with NULL as the type.
    ((null elem-type)
     (append
       (list elem-var '/ elem-type)
       (reduce #'append arc-args)))
    ;; Simple element type.
    ;; Put the element type directly in the top node.
    ((symbolp elem-type)
     (append
       (list elem-var '/ elem-type)
       (reduce #'append arc-args)))
    ;; Simple element type with variable attached.
    ;; Put the element type directly in the top node,
    ;; ignoring its variable.
    ((and (= 3 (length elem-type))
          (symbolp (third elem-type)))
     (append
       (list elem-var '/ (third elem-type))
       (reduce #'append arc-args)))
    ;; Complex element type.
    ;; Put the element type as an argument and make the node type
    ;; 'COMPLEX, which will be ignored during scoring.
    (t
      (append
        (list elem-var '/ 'complex)
        (list ':instance elem-type)
        ;; Interleave argument labels with the arguments.
        (reduce #'append arc-args)))))


;; Generates argument labels :arg0 - :argn.
(defun gen-arg-labels (n)
  (mapcar #'(lambda (x)
              (read-from-string (format nil ":ARG~s" x)))
          (loop for x from 0 below n by 1
                collect x)))


(defun ulf2amr (ulf &key (method 'arcmap) (use-mod t) (recover-failures t))
;~~~~~~~~~~~~~~~~~~~~
; Converts ULF formula to AMR syntax, but is not interpretable semantics in AMR.
; The syntax conversion is solely for evaluating using the smatch algorithm.
; ULF must be scoped.  Recursively defined for best handling of complex
; logical form structures.
;
; method is a symbol that indicate the type of output that is desired.
;   'arcmap is a simply constituency to arc translation. e.g.
;       (A B C D) -> (x / A :arg0 (y / B) :arg1 (z / C) :arg2 (a / D))
;       where A may be expanded into COMPLEX+:instance if it's not an atom.
;   'argstruct converts it into the most EL-semantically coherent form, where
;       the operator comes first, and lexical markings are given special relations.
;       e.g. (i.pro (pres know.v))
;         -> (x / know.v :arg0 (y / i.pro) :tense (z / pres))
;   'trips converts to a format that looks similar to the TRIPS system output
;       by flattening certain chains of type-shifts into single relations
;       and using bidirectional relations to represent stuff like relative clauses.
;   'lexop-flat-arcmap is in between 'argstruct and 'arcmap. It acts like 'arcmap
;       except that lexical operators don't correspond to surface strings are 
;       flattened, e.g. tense, aspect, plur, pasv. Example:
;       (i.pro (pres know.v))
;         -> (x / i.pro :arg0 (y / know.v :tense (z / pres)))
;
; use-mod if a boolean flag that determines whether the penman output includes :mod
;   relations, which are unordered. This gives a better semantic meaning, but doesn't
;   allow for reconstruction into the original form.
  (when *ulf2amr-debug*
    (format t "ulf2amr ~s~%~%~%" ulf))
  (when (member method '(arcmap lexop-flat-arcmap))
    (setf use-mod nil)) ; Don't use mod with arcmap.
  (labels
    (
    ;; Determines the type of an atomic element.
    ;; Leaves it ambiguous, because often there isn't enough information with
    ;; the atomic element alone to determine if it is a name or predicate, etc.
    (type-of-atom
      (ulf)
      (cond
        ;; Sanity check.
        ((not (or (symbolp ulf) (numberp ulf)))
         (progn
           (if *ulf2amr-debug*
             (format t "NOT AN ATOM!! ~s~%~%" ulf))
           nil))

        ;; ALL, SOME, EXISTS, MANY, etc.
        ;((or (quan? ulf) (member ulf '(exists exist a an the most few several that those my his one two three four five six seven eight nine ten no))) 'quant)

        ((lex-ps? ulf) 'sent-to-pred)
        ((lex-det? ulf) 'determiner)
        ((numberp ulf) 'pronoun)
        ((lex-coord? ulf) 'conj)
        ((lex-aux? ulf) 'aux)
        ((lex-verb? ulf) 'verb)
        ((lex-adv? ulf) 'adverb)
        ((member ulf '(such)) 'adverb)
        ((lex-noun? ulf) 'noun)
        ((lex-adjective? ulf) 'adjective)
        ((or (lex-pronoun? ulf)
             (lex-name? ulf)) 'pronoun)
        ;((lf-skolem? ulf) 'skolem)
        ((lex-prep? ulf) 'preposition)
        ;((lf-base? ulf "X") 'sentence)
        ((equal 'l ulf) 'lambda)
        ((member ulf '(k ka)) 'pred-nom)
        ((member ulf '(ke that tht whether qnom ans-to)) 'sent-nom)
        ((member ulf '(=i =r =f =p =q =l =o =a)) 'colon-key)
        ((member ulf '(adv-e adv-f adv-s)) 'adv-sent-mod)
        ((member ulf '(adv-a attr in-loc)) 'adv-pred-mod)
        ((member ulf '(not)) 'sent-op)
        ((member ulf '(perf prog)) 'perfprog)
        ((member ulf '(! ?)) 'punct)
        ((lex-p-arg? ulf) 'parg)
        ;((or (lf-degree? ulf) (member ulf '(quote-i))) 'identity)
        ((member ulf '(more-than less-than larger-than
                                 smaller-than equal-to as-as)) 'comp-op)
        ((member ulf '(past pres past? pres?)) 'tense)
        ((member ulf '(plur pasv)) 'pred-mod)
        ((member ulf '(n+preds)) 'n+preds)
        ((member ulf '(np+preds)) 'np+preds)
        ((member ulf '(= < > poss-by)) 'predicate)
        ((member ulf '(attr nn nnp mod-a mod-n)) 'mod-creator)
        ((member ulf '(quote)) 'quote)
        ;((member ulf '(|"| |"|)) quote) ; There are two here just for syntax highlighting.
        ;((member ulf '(quote-o quote-start quote-end)) 'quote-o)
        ;((member ulf '(quote-i)) 'quote-i)
        ;((member ulf '(lab rab lcb rcb lsb rsb lrb rrb ab sb cb rb)) 'bracket-op)
        ;((equal ulf 'pair) 'pair)
        ;((equal ulf 'logical-fragments.f) 'fragment)
        ((member ulf '(set-of)) 'set-form)
        ;;  TODO: add sub, qt-attr, rep, etc.
        ;((member ulf '(such-as)) 'such-as)
        (t
          (progn
            (if *ulf2amr-debug*
              (format t "Unknown atomic-type ~s~%~%" ulf))
            'unknown))))

    ;; Determines the type of the node based on the types of the
    ;; constituent element types.
    ;; TODO: update to ULF.
    (type-of-node
      (elem-types)
      (cond
        ;; Identify large cases here:
        ;;        quanifiers, conj, colon, lambda, etc.
        ;;      Figure out the details during the formation:
        ;;        and vs or, 1-place pred vs curried-pred, etc.

        ;; Basic sentence structures.
        ((and (> (list-length elem-types) 1)
              (equal '(quant noun) (subseq elem-types 0 2))) 'quant)
        ((and (> (list-length elem-types) 1)
              (equal 'quant (first elem-types))) 'bad-quant)

        ((and (> (list-length elem-types) 1)
              (equal 'preposition (first elem-types))) 'pred1)

        ;; Conjunctions.
        ((member 'conj elem-types) 'conj)

        ;; Lambda expression.
        ((equal (first elem-types) 'lambda) 'lambda)

        ;; Colon-keyword
        ((equal (first elem-types) 'colon-key) 'colon-key)

        ;; Otherwise, default lf (handle straight-forwardly).
        (t 'default)))


    ;; Helper function that does all the heavy lifting, but requires extra
    ;; arguments.
    ;;  el:       ULF formula
    ;;  el-vars:  list of semantically assigned variables.
    ;;  amr-vars: list of syntactically assigned variables.
     (helper
       (ulf el-vars all-vars start)
       ;; Uses following algorithm:
       ;;   1. If atomic, determine type and return.
       ;;     Otherwise
       ;;   2. Generate variables for current amr node.
       ;;   3. If quantifier, generate amr variable for quantifier variable.
       ;;   4. For each element recurse to get type and AMR representation.
       ;;   5. Use element types to determine LF type.
       ;;   6. Construct AMR LF:
       ;;     a. preprocess as necessary (e.g. remove and.cc)
       ;;     b. form AMR as ordered or orderless AMR node.

       ;; 1. If ULF variable, return the AMR variable leaf node.
       ;(format t "Before variable identification ~s ~s ~s ~s~%~%" ulf el-vars all-vars start)
       (if (member ulf el-vars :key #'car)
         (progn
           (if *ulf2amr-debug*
             (format t "in variable identification ~s~%~%" (cadar (member ulf el-vars :key #'(lambda (x) (car x))))))
           (let* ((entry (car (member ulf el-vars :key #'car)))
                  (ev (first entry))
                  (av (second entry))
                  (used (third entry)))
             (if used
               ;; Variable already used, don't need to make a var leaf node,
               ;; just use the var directly.
               (return-from
                 helper (list av 'var start))
               ;; Otherwise, update the entry in el-vars and return a var
               ;; leaf node.
               (progn
                 (setf (third (car (member ulf el-vars :key #'car))) t)
                 (return-from
                   helper (list (list av '/ 'var) 'var start)))))))

       (let* ((varbase 'v)
              ;; 2. Generate variables for current AMR node.
              (retval (generate-variable varbase start all-vars))
              (topvar (first retval))
              (start (second retval))
              (all-vars (third retval))
              ;; Local variables.
              quantvar processed-elems node-type lf-elems type-elems
              operator-pos operator operator-type
              operand-pos operand operand-type)

         ;; If it's a name or a lexical name, change representation to a string.
         (if (or (lex-name-pred? ulf) (lex-name? ulf))
           (return-from helper (list (list topvar '/ (format nil "~s" ulf))
                                     (type-of-atom ulf) start)))

         ;; If it's atomic, return atomic type, leaf node.
         (if (atom ulf)
           (return-from helper (list (list topvar '/ ulf)
                                     (type-of-atom ulf) start)))

         ;; 3. Check if it's a lambda expression.
         ;;    If so, generate AMR variable for quantifier variable.
         ;;    Add the variables to the ULF-AMR variable mapping and
         ;;    the set of AMR variables used.
         (if (and (listp ulf)
                  (symbolp (first ulf))
                  (symbolp (second ulf))
                  (or
                    (equal 'l (first ulf))
                    (equal '=l (first ulf))))
           (progn
             (setf retval (generate-variable varbase start all-vars))
             (setf quantvar (first retval))
             (setf start (second retval))
             (setf all-vars (third retval))
             ;; Insert new variable into the el-vars and mark that it
             ;; has not been used yet.
             (setf el-vars (cons (list (second ulf) quantvar nil) el-vars))
             (setf all-vars (cons (second ulf) all-vars))
             ))


         ;; 4. Recurse for each element to get the type and AMR representation
         ;;    for each element in the current node.
         (setf processed-elems
               ;; Assumes these are processed sequentially for correct
               ;; variable name generation.
               (reduce
                 #'(lambda (x y)
                     (setf start (third (car y)))
                     (cons (helper x el-vars all-vars start) y))
                 (reverse ulf)
                 :initial-value (list (list nil nil start))
                 :from-end t))
         (setf start (third (car processed-elems)))
         ;; Remove extra one we added on for processing and reverse back.
         (setf processed-elems (cdr (reverse processed-elems)))
         ;; Element list with only the lfs or types.
         (setf lf-elems (mapcar #'first processed-elems))
         (setf type-elems (mapcar #'second processed-elems))

         (when *ulf2amr-debug*
           (format t "lf elems ~s~%~%" lf-elems)
           (format t "type elems ~s~%~%" type-elems))

         ;; 5. Determine node type from types of the elements.
         (setf node-type
               (type-of-node (mapcar #'second processed-elems)))

         (when *ulf2amr-debug*
           (format t "node type ~s~%~%" node-type))

         ;; 6. Construct AMR lf.
         ;;    a. Determine components with special preprocessing
         ;;       based on node-type.
         ;;    b. Construct AMR lf with components.
         (setf lf-comp-fn
               (case method
                 (orig #'orig-lf-components)
                 (arcmap #'arcmap-lf-components)
                 (argstruct #'argstruct-lf-components)
                 (lexop-flat-arcmap #'arcmap-lf-components)))
         (setf lf-components
               (apply lf-comp-fn
                      (list type-elems lf-elems node-type ulf
                            el-vars all-vars start)))

         (if *ulf2amr-debug*
           (format t "before final lf ~s ~s~%~%" topvar lf-components))
         ;; Construct the AMR LF based on ordered or unordered.
         (setf construct-fn
               (case method
                 (orig #'amr-elem)
                 (arcmap #'amr-elem)
                 (argstruct #'argstruct-amr-elem)
                 (lexop-flat-arcmap #'amr-elem)))
         (setf final-lf
               (apply construct-fn
                      (list topvar (first lf-components) (second lf-components)
                            (or (not use-mod)
                                (equal (fourth lf-components) 'ordered)))))
         (setf start (fifth lf-components))
         (if *ulf2amr-debug*
           (format t "Final lf ~s~%~%~%" final-lf))

         ;; Return the a list of (AMR-lf, type, start)
         (list final-lf (third lf-components) start)
         ) ; end of let*
         ) ; end of helper
       ) ; end of labels function definitions.

    (if *ulf2amr-debug*
      (format t "in ulf2amr body!~%"))
    ;; Body of ulf2amr
    ;; Replace colon keywords with = sign because colon (:) has a special
    ;; meaning in AMR.
    (let ((ulf (read-from-string
                 (cl-ppcre:regex-replace-all
                   "\\:" (format nil "~s" ulf) "=")))
          ulfamr)
      (setf ulfamr (car (helper ulf '() '() 0)))
      (ulfamr-postprocess ulfamr :method method)))) ; end of ulf2amr

