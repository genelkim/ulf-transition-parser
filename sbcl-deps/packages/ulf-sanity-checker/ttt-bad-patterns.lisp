;; TTT patterns for errors.

(in-package :ulf-sanity-checker)

(defparameter *ttt-bad-det*
  '(!1 (_* det? _*)
      ;; The only things that can be arguments to determiners are nouns,
      ;; rarely prepositions, and typeless predicates (e.g. (= ..)).
       ~ (det? (! noun? pp? (= _!) unknown?))
         ((* det?) lex-coord? (+ det?))))
;  '(!1
;      (det? _! _+)
;      (det? (! ~ noun? pp? (= _!) unknown?))
;      (_+ det? _*)))
(defparameter *bad-det-msg*
  "Determiners take 1 nominal (noun) or, rarely, prepositional argument.")

(defparameter *ttt-bad-prep*
  '(!1 (lex-p? _! _+)
      (lex-p? (! ~ term? unknown?))
      (_+ lex-p? _*)))
(defparameter *bad-prep-msg*
  "Simple prepositions (*.p) take terms as arguments and take 1 argument.")

(defparameter *ttt-bad-be*
  '(!1 (be.v _! _+)
      (be.v (! ~ pred?))
      ((lex-tense? be.v) _! _+)
      ((lex-tense? be.v) (! ~ pred?))))
(defparameter *bad-be-msg*
  "be.v takes a single predicate argument (unless used after an it-cleft -- ignore this if that's the case).")

(defparameter *ttt-bad-tensed-sent-op*
  '(!1 ((!2 tensed-sent-reifier? lex-ps?) _! _+)
      ((!3 tensed-sent-reifier? lex-ps?) (! ~ tensed-sent?))))
(defparameter *bad-tensed-sent-op-msg*
  "that/tht/whether/ans-to take a single tensed sentence argument.")

(defparameter *ttt-bad-sent-op*
  '(!1 (sent-reifier? _! _+)
      (sent-reifier? (! ~ sent?))
      (_+ sent-reifier? _*)))
(defparameter *bad-sent-op-msg*
  "ke takes a single untensed sentence argument.")

(defparameter *ttt-bad-verb-reifier*
  '(!1 (verb-reifier? _! _+)
      (verb-reifier? (! ~ verb?))
      (_+ verb-reifier? _*)))
(defparameter *bad-verb-reifier-msg*
  "ka/to/gd take a single untensed verb argument.")

(defparameter *ttt-bad-noun-reifier*
  '(!1 (noun-reifier? _! _+)
       (noun-reifier? (! ~ noun?))
       (_+ noun-reifier? _*)))
(defparameter *bad-noun-reifier-msg*
  "k takes a single nominal argument.")

(defparameter *ttt-conservative-bad-sent-reifier*
  '(!1
     (sent-reifier? _! _+)
     (sent-reifier? term?)
     (_+ sent-reifier? _*)
     (tensed-sent-reifier? _! _+)
     (tensed-sent-reifier? term?)
     (_+ tensed-sent-reifier? _*)))
(defparameter *conservative-bad-sent-reifier-msg*
  "ke/that/tht/whether/ans-to cannot take a term argument, and take a single argument each")

;(defparameter *ttt-bad-sent-reifier*
;  '(!1 (sent-reifier? _! _+)
;       (sent-reifier? (! ~ sent?))
;       (_+ sent-reifier? _*)))
;(defparameter *bad-sent-reifier-msg*
;  "ke takes a single untensed sentence argument.")
;
;(defparameter *ttt-bad-tensed-sent-reifier*
;  '(!1 (tensed-sent-reifier? _! _+)
;       (tensed-sent-reifier? (! ~ tensed-sent?))
;       (_+ tensed-sent-reifier? _*)))
;(defparameter *bad-tensed-sent-reifier-msg*
;  "that/tht/whether/ans-to take a single tensed sentence argument.")

(defparameter *ttt-bad-plur*
  '(!1 (plur _! _+)
      (plur (! ~ noun?))
      (_+ plur _*)))
(defparameter *bad-plur-msg*
  "plur takes a single nominal (noun) argument.")

(defparameter *ttt-bad-aux*
  '(!1 (aux? _! _+)
      (aux? (! ~ verb?))
      (tensed-aux? _! _+)
      (tensed-aux? (! ~ verb?))
      (_+ tensed-aux? _*)))
(defparameter *bad-aux-msg*
  "auxiliaries take a single untensed verb argument.")

(defparameter *ttt-bad-advformer*
  '(!1 (advformer? _! _+)
      (advformer? (! ~ pred?))
      (_+ advformer? _*)))
(defparameter *bad-advformer-msg*
  "Adverb formers (adv-a, adv-e,..) take a single predicate argument and cannot be used as an argument.")

(defparameter *ttt-bad-detformer*
  '(!1 (detformer? _! _+)
      (detformer? (! ~ adj? unknown?))))
(defparameter *bad-detformer-msg*
  "Determiner formers (nquan, fquan) take a single adjective argument.")

(defparameter *ttt-bad-np-preds*
  '(!1 (_+ np+preds _*) ; np+preds not used as prefix operator.
      (np+preds _!) ; Only 1 arg.
      (np+preds (! ~ term?) _+) ; first argument is not a term.
      (np+preds term? _*.1 (! ~ pred?) _*.2))) ; argument that is not the first is not a predicate.

(defparameter *bad-np-preds-msg*
  "np+preds takes at least 2 arguments where the first is a term and the rest are predicates.")


(defparameter *ttt-bad-n-preds*
  '(!1 (_+ n+preds _*) ; n+preds not used as prefix operator.
      (n+preds _!) ; Only 1 arg.
      (n+preds (! ~ noun?) _+) ; first argument is not a noun.
      (n+preds noun? _*1 (! ~ pred?) _*2) ; argument that is not the first is not a predicate.
      ))
(defparameter *bad-n-preds-msg*
  "n+preds takes at least 2 arguments where the first is a noun and the rest are predicates.")

(defparameter *ttt-bad-sent-punct*
  '(!1 (_! _+ sent-punct?) ; more than 1 arg.
      ((! ~ tensed-sent? unknown?) sent-punct?))) ; arg is not a tensed sentence.
(defparameter *bad-sent-punct-msg*
  "Sentence punctuation takes a single tensed sentence argument and is post-fixed.")

(defparameter *ttt-bad-double-tense*
  '(!1 (tensed-aux? tensed-verb?)
       (tensed-verb? tensed-verb?)))
(defparameter *bad-double-tense-msg*
  "Each embedded sentence should only have 1 tense operator.")

(defparameter *ttt-no-periods-or-commas*
  '(! \, \.))
(defparameter *no-periods-or-commas-msg*
  "Annotating commas and periods is no longer supported.")

(defparameter *ttt-old-ps-ann*
  '((! adv-e adv-a adv-s)
    (lex-ps? tensed-sent?)))
(defparameter *old-ps-ann-msg*
  "(adv-s (*.ps ...)) is no longer the way to annotate *.ps.")

(defparameter *ttt-bad-possessive*
  '(!1 (((!2 ~ term?) 's) noun?) ; first arg is not a term
       ((term? 's) (!2 ~ noun?)) ; second arg is not a noun
       ('s _+)      ; 's used as a prefix operator
       (_+1 's _+2) ; 's used flat
        ))
(defparameter *bad-possessive-msg*
  "The 's operator must be used in EXACTLY the following format ((<term> 's) <noun>).")

(defparameter *ttt-bad-pu*
  '(pu _! _+))
(defparameter *bad-pu-msg*
  "The 'pu' operator takes a single phrase.")

(defparameter *ttt-bad-flat-mod*
  '((*1 ~ verb?)
     (!2 adj? noun? term?)
     (!3 adj? noun? term?)
     (!4 adj? noun? term?)
     _*2))
(defparameter *bad-flat-mod-msg*
  "Predicate modifications should be scoped into operator-operand pairs.")

(defparameter *ttt-bad-single-bracket*
  '(_!))
(defparameter *bad-single-bracket-msg*
  "Brackets should not scope around a single constituent (need at least two members in its scope as an operator-operand pair).")

(defparameter *ttt-bad-equal*
  '(!1 (_* = _! _+)
       (_+ _! = _*)
       (_* = (! ~ term?))
       ((! ~ term?) = _*)))
(defparameter *bad-equal-msg*
  "Equality takes at most 1 argument on each side and the arguments must be individuals.")


(defun contains-var? (x sym)
  (ttt::match-expr (list '^* sym) x))
(defun contains-sub-var? (x)
  (contains-var? x '[*h]))
(defun contains-rep-var? (x)
  (contains-var? x '[*p]))
(defun contains-qt-attr-var? (x)
  (contains-var? x '[*qt]))
(defparameter *ttt-bad-sub*
  '(!1
     (sub _!)
     (sub)
     (sub _!2 _!3 _+)
     (_+ sub _*)
     (sub _!4 (!5 ~ contains-sub-var?))
     ))
(defparameter *bad-sub-msg*
  "'sub' operator should take two arguments and the second argument should contain a '*h'")

(defparameter *ttt-bad-rep*
  '(!1
     (rep _!)
     (rep)
     (rep _!2 _!3 _+)
     (_+ rep _*)
     (rep (!4 ~ contains-rep-var?) _!5)))
(defparameter *bad-rep-msg*
  "'rep' operator should take two arguments and the first argument should contain a '*p'")

(defparameter *ttt-bad-qt-attr*
  '(!1
     (_+ qt-attr _*)
     (qt-attr _! _+)
     (qt-attr)
     (qt-attr (!2 ~ contains-qt-attr-var?))))
(defparameter *bad-qt-attr-msg*
  "'qt-attr' operator should take one argument and it should contain a '*qt'")

(defparameter *bad-rel-sent-msg*
  "A relativizer (*.rel) must sit inside of a tensed sentence.")

(defparameter *ttt-bad-noun-pp*
  '(!1
     (noun? pp?)))
(defparameter *bad-noun-pp-msg*
  "A noun cannot be directly combined with a prepositional phrase.  Either use an n+preds variant or use a type-shifter.")

(defparameter *ttt-bad-verb-sent*
  '(!1
     (verb? sent?)
     (verb? tensed-sent?)))
(defparameter *bad-verb-sent-msg*
  "A verb cannot be directly combined with a sentence.")

(defparameter *ttt-bad-aux-before-arg*
  '(!1
     ((aux? verb?) (* adv-a?) term?)
     ((tensed-aux? verb?) (* adv-a?) term?)
     (((* adv-a?) (aux? verb?) (* adv-a?)) term?)))
(defparameter *bad-aux-before-arg-msg*
  "The auxiliary should be applied after all non-subject arguments. You can IGNORE this message if this is occurring within it-extra.pro.")

(defparameter *ttt-bad-pasv*
  '(!1
     (_+ pasv _*)
     (pasv (! ~ lex-verb?))))
(defparameter *bad-pasv-msg*
  "'pasv' must be in the following construction (pasv <verb>).")

(defparameter *ttt-bad-verb-args*
  '(!1
     (((! verb? tensed-verb?) _*1 (! term? p-arg? pred?) _*2) _*3 (! term? p-arg? pred?) _*4)
     ~  
     ((+ verb?) lex-coord? (+ verb?))
     ((+ tensed-verb?) lex-coord? (+ tensed-verb?))))
(defparameter *bad-verb-args-msg*
  "Verbs (both tensed and untensed) *must* take all non-subject arguments in a flat construction.")

(defparameter *ttt-bad-adv-a-arg*
  '(!1
     (((! verb? tensed-verb?) adv-a?) _*1 (! term? p-arg? pred?) _*2)
     ((adv-a? (! verb? tensed-verb?)) _*1 (! term? p-arg? pred?) _*2)
     ~  
     ((+ verb?) lex-coord? (+ verb?))
     ((+ tensed-verb?) lex-coord? (+ tensed-verb?))))
(defparameter *bad-adv-a-arg-msg*
  "All non-subject arguments must by supplied to the verb before directly applying action adverbs.")

(defparameter *ttt-suspicious-locative*
  '(!1 here.pro there.pro where.pro))
(defparameter *suspicious-locative-msg*
  "Suspicious: 'Here' and 'there' probably shouldn't be annotated with *.pro, since they're usually not substitutable with other generic terms.  It's usually *.adv-e or *.a.  Please double check.")

(defparameter *ttt-suspicious-do*
  '(_*1 do.aux-v _*2))
(defparameter *suspicious-do-msg*
  "Suspicious: 'do.aux-v' is only used for emphatic do.")

(defparameter *ttt-suspicious-will*
  '(_*1 will.aux-v _*2))
(defparameter *suspicious-will-msg*
  "Suspicious: 'will.aux-v' is only used when 'will.aux-s' doesn't make sense and 'will' is used for emphasis.")

(defparameter *ttt-bad-name-decomp*
  '(_*1 (! lex-name? lex-name-pred?) (! lex-name? lex-name-pred?) _*2))
(defparameter *bad-name-decomp-msg*
  "Names and name predicates should only be broken down on prepositions.")

(defparameter *ttt-bad-voc*
  '(!1 (voc (! ~ term? unknown?))
       (voc-O (! ~ term? unknown?))
       (_+ (! voc voc-O) _*)
       ((! voc voc-O) _! _+)))
(defparameter *bad-voc-msg*
  "Vocative operators ('voc', 'voc-O') take a single term argument.")

(defparameter *ttt-bad-sent-term*
  '(!1 (term? (! sent? tensed-sent?))
       ((! sent? tensed-sent?) term?)))
(defparameter *bad-sent-term-msg*
  "Sentences and terms cannot combine.")

(defparameter *ttt-old-adj-mod*
  '(!1 (adv-a? adj?)
       (adj? adv-a?)))
(defparameter *old-adj-mod-msg*
  "adv-a is no longer the modifier for adjectives.  Please use mod-a.")

;; Function definitions for this.
(defun bad-det? (x) (ttt::match-expr *ttt-bad-det* x))
(defun bad-prep? (x) (ttt::match-expr *ttt-bad-prep* x))
(defun bad-be? (x) (ttt::match-expr *ttt-bad-be* x))
(defun bad-tensed-sent-op? (x) (ttt::match-expr *ttt-bad-tensed-sent-op* x))
(defun bad-sent-op? (x) (ttt::match-expr *ttt-bad-sent-op* x))
(defun bad-verb-reifier? (x) (ttt::match-expr *ttt-bad-verb-reifier* x))
(defun bad-noun-reifier? (x) (ttt::match-expr *ttt-bad-noun-reifier* x))
(defun bad-plur? (x) (ttt::match-expr *ttt-bad-plur* x))
(defun bad-aux? (x) (ttt::match-expr *ttt-bad-aux* x))
(defun bad-advformer? (x) (ttt::match-expr *ttt-bad-advformer* x))
(defun bad-detformer? (x) (ttt::match-expr *ttt-bad-detformer* x))
(defun bad-np-preds? (x) (ttt::match-expr *ttt-bad-np-preds* x))
(defun bad-n-preds? (x) (ttt::match-expr *ttt-bad-n-preds* x))
(defun bad-sent-punct? (x) (ttt::match-expr *ttt-bad-sent-punct* x))
(defun bad-double-tense? (x) (ttt::match-expr *ttt-bad-double-tense* x))
(defun no-periods-or-commas? (x) (ttt::match-expr *ttt-no-periods-or-commas* x))
(defun old-ps-ann? (x) (ttt::match-expr *ttt-old-ps-ann* x))
(defun bad-possessive? (x) (ttt::match-expr *ttt-bad-possessive* x))
(defun bad-pu? (x) (ttt::match-expr *ttt-bad-pu* x))
(defun bad-flat-mod? (x) (ttt::match-expr *ttt-bad-flat-mod* x))
(defun bad-single-bracket? (x) (ttt::match-expr *ttt-bad-single-bracket* x))
(defun bad-sub? (x) (ttt::match-expr *ttt-bad-sub* x))
(defun bad-rep? (x) (ttt::match-expr *ttt-bad-rep* x))
(defun bad-qt-attr? (x) (ttt::match-expr *ttt-bad-qt-attr* x))
(defun bad-equal? (x) (ttt::match-expr *ttt-bad-equal* x))
(defun bad-rel-sent? (x)
  (and (contains-relativizer x)
       (sent? x)
       (not (tensed-sent? x))))
(defun conservative-bad-sent-reifier? (x)
  (ttt::match-expr *ttt-conservative-bad-sent-reifier* x))
(defun bad-noun-pp? (x) (ttt::match-expr *ttt-bad-noun-pp* x))
(defun bad-verb-sent? (x) (ttt::match-expr *ttt-bad-verb-sent* x))
(defun bad-aux-before-arg? (x) (ttt::match-expr *ttt-bad-aux-before-arg* x))
(defun bad-pasv? (x) (ttt::match-expr *ttt-bad-pasv* x))
(defun bad-verb-args? (x) (ttt::match-expr *ttt-bad-verb-args* x))
(defun bad-adv-a-arg? (x) (ttt::match-expr *ttt-bad-adv-a-arg* x))
(defun suspicious-locative? (x) (ttt::match-expr *ttt-suspicious-locative* x))
(defun suspicious-do? (x) (ttt::match-expr *ttt-suspicious-do* x))
(defun suspicious-will? (x) (ttt::match-expr *ttt-suspicious-will* x))
(defun bad-name-decomp? (x) (ttt::match-expr *ttt-bad-name-decomp* x))
(defun bad-voc? (x) (ttt::match-expr *ttt-bad-voc* x))
(defun bad-sent-term? (x) (ttt::match-expr *ttt-bad-sent-term* x))
(defun old-adj-mod? (x) (ttt:match-expr *ttt-old-adj-mod* x))

(defparameter *bad-pattern-test-pairs*
  (list
    (list #'bad-det? *bad-det-msg*)
    (list #'bad-prep? *bad-prep-msg*)
    (list #'bad-be? *bad-be-msg*)
    (list #'bad-tensed-sent-op? *bad-tensed-sent-op-msg*)
    (list #'bad-sent-op? *bad-sent-op-msg*)
    (list #'bad-verb-reifier? *bad-verb-reifier-msg*)
    (list #'bad-noun-reifier? *bad-noun-reifier-msg*)
    (list #'bad-plur? *bad-plur-msg*)
    (list #'bad-aux? *bad-aux-msg*)
    (list #'bad-advformer? *bad-advformer-msg*)
    (list #'bad-detformer? *bad-detformer-msg*)
    (list #'bad-np-preds? *bad-np-preds-msg*)
    (list #'bad-n-preds? *bad-n-preds-msg*)
    (list #'bad-sent-punct? *bad-sent-punct-msg*)
    (list #'bad-double-tense? *bad-double-tense-msg*)
    (list #'no-periods-or-commas? *no-periods-or-commas-msg*)
    (list #'old-ps-ann? *old-ps-ann-msg*)
    (list #'bad-possessive? *bad-possessive-msg*)
    (list #'bad-pu? *bad-pu-msg*)
    (list #'bad-flat-mod? *bad-flat-mod-msg*)
    (list #'bad-equal? *bad-equal-msg*)
    (list #'conservative-bad-sent-reifier? *conservative-bad-sent-reifier-msg*)
    (list #'bad-noun-pp? *bad-noun-pp-msg*)
    (list #'bad-verb-sent? *bad-verb-sent-msg*)
    (list #'bad-aux-before-arg? *bad-aux-before-arg-msg*)
    (list #'bad-pasv? *bad-pasv-msg*)
    (list #'bad-verb-args? *bad-verb-args-msg*)
    (list #'suspicious-locative? *suspicious-locative-msg*)
    (list #'suspicious-do? *suspicious-do-msg*)
    (list #'suspicious-will? *suspicious-will-msg*)
    (list #'bad-name-decomp? *bad-name-decomp-msg*)
    (list #'bad-sent-term? *bad-sent-term-msg*)
    (list #'bad-adv-a-arg? *bad-adv-a-arg-msg*)
    (list #'old-adj-mod? *old-adj-mod-msg*)
    ))

;; Same as above but run on raw formulas (before preprocessing).
(defparameter *raw-bad-pattern-test-pairs*
  (list
    (list #'bad-single-bracket? *bad-single-bracket-msg*)
    (list #'bad-sub? *bad-sub-msg*)
    (list #'bad-rep? *bad-rep-msg*)
    (list #'bad-qt-attr? *bad-qt-attr-msg*)
    (list #'bad-rel-sent? *bad-rel-sent-msg*)
    (list #'bad-voc? *bad-voc-msg*)
    ))

