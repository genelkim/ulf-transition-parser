;; Reachability Indexing

(in-package :util)

(defun reachable (node1 node2 inds ints)
  "Returns T if NODE1 is reachable via a downward path from NODE2;
   INDS must evaluate to the name of the indicator used to store
        the list of indices of a node in the graph to which NODE1
        and NODE2 belong;
   INTS must evaluate to the name of the indicator used to store
        the list of intervals of a node in the graph to which NODE1
        and NODE2 belong."
  (prog (indices intervals)
        (setq indices (get node1 inds))
        (setq intervals (get node2 ints))
        (dolist (i indices)
          (setq intervals
                (member-if #'(lambda (int)
                               (<= i (cdr int))) intervals) )
          (if (null intervals)
              (return-from reachable nil))
          (if (not (< i (caar intervals)))
              (return-from reachable t)))))


(defun depth-label-graph (g in out inds ints vis)
  "The graph G is given as a list of vertices, where vertices are
   atoms with properties named by IN and OUT. The values corresponding
   to these properties are lists of vertices from which edges are
   incident, and to which they are directed, respectively. IN and OUT
   are pre-existing properties, while INDS, INTS, and VIS are created
   in this program. VIS IN, OUT, INDS, INTS, VIS are all *names* of
   atoms to be used as indicators, not the indicator atoms themselves.

   The result is that all vertices have properties named INDS and INTS
   (indices and intervals), such that vertex m is a proper ancestor of
   n iff some index in the INDS list of n is contained in some interval
   in the INTS list of m. Intervals are represented as dotted pairs of
   positive integers, e.g., an INTS list might be ((1 . 3) (5 . 5) (8 . 9)).
   VIS ('visited') is used as a temporary flag. The VIS flags are reset
   to NIL at the end."
  (let ((roots (remove-if #'(lambda (x) (get x in)) g))
        (cyclic nil) (cyclic-index nil)
        (i 1) ; 1st depth-first index that is to be used
        (h nil)) ; a set of vertices of G

    ;; Erase the inds and  ints values of all vertices (i.e., new
    ;; start; vis values are reset to nil at the end):
    (dolist (x g)
      (setf (get x inds) nil)
      (setf (get x ints) nil))

    ;; Perform depth-first searches from the roots, labelling *edges*
    ;; (rather than vertices, at this point) with depth-first indices:
    (dolist (r roots)
      (setq cyclic-index (depth-label-root r i in out inds ints vis))
      (setq cyclic (or cyclic (car cyclic-index)))
      (setq i (cdr cyclic-index)) )

    ;; Similarly process any remaining unvisited vertices (on rootless
    ;; cycles):
    (setq h (remove-if #'(lambda (x) (get x vis)) g))
    (setq cyclic (or cyclic h)); unvisited vertices imply a cycle
    (dolist (x h)
      (when (null (get x vis))
        (setq cyclic-index (depth-label-root x i in out inds ints vis))
        (setq i (cdr cyclic-index)) ))

    ;; Do additional interval propagation for cyclic graphs:
    (when cyclic
      (setq h g)
      (do (x) ((null h))
        (setq x (pop h))
        (dolist (y (get x in))
          (when (not (subsumes-intervals y x ints))
            (overlay-intervals x y ints)
            (push y h) ))))

    ;; Project edge labels onto vertices (as their inds-values), and erase
    ;; the edge labels (restoring the out-lists to their original values):
    (dolist (x g)
      (setf (get x vis) nil)
      (dolist (y (get x out))
        (add-to-indices (cdr y) (car y) inds))
      (setf (get x out) (mapcar #'car (get x out))))))


(defun depth-label-root (r i in out inds ints vis)
  "R is the start vertex for the depth-first labelling; it is assumed
   not to have been visited yet (its vis value is nil); I is the depth-
   first edge index that is to be assigned to the first unlabelled edge
   encountered that leads to a terminal or already visited vertex.

   The result is that all vertices which are reachable from R and, to begin
   with, are not flagged as visited, are marked as visited, and all edges
   that are reachable from r and, to begin with, are not indexed become
   indexed with depth-first numbers.

   The value returned is a pair (CYCLIC J) where CYCLIC = T if a cycle was
   encountered in the depth-first search, and J is the next index that can
   be assigned to an edge, i.e., it is the highest index that was assigned,
   plus 1."

  (let ((j i)
        (cyclic nil)
        cyclic-index)
    (setf (get r vis) t) ; Mark R as visited

    ;; Iterate through the outgoing edges from R:
    (do ((children (get r out)) x)
        ((null children))
      (setq x (car children))

      ;; If X is nonterminal and marked as visited, but is still missing
      ;; edge-indices for some of its out-list members (exiting edges),
      ;; then cyclic := T:
      (if (and (get x out)
               (get x vis)
               (find-if #'atom (get x out)) )
          (setq cyclic t) )

      ;; If X is terminal (i.e., has an empty out-list) or is already
      ;; marked visited, then  pair X on R's out-list with edge-index J:
      (if (or (null (get x out)) (get x vis))
          (rplaca children (cons x j))

        ;; Otherwise, continue depth-first:
        (progn (setq cyclic-index
                     (depth-label-root x j in out inds ints vis) )
               (setq cyclic (or cyclic (car cyclic-index)))
               (setq j (cdr cyclic-index))
               (rplaca children (cons x j)) ))
      (add-to-intervals j r ints)
      (incf j)
      (overlay-intervals x r ints)
      (pop children) )
    (cons cyclic j)))


(defun overlay-intervals (x y ints)
  "X is a child of Y; the intervals of X are merged into the intervals
   of Y, maintaining maximal, ascending, disjoint integer intervals;
   Besides changing the INTS property of Y, the function also returns
   the merged intervals."
  (let ((intx (get x ints))
        (inty (get y ints))
        result) ; lists of intervals

      ;; If INTX is empty, no action is required, and if INTY is empty
      ;; then the INTS list of Y should be reset to the INTS list of X:
      (if (null intx)
          (return-from overlay-intervals inty)
        (if (null inty)
            (return-from overlay-intervals (setf (get y ints) intx))))

      ;; Work through the interval lists in parallel, merging them:
      (do (ix iy i j) ((or (null intx) (null inty)))
        (setq ix (car intx) iy (car inty))
        (if (and (> (+ (cdr iy) 2) (car ix))   ; are the front intervals
                 (> (+ (cdr ix) 2) (car iy)) ) ; adjacent or overlapping?
            (progn (pop intx) (pop inty)
                   ;; Form combined interval (I . J):
                   (setq i (min (car ix) (car iy)))
                   (setq j (max (cdr ix) (cdr iy)))

                     ;; If INTX is empty or J+1 < 1st(1st(INTX)),
                     ;; push this interval onto INTX, else onto INTY:
                     (if (or (null intx) (< (+ j 1) (caar intx)))
                         (push (cons i j) intx)
                         (push (cons i j) inty) ))

              ;; For nonoverlapping front intervals, pop the lower
              ;; off its list and push it onto RESULTS:
              (if (< (car ix) (car iy)) (push (pop intx) result)
                                        (push (pop inty) result) )))

      ;; Combine the merged intervals with the remainder (on INTX or INTY):
      (setq result (append (reverse result) (if intx intx inty)))
      (setf (get y ints) result) ; reset the INTS property of Y
      result))


(defun add-to-indices (i r inds)
  "Add index I to the INDS list of vertex R, keeping the list in ascending
   order."
  (let ((ii (get r inds)) (result nil))
    (do () ((or (null ii) (<= i (car ii)))) (push (pop ii) result))
    (if (and ii (= i (car ii)))
        (return-from add-to-indices (get r inds)) )
    (setq result (append (reverse result) (cons i ii)))
    (setf (get r inds) result)
    result))


(defun add-to-intervals (i r ints)
  "Add integer I to the INTS list of vertex R, maintaining maximal,
   ascending, disjoint intervals."

  (let ((intr (get r ints)) result tail j k)
    (do () ((or (null intr) (< (- i 2) (cdar intr))))
      (push (pop intr) result) )
    (if (or (null intr) (< (+ i 1) (caar intr)))
        (setq tail (cons (cons i i) intr))
      (progn (setq j (caar intr) k (cdar intr))
             (if (= (+ i 1) j)
                 (setq tail (cons (cons i k) (cdr intr)))
               (if (< (- i 1) k); if so, no change to INTS of R
                   (return-from add-to-intervals (get r ints))
                 (if (and (cdr intr) (= (+ i 1) (caadr intr)))
                     (setq tail (cons (cons j (cdadr intr))
                                      (cddr intr) ))
                   (setq tail (cons (cons j i) (cdr intr))) )))))
    (setq result (append (reverse result) tail))
    (setf (get r ints) result)
    result))



(defun subsumes-intervals (x y ints)
  "Return T iff the INTS-intervals of X contain all the INTS-intervals of Y."

  (let ((intx (get x ints))
        (inty (get y ints)))
    (do (i1 i2 j1 j2) ((null inty))
      (if (null intx)
          (return-from subsumes-intervals nil))
      (setq j1 (cdar intx) i2 (caar inty))
      (if (< j1 i2)
          (pop intx)
        (progn (setq i1 (caar intx))
               (if (< i2 i1)
                   (return-from subsumes-intervals nil))
               (setq j2 (cdar inty))
               (if (< j1 j2)
                   (return-from subsumes-intervals nil))
               (pop inty))))
    t))



;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

;; The following is an example of a routine that uses a call to
;; depth-label-graph to set up reachability indexing for a given
;; graph -- in this case, the graph of syntactic types used by a
;; parser.


;; A simple example, using 'in and 'out and IN and OUT indicators, and
;; 'indices and 'intervals as INDS and INTS indicators:

; [1] USER(3): (setf (get 'a 'out) '(d b))
; [1] USER(4): (setf (get 'b 'out) '(e c))
; [1] USER(5): (setf (get 'c 'out) '(g))
; [1] USER(6): (setf (get 'd 'out) '(f))
; [1] USER(7): (setf (get 'e 'out) '(f g))
; [1] USER(8): (setf (get 'g 'out) '(h i))
; [1] USER(9): (setf (get 'b 'in) '(a))
; [1] USER(10): (setf (get 'c 'in) '(b))
; [1] USER(11): (setf (get 'd 'in) '(a))
; [1] USER(12): (setf (get 'e 'in) '(b))
; [1] USER(13): (setf (get 'f 'in) '(d e))
; [1] USER(14): (setf (get 'g 'in) '(e c))
; [1] USER(15): (setf (get 'h 'in) '(g))
; [1] USER(16): (setf (get 'i 'in) '(g))
; [1] USER(17): (depth-label-graph '(a b c d e f g h i) 'in 'out 'indices 'intervals
; 'vis)
; NIL
; [1] USER(18): (reachable 'a 'e 'indices 'intervals)
; NIL
; [1] USER(19): (reachable 'e 'a 'indices 'intervals)
; T
; [1] USER(20): (reachable 'h 'a 'indices 'intervals)
; T
; [1] USER(21): (reachable 'i 'c 'indices 'intervals)
; T
; [1] USER(22): (mapcar #'(lambda (x) (list (get x 'indices) (get x 'intervals)))
;                 '(a b c d e f g h i))
; ((NIL ((1 . 10))) ((10) ((3 . 9))) ((9) ((4 . 5) (8 . 8)))
;  ((2) ((1 . 1))) ((7) ((3 . 6))) ((1 3) NIL) ((6 8) ((4 . 5)))
;  ((4) NIL) ((5) NIL))

