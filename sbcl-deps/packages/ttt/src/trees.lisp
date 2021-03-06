(in-package :ttt)
;; trees.lisp
(defclass tree ()
  ((children :accessor children :type (or list symbol number))
   (to-expr :accessor to-expr :type (or list symbol number))
   (nchildren :accessor nchildren :type fixnum)
   (height   :accessor height :type fixnum)
   (keys     :accessor keys)
   (to-subtrees :accessor to-subtrees)
   (parent :initform nil :accessor parent)
   (parent-idx :initform nil :accessor parent-idx)
   (dfs-order :accessor dfs-order)))
(declaim (ftype (function (tree) fixnum) height nchildren)
         (ftype (function (tree) (or list symbol number)) to-expr)
         (ftype (function (tree) (or list symbol number)) children))

(defmethod leaf? ((node tree))
  (not (consp (children node))))

(defun build-tree (expression &key (root t) (index-subtrees t))
  "Convert a Lisp expression into a nodal tree.
   This function should be optimized, because it will always
   be called once per tree expression to be matched.
   (Eg., 25k times for Brown, 5m times for BNC)"
  (let ((node (make-instance 'tree)))
    (setf (keys node) (extract-keys expression :no-ops t))
    (setf (to-expr node) expression)
    (if (atom expression)
        (setf (children node) expression
              (nchildren node) 0
              (height node) 0)
        (let (tmp-node (n -1))
          (declare (type fixnum n))
          (setf (children node) nil)
          (dolist (item expression)
            (setf tmp-node (build-tree item :root nil :index-subtrees nil))
            (setf (parent tmp-node) node)
            (setf (parent-idx tmp-node) (incf n))
            (push tmp-node (children node)))
          (setf (children node) (nreverse (children node)))
          (setf (height node)
                (the fixnum
                     (1+ (the fixnum
                              (apply #'max (mapcar #'height (children node)))))))
          (setf (nchildren node) (length (children node)))))
    (when root
      (let ((to-subtrees (if index-subtrees (make-index)))
            (dfs-order 0)
            stack)
        (declare (type fixnum dfs-order))
        (push node stack)
        (loop while stack do
             (let ((current (pop stack)))
               (setf (dfs-order current) (incf dfs-order))
               (if index-subtrees
                   (dolist (key (keys current))
                     (add-to-index key current to-subtrees)))
               (unless (leaf? current)
                 (dolist (child (children current))
                   (push child stack)))))
        (when index-subtrees
          (setf (to-subtrees node) to-subtrees))))
    node))

(defmethod update-dfs-order ((root tree))
  "recomputes dfs-order for all subtrees.
   particularly useful after transductions modify a tree."
  (let ((dfs-order -1)
        (stack (list root)))
    (declare (type fixnum dfs-order))
    (loop while stack do
         (let ((current (pop stack)))
           (setf (dfs-order current) (incf dfs-order))
           (unless (leaf? current)
             (dolist (c (children current))
               (push c stack)))))))
(defmethod update-subtree-index ((root tree))
  "recomputes to-subtrees index for a tree.
   particularly useful after transductions modify a tree.
   assumes keys are valid."
  (let ((stack (list root))
        (indexed-subtrees (make-index)))
    (loop while stack do
         (let ((current (pop stack)))
           (dolist (key (keys current))
             (add-to-index key current indexed-subtrees))
           (unless (leaf? current)
             (dolist (c (children current))
               (push c stack)))))
    (setf (to-subtrees root) indexed-subtrees)))

