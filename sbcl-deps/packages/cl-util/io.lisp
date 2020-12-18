
(in-package :util)

(defun read-file-lines (filename)
  "Reads a file line by line and return a list of strings."
  (labels
    ((helper (in acc)
       (multiple-value-bind (line endp)
           (read-line in nil)
         (if endp
           (cons line acc)
           (helper in (cons line acc))))))
    (with-open-file (stream filename)
      (reverse (remove-if #'null (helper stream '()))))))

(defun read-file-lines2 (filename)
  "Reads a file line by line and return a list of strings.
  Done in a loop so you won't get a stack overflow even with bad compiler
  parameters."
  (with-open-file (fh filename)
    (let ((done nil)
          (acc '()))
      (loop while (not done)
            do (multiple-value-bind (line done) 
                   (read-line fh nil)
                 (push line acc)))
      (reverse acc))))

(defun read-all-from-stream (s)
  "Reads all s-expressions from a character stream until exhausted. 
  It will raise an error if the stream does not represent a sequence of
  s-expresssions."
  (labels
    ((helper (in acc)
       (let ((expr (read in nil)))
         (if (null expr)
           acc
           (helper in (cons expr acc))))))
    (reverse (helper s nil))))

(defun read-all-from-file (filename)
  "Reads all s-expressions from the given file until exhausted.
  It will raise an error if the file does not contain a sequence of valid
  s-expresssions."
  (with-open-file (s filename)
    (read-all-from-stream s)))

;; TODO: move this to string-ops.lisp or maybe stream.lisp once this directory
;; is packaged and internal dependencies are better managed.
(defun read-all-from-string (str)
  "Reads all s-expressions from the given string.
  Raises an error if the string does not represent a series of valid
  s-expressions. Same as READ-ALL-FROM-FILE, but for strings."
  (with-input-from-string (s str)
    (read-all-from-stream s)))

(defun write-to-file (str filename)
  "Writes a string to a file."
  (declare (type simple-string str))
  (with-open-file (fh filename :direction :output)
    (format fh str)))

(defun write-list-to-file (lst filename &optional (sep "~%"))
  "Writes a list to a file.
  Depends on write-to-file."
  (write-to-file (list-to-string lst sep) filename))

(defun princln (x)
  "CL version of 'println' in Java.
  The name PRINCLN is meant to reflect the CL naming conventions for prints."
  (princ x)
  (format t "~%"))

