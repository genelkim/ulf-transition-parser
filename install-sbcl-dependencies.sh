# Make sure the local-projects directory exists.
mkdir -p ~/quicklisp/local-projects
# Update asdf.lisp file.
rm ~/quicklisp/asdf.lisp
cp sbcl-deps/asdf.lisp ~/quicklisp/
# Copy over ULF-specific dependencies.
cp -r sbcl-deps/packages/* ~/quicklisp/local-projects/

