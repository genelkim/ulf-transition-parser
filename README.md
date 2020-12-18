# TULF

This is a Transition-based Unscoped Episodic Logical Formula parser.


## Setting up the environment

This code was developed and tested using Python 3.6 and PyTorch 1.2. All other Python dependencies are listed in requirements.txt. We recommend using Anaconda:

```
conda create -n tulf python=3.6
conda activate tulf
pip install -r requirements.txt
```

The data preparation requires Java 8 to run the Stanford CoreNLP toolkit. Get the version of the Stanford CoreNLP used in our experiments by running `./download_stanford_corenlp.sh`. Otherwise, download and unzip the Stanford CoreNLP version of your choice in the `tools/` directory and modify the `CORENLP_VER` variable in `preprocess-and-run-oracle.sh` to the name of the folder extracted from the zip file.

Download the embeddings.
```
./ulfctp/scripts/download_embeddings.sh
```

Type composition checking and the conversion back from ULF-AMR to regular ULF format relies on Common Lisp libraries, which were tested on SBCL. 

```
# Install SBCL
conda install -c conda-forge sbcl
```

The SBCL pacakges are not necessary for all versions of the parser. If you are not interested in running the type composition models or getting the base ULF forms of the parses, skip this installation step and follow the instructions to run the parser without using the SBCL dependencies. The SBCL still needs to be installed (the step above) since a connection is made at startup. To install SBCL packages, first get [Quicklisp](https://www.quicklisp.org/beta/) (a common lisp package manager) and then:

```
> ./install-sbcl-dependencies.sh
> # Manually compile TTT.
> cd ~/quicklisp/local-projects/ttt/src
> sbcl
* (load "load")
```

You might find it necessary to go into the other lisp dependencies (they can be found in `~/quicklisp/local-projects/` after running the command above) and force the projects to compile by loading them manually as well. This can be done by going into the folder with the `load.lisp` file and running the following.
```
> sbcl

...[startup messages]...
* (load "load")
...
```
If the loading script crashes, select the option to recompile and try again.

## Running the code

```
# Run the preprocessing and oracle.
# Update the CORENLP_VER in preprocess-and-run-oracle.sh if you have a version of corenlp different from that which is downloaded by download_stanford_corenlp.sh
./preprocess-and-run-oracle.sh

# Train and test the model with type composition filtering
python3 -u -m ulfctp.commands.train ulfctp/params/[param file]
# Train and test the model without type composition filtering
python3 -u -m ulfctp.commands.train ulfctp/params/[param file] --tuning_off
```

Parameter files:
- ulfctp/params/emnlp-full.yaml (full model)
- ulfctp/params/emnlp-best.yaml (best model)

