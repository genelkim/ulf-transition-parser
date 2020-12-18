
# Returns whether the given ulf atom is a name.
# | ..| or "| ..|"
# NB: this is the same as bar_wrapped but a space following the first bar.
def is_ulf_atom_name(atom):
  return is_bar_wrapped(atom) and \
      ((atom[0] == "|" and atom[1] == " ") or
          (atom[1] == "|" and atom[2] == " "))

# Returns whether the given ulf atom is wrapped in bars.
# |..| or "|..|"
def is_bar_wrapped(atom):
  return atom[-1] == "|" or \
    (
      len(atom) > 3 and
      atom[-2] == "|" and
      atom[-1] == "\""
    )

def strip_bars(atom):
  # Names have bars and spaces, which should be stripped.
  if is_ulf_atom_name(atom):
    if atom[-1] == "|":
      return atom[1:-1].strip()
    else:
      return atom[2:-2].strip()
  elif is_bar_wrapped(atom):
    if atom[-1] == "|":
      stripped = atom[1:-1]
    else:
      stripped = atom[2:-2]
    # If all but the suffix is numerical, this is a number so we can deal with this.
    if ".".join(stripped.split(".")[:-1]).replace(".", "", 1).isdigit():
      return stripped
    elif stripped == "\\\"" or stripped in ["(", ")"]:
      # For quotes and parentheses, just return the whole thing, since we
      # already have special handling for this case.
      return atom
    else:
      raise Exception("Unknown bar wrapped symbol: {}".format(atom))
  else:
    # No bars to remove
    return atom

# TODO: put this in some utility file...
# Splits a ULF atom to the string, the suffix, and whether it is a name.
# A simple name |John| returns a "pro" suffix with name=True.
def split_ulf_atom(atom):
  is_name = is_ulf_atom_name(atom)
  unbarred = strip_bars(atom)
  dotsplit = unbarred.split(".")
  if len(dotsplit) == 1:
    return (unbarred, "PRO" if is_name else None, is_name)
  else:
    return (".".join(dotsplit[:-1]), dotsplit[-1], is_name)

