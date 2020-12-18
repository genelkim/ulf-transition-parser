'''
categorize amr; generate linearized amr sequence
'''
import string
from .amr_graph import AMR
from .date_extraction import *
from collections import defaultdict
import re
re_symbols = re.compile("['\".\'`\-!#&*|\\/@=\[\]_]")

def get_amr(concept_line_reprs, category_map):
    """Builds AMR graph from the Hypothesis class (concept_line_reprs) and
    the annotated sentence info (category_map).
    """
    def register_var(token):
        num = 0
        while True:
            currval = ('%s%d' % (token[0], num)) if token[0] in string.ascii_letters else ('a%d' % num)
            if currval in var_set:
                num += 1
            else:
                var_set.add(currval)
                return currval

    def children(nodelabel, node_to_children):
        ret = set()
        stack = [nodelabel]
        visited_nodes = set()
        while stack:
            curr_node = stack.pop()
            if curr_node in visited_nodes:
                continue
            visited_nodes.add(curr_node)
            ret.add(curr_node)
            if curr_node in node_to_children:
                ret |= node_to_children[curr_node]
                for child in node_to_children[curr_node]:
                    stack.append(child)
        return ret

    # In case there is a loop, root is a node that all parents can be visited?
    def valid_root(nodelabel, node_to_children, parents):
        canbevisited = children(nodelabel, node_to_children)
        return len(parents-canbevisited) == 0

    def is_const(s):
        const_set = set(['interrogative', 'imperative', 'expressive', '-'])
        return s.isdigit() or s in const_set or isNumber(s)

    def value(s, delimiter="_"): #Compute the value of the number representation
        number = 1

        for v in s.split(delimiter):
            if isNumber(v):
                if '.' in v:
                    if len(s.split(delimiter)) == 1:
                        return v
                    number *= float(v)
                else:
                    v = v.replace(",", "")
                    number *= int(v)
            else:
                v = v.lower()
                if v in quantities:
                    # assert v in quantities, v
                    number *= quantities[v]
                number = int(number)
        return str(number)

    def analyse_subgraph(subgraph_repr):
        # subgraph_repr = subgraph_repr.strip()
        if subgraph_repr[0] == "(": # Look until we find the pairing )
            root_repr = subgraph_repr[1:].split()[0]
            assert not is_const(root_repr)
            nodelabel = register_var(root_repr)
            #if not init:
            amr.node_to_concepts[nodelabel] = root_repr
            _ = amr[nodelabel]
            next_offset = 1 + len(root_repr)
            while subgraph_repr[next_offset] != ")":  # Search for matching par.
                if subgraph_repr[next_offset:next_offset+2] != " :":
                    print("Wrong subgraph reprensentation: %s" % (subgraph_repr))
                    assert subgraph_repr[next_offset:next_offset+2] == " :"
                relation = subgraph_repr[next_offset+2:].split()[0]
                rel_length = len(relation)
                next_offset += (3+rel_length)
                consumed_length, _, child_v = analyse_subgraph(subgraph_repr[next_offset:])
                child = tuple([child_v])
                next_offset += consumed_length
                amr._add_triple(nodelabel, relation, child)
            return next_offset + 1, root_repr, nodelabel
        else: # A single concept.
            offset = 0
            while subgraph_repr[offset] != " " and subgraph_repr[offset] != ")":
                offset += 1
            curr_concept = subgraph_repr[:offset]
            if not is_const(curr_concept) and not isNum(curr_concept):
                nodelabel = register_var(curr_concept)
                amr.node_to_concepts[nodelabel] = curr_concept
                _ = amr[nodelabel]
            else:
                nodelabel = curr_concept
            return offset, curr_concept, nodelabel

    var_set = set()
    nodeid_to_label = {}
    label_to_rels = defaultdict(list)
    visited = set()
    vertices = set()
    amr = AMR()

    connected_nodes = set()

    label_to_children = defaultdict(set)
    label_to_parents = defaultdict(set)

    and_structure = (len(category_map) > 0) and (category_map[0] == "AND||AND")
    node_labels = []

    for (concept_idx, concept_line) in enumerate(concept_line_reprs):
        concept_l, rel_str, par_str = concept_line
        map_repr = category_map[concept_idx]
        assert concept_l != "NONE"

        if concept_l[:3] == "NE_" and "NEG" not in concept_l:
            if concept_l == "NE":
                root_repr = "person"
            else:
                assert concept_l[:3] == "NE_", concept_l
                root_repr = concept_l[3:]
            nodelabel = register_var(root_repr)
            nodeid_to_label[concept_idx] = nodelabel
            if map_repr != "NONE":
                tok_repr = map_repr.split("||")[0]
                wiki_label = map_repr.split("||")[1]

                l = "wiki"
                child = tuple([("\"%s\"" % wiki_label)])
                amr._add_triple(nodelabel, l, child)

                l = "name"
                name_v = register_var("name")
                amr.node_to_concepts[name_v] = "name"

                child = tuple([name_v])
                amr._add_triple(nodelabel, l, child)

                for (op_index, s) in enumerate(tok_repr.split("_")):
                    l = "op%d" % (op_index+1)
                    child = tuple([("\"%s\"" % s)])
                    amr._add_triple(name_v, l, child)

        elif concept_l == "NUMBER":
            tok_repr = map_repr.split("||")[0]
            value_repr = value(tok_repr)
            nodelabel = value_repr
            root_repr = value_repr
            # assert rel_str == "NONE", "NUMBER with outgoing edges %s: %s" % (value_repr, rel_str)
            if (par_str == "NONE"):
                nodelabel = register_var("number")
                l = "quant"
                child = tuple([value_repr])
                amr._add_triple(nodelabel, l, child)
                nodeid_to_label[concept_idx] = nodelabel
                amr.node_to_concepts[nodelabel] = "number"
            else:
                nodeid_to_label[concept_idx] = value_repr # Should be able to add this node from its parent.
            rel_str = "NONE"
        elif concept_l[:5] == "MULT_" or concept_l[:4] == "NEG_":
            # TODO: add the module for
            subgraph_repr = map_repr.split("||")[1]
            _, root_repr, nodelabel = analyse_subgraph(subgraph_repr)
            nodeid_to_label[concept_idx] = nodelabel
            # continue
            # if "MULT" in concept_l:
            #     subgraph_repr = map_repr.split("||")[1]
            # else: # Negation
            #     subgraph_repr = "%s polarity:-" % concept_l[4:]

            # root_repr = subgraph_repr.split()[0]
            # rel_tuple = [tuple(curr_rel.split(":")) for curr_rel in subgraph_repr.split()[1:]]

            # nodelabel = register_var(root_repr)
            # nodeid_to_label[concept_idx] = nodelabel
            # amr.node_to_concepts[nodelabel] = root_repr

            # for (rel, tail_concept) in rel_tuple:
            #     child_v = tail_concept
            #     if not is_const(tail_concept):
            #         child_v = register_var(tail_concept)
            #     child = tuple([child_v])
            #     amr._add_triple(nodelabel, rel, child)
            #    if not is_const(tail_concept):
            #        amr.node_to_concepts[child_v] = tail_concept
            #        _ = amr[child_v]

        elif concept_l == "DATE": #Date entities
            root_repr = 'date-entity'
            nodelabel = register_var(root_repr)
            amr.node_to_concepts[nodelabel] = root_repr #Newly added
            nodeid_to_label[concept_idx] = nodelabel
            if map_repr != "NONE":
                tok_repr = map_repr.split("||")[0]
                date_rels = dateRepr(re.sub(re_symbols, " ", tok_repr).split())
                for l, subj in date_rels:
                    child = tuple([subj])
                    amr._add_triple(nodelabel, l, child)

        else:
            root_repr = concept_l
            if not is_const(concept_l):
                nodelabel = register_var(concept_l)
                assert not concept_idx in nodeid_to_label
                nodeid_to_label[concept_idx] = nodelabel
            else:
                nodeid_to_label[concept_idx] = concept_l
                nodelabel = concept_l

        # All the incoming and outgoing edges are to the root concept.
        if is_const(root_repr):
            if (rel_str != "NONE") or (par_str == "NONE"):
                nodelabel = register_var(root_repr)
                nodeid_to_label[concept_idx] = nodelabel
                amr.node_to_concepts[nodelabel] = root_repr
                vertices.add(nodelabel)
                _ = amr[nodelabel]
                if rel_str != "NONE": #Save the relations for further processing
                    label_to_rels[nodelabel] = rel_str
                if par_str == "NONE": #Roots are nodes without parents
                    amr.roots.append(nodelabel)
            else:
                vertices.add(root_repr)
        else:
            vertices.add(nodelabel)

            if (not is_const(root_repr)) and (not nodelabel in amr.node_to_concepts):
                amr.node_to_concepts[nodelabel] = root_repr
                _ = amr[nodelabel] #Put current node in the AMR

            if rel_str != "NONE": #Save the relations for further processing
                label_to_rels[nodelabel] = rel_str
            if par_str == "NONE": #Roots are nodes without parents
                amr.roots.append(nodelabel)
        node_labels.append(nodelabel)

    if and_structure:
        amr.roots = [node_labels[0]]
        and_l = node_labels[0]
        idx = 0
        for (i, l) in enumerate(node_labels[1:]):
            if l in amr.node_to_concepts and amr.node_to_concepts[l] == "and":
                continue
            idx += 1
            rel_str = "op%d" % idx
            label_to_children[and_l].add(l)
            label_to_parents[l].add(and_l)
            amr._add_triple(and_l, rel_str, tuple([l]))
    else:

        # Then we process the relations between all the concepts in the AMR.
        # Since we have registered all vertices, we can make edges
        # print(str(label_to_rels))
        for nodelabel in label_to_rels:
            visited_rels = set()
            rels = label_to_rels[nodelabel]
            try:
                triples = rels.split("#")
            except:
                print(nodelabel)
                print(rels)
                sys.exit(1)
            for rs in reversed(triples):
                l = rs.split(":")[0]
                assert("UNKNOWN" not in l)
                # if "UNKNOWN" in l:
                #     continue
                concept_idx = int(rs.split(":")[1])
                taillabel = nodeid_to_label[concept_idx]
                if not is_const(taillabel):
                    if l in visited_rels and taillabel in connected_nodes:
                        continue
                visited_rels.add(l)
                connected_nodes.add(taillabel)

                label_to_children[nodelabel].add(taillabel)
                label_to_parents[taillabel].add(nodelabel)

                amr._add_triple(nodelabel, l, tuple([taillabel]))

    for root_label in amr.roots:
        visited |= children(root_label, label_to_children)

    unfound = vertices - visited

    for label in unfound:
        if label in visited:
            continue
        parents = set()
        if label in label_to_parents:
            parents = label_to_parents[label]
        if valid_root(label, label_to_children, parents):
            amr.roots.append(label)
            visited |= children(label, label_to_children)

    try:
        assert len(visited) == len(vertices)
    except:
        print(vertices - visited)
        print(visited - vertices)
        print(concept_line_reprs)
        print(map_repr)
        sys.exit(1)

    #print("in get_amr, amr: {}".format(amr)) 
    return amr

