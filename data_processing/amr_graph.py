#!/usr/bin/env python2.7
import sys, re
from amr_parser import *
from amr_fragment import *
from collections import deque, defaultdict
from re_utils import extract_patterns, delete_pattern
from constants import *
from date_extraction import isNumber
import utils

class AMREdge(object):
    def __init__(self, label, graph, h_node, t_node = None):
        self.label = label
        self.head = h_node
        self.tail = t_node
        self.is_coref = False #Denoting the tail node is a reentrance
        self.graph = graph

    def __str__(self):
        return self.label

    def set_coref(self, val):
        self.is_coref = val

    def isLeaf(self):
        return self.tail is None

'''There is some difference between leaf variables and the constant'''
class AMRNode(object):
    def __init__(self, graph, is_const = False):
        self.v_edges = []   #edges for relations
        self.c_edge = None #edge for node variable
        self.p_edges = []
        self.is_const = is_const
        self.use_quote = False
        self.has_reentrance = False
        self.graph = graph

    # for traversal
    def get_unvisited_children(self, visited, is_sort = True):
        children = []
        for i in range(len(self.v_edges)):
            id, n = self.get_child(i)
            if id not in visited:
                children.append((id, n.node_str_nosuffix(), n)) # (n, concept, node)
        if is_sort == True:
            return sorted(children, key=lambda tup: tup[1])
        else:
            return children

    # useful for named entity: (n/name op1 op2 ...)
    def get_children_str(self):
        return ' '.join([self.get_child(i)[1].node_str() for i in range(len(self.v_edges))])

    def get_child(self, i):
        id = self.graph.edges[self.v_edges[i]].tail
        return (id, self.graph.nodes[id])

    def get_children(self):
        children = []
        for i in range(len(self.v_edges)):
            children.append(self.get_child(i))
        return children

    #Return all outgoing edges together with the concept ids
    def childTriples(self):
        triples = []
        for edge_index in self.v_edges:
            curr_edge = self.graph.edges[edge_index]
            triples.append((curr_edge.label, curr_edge.tail))
        return triples

    def parentTriples(self):
        triples = []
        for edge_index in self.p_edges:
            curr_edge = self.graph.edges[edge_index]
            triples.append((curr_edge.label, curr_edge.head))
        return triples

    # more node types
    def is_negative_polarity(self):
        return self.node_str() == '-' and self.graph.edges[self.p_edges[0]].label == 'polarity'

    def search_polarity(self):
        for edge_idx in self.v_edges:
            if self.graph.edges[edge_idx].label == "polarity":
                return edge_idx
        return -1

    def is_var_node(self):
        return not self.is_const

    def is_location(self):
        pass

    def is_date(self):
        pass

    def set_quote(self, val):
        self.use_quote = val

    def set_reenter(self, val):
        self.has_reentrance = val

    def edge_set(self):
        for edge_index in self.p_edges:
            yield edge_index
        yield self.c_edge
        for edge_index in self.v_edges:
            yield edge_index

    def is_pred(self):
        node_l = self.node_label()
        if node_l in self.graph.dict:
            concept_l = self.graph.dict[node_l]
            if '-' in concept_l:
                parts = concept_l.split('-')
                if len(parts) >= 2:
                    return len(parts[-1]) == 2 and parts[-1][0] in '0123456789' and parts[-1][1] in '0123456789'
        return False

    def is_leaf(self):
        return len(self.v_edges) == 0

    def add_incoming(self, edge):
        self.v_edges.append(edge)

    def set_const_edge(self, edge):
        self.c_edge = edge

    def add_parent_edge(self, edge):
        self.p_edges.append(edge)

    def node_str(self):
        node_l = self.node_label()
        return self.graph.dict[node_l] if node_l in self.graph.dict else node_l

    def node_str_nosuffix(self):
        return self.node_str().split('-')[0]

    def node_label(self):
        return self.graph.edges[self.c_edge].label

    def is_date_entity(self):
        v_label_set = set([self.graph.edges[x].label for x in self.v_edges])
        return 'date-entity' in v_label_set

    def is_named_entity(self): #Added for the new labeled data
        v_label_set = set([self.graph.edges[x].label for x in self.v_edges])
        if 'name' in v_label_set:
            try:
                assert 'wiki' in v_label_set
            except:
                #print 'ill-formed entity found'
                #print str(self.graph)
                return False
            return True
        return False

    def getWiki(self): #For a named entity, get its wiki label
        assert self.is_named_entity()
        for edge_index in self.v_edges:
            curr_edge = self.graph.edges[edge_index]
            if curr_edge.label == 'wiki':
                wiki_node = self.graph.nodes[curr_edge.tail]
                return wiki_node.node_str()

        return '-'

    #Assume it to be entity mention
    def namedEntityMention(self):
        assert self.is_named_entity()
        all_ops = []
        for edge_index in self.v_edges:
            curr_edge = self.graph.edges[edge_index]
            if curr_edge.label != 'name':
                continue

            name_node = self.graph.nodes[curr_edge.tail]
            assert name_node.v_edges, str(self.graph)
            for op_edge_index in name_node.v_edges:
                curr_op_edge = self.graph.edges[op_edge_index]
                if curr_op_edge.label[:2] != 'op':
                    print('Extra operand: %s' % curr_op_edge.label)
                    continue

                op_index = int(curr_op_edge.label[2:])
                op_name = self.graph.nodes[curr_op_edge.tail].node_str()
                all_ops.append((op_index, op_name))
        all_ops = sorted(all_ops, key=lambda x: x[0])
        return [y for (x, y)  in all_ops]

    # def is_date_entity(self):
    #     return 'date-entity' in str(self)

    def entity_name(self):
        entity_map = {'phone-number-entity': 'phone', 'url-entity': 'url', 'percentage-entity': 'percentage', 'email-address-entity': 'email'}
        curr_str = self.node_str()
        if curr_str in entity_map:
            return entity_map[curr_str]
        return None

    def __str__(self):
        result = self.node_label()
        if self.is_var_node():
            tmp_result = ('/' + self.graph.dict[result])
            while result[-1] in '0123456789':
                result = result[:-1]
            result += tmp_result

        if self.use_quote:
            result = '"%s"' % result
        return result

class AMRGraph(object):
    def initialize(self):
        self.dict = {}
        self.nodes = []   #used to record all the nodes
        self.node_dict = {}
        self.edges = []
        self.edge_dict = {}
        self.root = None

        self.ne_index = 0
        self.ent_index = 0
        self.verb_index = 0
        self.var_index = 0
        self.const_index = 0

    def __init__(self, line=None):
        if line is None:
            self.initialize()
            return

        vars, var_values, rel_links = from_AMR_line(line)

        self.dict = {}
        label_to_node = {}
        self.nodes = []   #used to record all the nodes
        self.node_dict = {}
        self.edges = []
        self.edge_dict = {}
        self.root = None

        self.ne_index = 0
        self.ent_index = 0
        self.verb_index = 0
        self.var_index = 0
        self.const_index = 0

        for i in range(len(vars)):
            curr_var = vars[i]
            self.dict[curr_var] = var_values[i] #maintain a dict for all the variable values

            #Setting a constant edge for a node: the variable name
            if curr_var not in label_to_node.keys(): #Haven't created a node for the current variable yet
                curr_node = AMRNode(self)
                self.node_dict[curr_node] = len(self.nodes)
                self.nodes.append(curr_node)
                curr_node_idx = self.node_dict[curr_node]

                if i == 0:
                    self.root = curr_node_idx
                const_edge = AMREdge(curr_var, self, curr_node_idx)
                self.edge_dict[const_edge] = len(self.edges)
                self.edges.append(const_edge)
                curr_edge_index = self.edge_dict[const_edge]

                curr_node.set_const_edge(curr_edge_index) #The const edge is set immediately after the initialization of a node
                label_to_node[curr_var] = curr_node_idx

        for i in range(len(vars)):
            curr_var = vars[i]

            curr_node_idx = label_to_node[curr_var]
            curr_node = self.nodes[curr_node_idx]
            if curr_var in rel_links:
                for rel, linked_val, is_var, is_coref in rel_links[curr_var]:
                    if is_var:
                        assert linked_val in label_to_node.keys(), 'Current coref variable %s is not a node yet' % linked_val
                        tail_node_idx = label_to_node[linked_val] #Find the existed linked node index
                        edge = AMREdge(rel, self, curr_node_idx, tail_node_idx)
                        if is_coref:  #The node for this variable has already been generated
                            edge.set_coref(True)

                        self.edge_dict[edge] = len(self.edges)
                        self.edges.append(edge)
                        curr_edge_index = self.edge_dict[edge]

                        curr_node.add_incoming(curr_edge_index)

                        tail_node = self.nodes[tail_node_idx]
                        tail_node.add_parent_edge(curr_edge_index)
                    else:
                        tail_node = AMRNode(self, True)  #Add a flag that it is a const node
                        if linked_val[0] == "\"" and linked_val[-1] == "\"":
                            linked_val = linked_val[1:-1]
                            tail_node.set_quote(True)

                        if '/' in linked_val:  #Dangerous here, pruned during the read amr procedure
                            try:
                                assert False, 'This should not happen again'
                            except:
                                print >> sys.stderr, linked_val
                                linked_val = linked_val.replace('/', '@@@@')
                                print >> sys.stderr, linked_val

                        self.node_dict[tail_node] = len(self.nodes)
                        self.nodes.append(tail_node)
                        tail_node_idx = self.node_dict[tail_node]

                        tail_const = AMREdge(linked_val, self, tail_node_idx)

                        self.edge_dict[tail_const] = len(self.edges)
                        self.edges.append(tail_const)
                        tail_edge_index = self.edge_dict[tail_const]

                        tail_node.set_const_edge(tail_edge_index)
                        edge = AMREdge(rel, self, curr_node_idx, tail_node_idx)

                        self.edge_dict[edge] = len(self.edges)
                        self.edges.append(edge)
                        curr_edge_index = self.edge_dict[edge]

                        curr_node.add_incoming(curr_edge_index)
                        tail_node.add_parent_edge(curr_edge_index)

    def setStats(self, stats):
        self.stats = stats

    def addConcept(self, is_var=False, var_id=None, var_name=None, use_quote=False):
        new_node = AMRNode(self)
        new_node_idx = self.node_dict.setdefault(new_node, len(self.nodes))
        self.nodes.append(new_node)

        if is_var:
            self.dict[var_id] = var_name
            edge_index = self.addEdge(new_node_idx, var_id)
        else:
            new_node.is_const = True
            edge_index = self.addEdge(new_node_idx, var_name)
            if use_quote:
                new_node.set_quote(True)
        new_node.set_const_edge(edge_index)
        return new_node_idx

    def addEdge(self, head_index, rel, tail_index=None, is_coref=False):
        edge = AMREdge(rel, self, head_index, tail_index)
        edge.set_coref(is_coref)
        edge_index = self.edge_dict.setdefault(edge, len(self.edges))
        self.edges.append(edge)
        head_node = self.nodes[head_index]
        if tail_index is None:
            head_node.set_const_edge(edge_index)
        else:
            head_node.add_incoming(edge_index)
            tail_node = self.nodes[tail_index]
            tail_node.add_parent_edge(edge_index)
        return edge_index

    def getFreq(self, node_idx):
        curr_node = self.nodes[node_idx]
        curr_var = curr_node.node_str()
        if self.is_predicate(curr_node):
            return self.stats.num_predicates[curr_var]

        elif curr_var in self.stats.num_nonpredicate_vals:
            return self.stats.num_nonpredicate_vals[curr_var]

        return 0

    def getFragment(self, node_idx, verb_map, pred_freq_thre, var_freq_thre, nodeid_to_frag):
        curr_node = self.nodes[node_idx]
        curr_var = curr_node.node_str()
        curr_frag = None
        if self.is_named_entity(curr_node):
            exclude_rels = ['wiki','name']
            assert node_idx in nodeid_to_frag
            entity_name = curr_node.node_str()
            curr_frag = nodeid_to_frag[node_idx]
            ret_var = 'NE_%s' % entity_name

            return curr_frag, exclude_rels, ret_var, True
        elif self.is_date_entity(curr_node):
            ret_var = 'DATE'
            assert node_idx in nodeid_to_frag
            curr_frag = nodeid_to_frag[node_idx]
            return curr_frag, date_relations, ret_var, True
        elif self.is_number(curr_node):
            ret_var = 'NUMBER'
            curr_frag = self.build_entity_fragment(curr_node)
            return curr_frag, [], ret_var, True

        else:
            curr_frag = self.build_entity_fragment(curr_node)
            if self.stats.num_nonpredicate_vals[curr_var] >= var_freq_thre:
                ret_var = curr_var
                categorized = False
            else:
                ret_var = SURF
                categorized = True
            return curr_frag, [], ret_var, categorized

        return None, [], curr_var, False

    def get_symbol(self, node_idx, verb_map, pred_freq_thre, var_freq_thre):
        curr_node = self.nodes[node_idx]
        curr_var = curr_node.node_str()
        if self.is_named_entity(curr_node):
            exclude_rels = ['wiki','name']
            entity_name = curr_node.node_str()
            ret_var = 'NE_%s' % entity_name

            return exclude_rels, ret_var, True
        elif self.is_date_entity(curr_node):
            ret_var = 'DATE'
            return date_relations, ret_var, True

        elif self.is_entity(curr_node) and (node_idx not in verb_map):
            entity_name = curr_node.node_str()
            ret_var = 'ENT_%s' % entity_name
            #ret_var = 'ENT_%s-%d' % (entity_name, self.ent_index)
            #self.ent_index += 1
            return [], ret_var, True
        elif self.is_predicate(curr_node):
            if self.stats.num_predicates[curr_var] >= pred_freq_thre:
                if node_idx in verb_map:
                    ex_rels = self.findMatch(curr_node, verb_map[node_idx])
                    ret_var = curr_var + '_VERBAL'
                    return ex_rels, ret_var, True
                ret_var = curr_var
                categorized = False
            else:
                if node_idx in verb_map:
                    ex_rels = self.findMatch(curr_node, verb_map[node_idx])
                    ret_var = 'VERBAL'
                    return ex_rels, ret_var, True
                ret_var = VERB
                #ret_var = VERB + ('%d' % self.verb_index)
                #self.verb_index += 1
                categorized = True
            return [], ret_var, categorized
        elif self.is_const(curr_node):
            if curr_var in ['interrogative', 'imperative', 'expressive', '-']:
                return [], curr_var, False
            else:
                ret_var = CONST
                #ret_var = CONST + ('%d' % self.const_index)
                #self.const_index += 1
                return [], ret_var, True

        else:
            if self.stats.num_nonpredicate_vals[curr_var] >= var_freq_thre:
                if node_idx in verb_map:
                    ex_rels = self.findMatch(curr_node, verb_map[node_idx])
                    ret_var = curr_var + '_VERBAL'
                    return ex_rels, ret_var, True
                ret_var = curr_var
                categorized = False
            else:
                if node_idx in verb_map:
                    ex_rels = self.findMatch(curr_node, verb_map[node_idx])
                    ret_var = 'SURF_VERBAL'
                    return ex_rels, ret_var, True
                ret_var = SURF
                #ret_var = SURF + ('%d' % self.var_index)
                #self.var_index += 1
                categorized = True
            return [], ret_var, categorized

        return [], curr_var, False

    def get_ancestors(self, n, stop_if_see = None):
        set_n = {n:('',-1)}
        queue = [n,]
        while len(queue) > 0:
            cur = queue.pop(0)
            for ei in self.nodes[cur].p_edges:
                e = self.edges[ei]
                prn = e.head
                if prn not in set_n:
                    set_n[prn] = (e.label, cur)
                    queue.append(prn)
                if stop_if_see != None and prn in stop_if_see:
                    return (set_n, prn)
        return (set_n, None)

    def get_path_v2(self, n1, n2):
        if n1 == n2:
            return [], []

        prn_n1, root = self.get_ancestors(n1, {n2:None,})
        if root != None:
            result = []
            while n2 != n1:
                result.insert(0,prn_n1[n2][0])
                n2 = prn_n1[n2][1]
            return (result, [])

        prn_n2, root = self.get_ancestors(n2, prn_n1)
        assert root != None
        if n1 == root:
            result = []
            while n1 != n2:
                result.append(prn_n2[n1][0])
                n1 = prn_n2[n1][1]
            return ([], result)
        else:
            up = []
            tmp = root
            while tmp != n1:
                up.insert(0,prn_n1[tmp][0])
                tmp = prn_n1[tmp][1]
            down = []
            tmp = root
            while tmp != n2:
                down.append(prn_n2[tmp][0])
                tmp = prn_n2[tmp][1]
            return (up, down)


    def get_path(self, n1, n2):
        assert False, 'not fully tested, don\'t use'
        assert n1 < len(self.nodes) and n2 < len(self.nodes)

        print('---------------', n1, n2)
        set_n1 = set([n1,])
        labels_n1 = []
        while len(self.nodes[n1].p_edges) > 0:
            edge = self.edges[self.nodes[n1].p_edges[0]]
            n1_prime = edge.head
            if n1_prime in set_n1:
                break
            print('n1 part', (n1, n1_prime, edge.label))
            labels_n1.append((n1_prime, edge.label))
            n1 = n1_prime

        set_n2 = set([n2,])
        labels_n2 = []
        while n2 not in set_n1 and len(self.nodes[n2].p_edges) > 0:
            edge = self.edges[self.nodes[n2].p_edges[0]]
            n2_prime = edge.head
            if n2_prime in set_n2:
                break
            print('n2 part', (n2, n2_prime, edge.label))
            labels_n2.append((n2_prime, edge.label))
            n2 = n2_prime

        result = []
        for (n1, label) in labels_n1:
            if n1 != n2:
                result.append(label)
        result = result + [x[1] for x in labels_n2]
        print(result)
        return result


    def get_relation_edges(self):
        relation_edges = {}
        for edge in self.edges:
            if edge.head != None and edge.tail != None: # get rid of const edges
                relation_edges[(edge.head, edge.tail)] = edge.label
        return relation_edges

    def get_from_path(self, path):
        if path == '1':
            return ('n', self.root)
        path = path.split('.')
        cur_node = self.nodes[self.root]
        for k in range(1,len(path)):
            offset = int(path[k])-1
            if k == len(path)-2 and path[-1] == 'r':
                return ('e', cur_node.v_edges[offset])
            elif k == len(path)-1:
                return ('n', cur_node.get_child(offset)[0])
            else:
                cur_node = cur_node.get_child(offset)[1]
        assert False, 'Invalid Path:%s' % path

    #Check if a node is the root of an entity
    def is_named_entity(self, node):
        edge_label_set = set([self.edges[x].label for x in node.v_edges])
        if 'name' in edge_label_set:
            try:
                assert 'wiki' in edge_label_set
            except:
                print('ill-formed entity found')
                print(str(self))
                return False
            return True
        return False

    def is_date_entity(self, node):
        if node.is_var_node():
            var_concept = node.node_str()
            return var_concept.strip() == 'date-entity'
        return False

    def is_number(self, node):
        if node.is_var_node():
            return False
        return isNumber(node.node_str())

    def is_entity(self, node):
        if node.is_var_node():
            var_concept = node.node_str()
            return var_concept.endswith('-entity') or var_concept.endswith('-quantity') or var_concept.endswith('-organization') or var_concept == 'amr-unknown'
        return False

    def is_predicate(self, node):
        if node.is_var_node():
            var_concept = node.node_str()
            return re.match('.*-[0-9]+', var_concept) is not None
        return False

    def is_const(self, node):
        return node.is_const

    def set_sentence(self, s):
        self.sent = s

    def set_lemmas(self, s):
        self.lems = s

    def set_poss(self, s):
        self.poss = s

    def set_ners(self, s):
        self.ners = s

    def print_info(self):
        #print all nodes info
        print('Nodes information:')
        print('Number of nodes: %s' % len(self.nodes))
        for node in self.node_dict.keys():
            print(str(node), ',', self.node_dict[node])

        #print all edges info
        print('Edges information')
        print('Number of edges: %s' % len(self.edges))
        for edge in self.edge_dict.keys():
            print(edge.label, ',', self.edge_dict[edge])

    def check_self_cycle(self):
        visited_nodes = set()
        sequence = []
        root_node = self.nodes[self.root]
        if len(root_node.p_edges) > 0:
            return True

        stack = [(self.root, 0)]
        while stack:
            curr_node_idx, depth = stack.pop()
            if curr_node_idx in visited_nodes:
                continue
            if depth >= len(sequence):
                sequence.append(curr_node_idx)
            else:
                sequence[depth] = curr_node_idx

            visited_nodes.add(curr_node_idx)
            curr_node = self.nodes[curr_node_idx]
            if len(curr_node.v_edges) > 0:
                for edge_index in reversed(curr_node.v_edges):  #depth first search
                    curr_edge = self.edges[edge_index]
                    child_index = curr_edge.tail
                    if child_index in sequence[:depth+1]:
                        return True
                    stack.append((child_index, depth+1))
        return False

    #Return the order each node was visited
    def dfs(self):
        visited_nodes = set()
        sequence = []
        stack = [(self.root, None, 0, False)]
        while stack:
            curr_node_idx, par_rel, depth, is_coref = stack.pop()
            #print curr_node_idx, par_rel, depth, is_coref
            sequence.append((curr_node_idx, is_coref, par_rel, depth)) #push a tuple recording the depth search info
            if is_coref:
                continue

            visited_nodes.add(curr_node_idx)
            curr_node = self.nodes[curr_node_idx]

            #print curr_node_idx, curr_node.p_edges, curr_node.v_edges
            if len(curr_node.v_edges) > 0:
                for edge_index in reversed(curr_node.v_edges):  #depth first search
                    curr_edge = self.edges[edge_index]
                    curr_rel = curr_edge.label
                    child_index = curr_edge.tail
                    if not curr_edge.is_coref:
                        stack.append((child_index, curr_rel, depth+1, False))
                    else:
                        stack.append((child_index, curr_rel, depth+1, True))
        return sequence

    def update_stats(self, conceptToOutGo, conceptToIncome, conceptOutgoFreq,
                     conceptIncomeFreq, concept_counts, relcounts, frequent_set, category=True):

        visited_nodes = set()
        sequence = []
        stack = [(self.root, None, 0, False)]

        while stack:
            curr_node_idx, par_rel, depth, is_coref = stack.pop()
            if is_coref:
                continue

            visited_nodes.add(curr_node_idx)
            curr_node = self.nodes[curr_node_idx]
            curr_label = curr_node.node_str()
            if category:
                curr_label = utils.getCategories(curr_label, frequent_set, True)

            concept_counts[curr_label] += 1

            if len(curr_node.v_edges) > 0:
                for edge_index in reversed(curr_node.v_edges):  #depth first search
                    curr_edge = self.edges[edge_index]
                    curr_rel = curr_edge.label

                    relcounts[curr_rel] += 1
                    child_index = curr_edge.tail

                    child_node = self.nodes[child_index]
                    child_label = child_node.node_str()
                    if category:
                        child_label = utils.getCategories(child_label, frequent_set, True)

                    if curr_label not in conceptToOutGo:
                        conceptToOutGo[curr_label] = defaultdict(int)
                    conceptToOutGo[curr_label][curr_rel] += 1
                    conceptOutgoFreq[curr_label] += 1

                    if child_label not in conceptToIncome:
                        conceptToIncome[child_label] = defaultdict(int)
                    conceptToIncome[child_label][curr_rel] += 1
                    conceptIncomeFreq[child_label] += 1

                    if not curr_edge.is_coref:
                        stack.append((child_index, curr_rel, depth+1, False))
                    else:
                        stack.append((child_index, curr_rel, depth+1, True))
        return sequence

    @classmethod
    def collapsedAMR(cls, amr, all_alignments):
        """
        Categories---
        NER: named entity
        DATE: date entity
        VERBAL: verbalization list
        NUMBER: number
        NEGATION: polarity
        PHRASE: ---

        """

        def category_info(node_idx, category, subgraph=None):
            curr_node = amr.nodes[node_idx]
            root_repr = curr_node.node_str()
            exclude_rels = []
            if category == "NER":
                assert amr.is_named_entity(curr_node)
                exclude_rels = ["wiki", "name"]
                category_name = "NE_%s" % root_repr
            elif category == "DATE":
                assert amr.is_date_entity(curr_node)
                exclude_rels = date_relations
                category_name = "DATE"
            elif category == "NUMBER":
                if not amr.is_number(curr_node):
                    category_name = root_repr
                else:
                    category_name = "NUMBER"
            elif category == "MULT":
                assert subgraph is not None
                all_rels = subgraph.split()[1:]
                exclude_rels = [tuple(s.split(":")) for s in all_rels]
                category_name = "MULT_%s" % root_repr
            elif category == "NEG": # Negation
                exclude_rels = ["polarity"]
                category_name = "NEG_%s" % root_repr
            else:
                category_name = root_repr
            return category_name, exclude_rels

        new_amr = AMRGraph()
        new_alignments = defaultdict(list)

        span_to_type = {}
        stack = [(amr.root, TOP, None)] #Start from the root of the AMR

        visited = set()
        collapsed_map = {}

        node_oldtonew = {}
        # nodeid_to_type = {}

        while stack:
            curr_node_idx, rel, parent = stack.pop()
            if curr_node_idx in collapsed_map:
                curr_node_idx = collapsed_map[curr_node_idx]
            curr_node = amr.nodes[curr_node_idx]
            curr_var = curr_node.node_str()

            if curr_node_idx in visited: #A reentrancy found
                new_node_idx = node_oldtonew[curr_node_idx]
                new_amr.addEdge(parent, rel, new_node_idx, True)

                continue

            visited.add(curr_node_idx)

            if curr_node_idx in all_alignments:
                _, _, graph_repr, category = all_alignments[curr_node_idx][0]
                curr_sym, exclude_rels = category_info(curr_node_idx, category, graph_repr)

            else:
                exclude_rels, curr_sym, categorized = [], curr_var, False

            if curr_node.is_var_node(): #Register the node, and its constant edge
                var_name = curr_node.node_label()
                new_node_idx = new_amr.addConcept(True, var_name, curr_sym)
            else:
                new_node_idx = new_amr.addConcept(False, None, curr_sym, curr_node.use_quote)

            if curr_node_idx == amr.root:
                # assert parent is None
                new_amr.root = new_node_idx

            if curr_node_idx in all_alignments:
                new_alignments[new_node_idx] = all_alignments[curr_node_idx]
                for (start, end, graph_repr, _) in all_alignments[curr_node_idx]:
                    span_to_type[(start, end)] = (new_node_idx, graph_repr, curr_sym)
                    # nodeid_to_type[new_node_idx] = (graph_repr, curr_sym)

            node_oldtonew[curr_node_idx] = new_node_idx

            if parent is not None:
                new_amr.addEdge(parent, rel, new_node_idx)

            for edge_index in reversed(curr_node.v_edges):
                curr_edge = amr.edges[edge_index]
                child_index = curr_edge.tail
                child_repr = amr.nodes[child_index].node_str()
                if ("MULT" not in curr_sym and curr_edge.label in exclude_rels) or (
                                "MULT" in curr_sym and (curr_edge.label, child_repr) in exclude_rels):
                    # visited.add(child_index)
                    collapsed_map[child_index] = curr_node_idx
                    if curr_sym[:2] != 'NE': #Might have other relations
                        tail_node = amr.nodes[child_index]

                        for next_edge_index in reversed(tail_node.v_edges):
                            next_edge = amr.edges[next_edge_index]
                            next_child_index = next_edge.tail
                            stack.append((next_child_index, next_edge.label, new_node_idx))
                    continue
                stack.append((child_index, curr_edge.label, new_node_idx))

        return new_amr, new_alignments, span_to_type

    def dfsCollapsed(self):
        visited_nodes = set()
        sequence = []
        stack = [(self.root, None, 0, False)]
        while stack:
            curr_node_idx, par_rel, depth, is_coref = stack.pop()
            sequence.append((curr_node_idx, is_coref, par_rel, depth)) #push a tuple recording the depth search info
            if is_coref:
                continue

            visited_nodes.add(curr_node_idx)
            curr_node = self.nodes[curr_node_idx]
            if self.is_named_entity(curr_node):
                exclude_rels = ['wiki', 'name']
            elif self.is_date_entity(curr_node):
                exclude_rels = date_relations
            else:
                exclude_rels = []

            if len(curr_node.v_edges) > 0:
                for edge_index in reversed(curr_node.v_edges):  #depth first search
                    curr_edge = self.edges[edge_index]
                    curr_rel = curr_edge.label
                    child_index = curr_edge.tail
                    if not curr_edge.is_coref:
                        stack.append((child_index, curr_rel, depth+1, False))
                    else:
                        stack.append((child_index, curr_rel, depth+1, True))
        return sequence

    def collapsed_dfs(self, root2fragment):
        visited_nodes = set()
        sequence = []

        collapsed_edges = bitarray(len(self.edges))
        if collapsed_edges.count() != 0:
            collapsed_edges ^= collapsed_edges
        assert collapsed_edges.count() == 0

        collapsed_nodes = bitarray(len(self.nodes))
        if collapsed_nodes.count() != 0:
            collapsed_nodes ^= collapsed_nodes
        assert collapsed_nodes.count() == 0

        stack = [(self.root, None, 0, False)]
        while stack:
            curr_node_idx, par_rel, depth, is_coref = stack.pop()
            if collapsed_nodes[curr_node_idx] == 1: #Disallow sharing of inside of named entity, delete for approximation
                continue

            sequence.append((curr_node_idx, is_coref, par_rel, depth)) #push a tuple recording the depth search info
            if is_coref:
                continue

            visited_nodes.add(curr_node_idx)
            curr_node = self.nodes[curr_node_idx]

            if curr_node_idx in root2fragment:  #Need to be collapsed
                collapsed_edges |= root2fragment[curr_node_idx].edges #Collapse all the entity edges
                collapsed_nodes |= root2fragment[curr_node_idx].nodes #These nodes should never be revisited
                collapsed_nodes[curr_node_idx] = 0

            if len(curr_node.v_edges) > 0:
                for edge_index in reversed(curr_node.v_edges):  #depth first search
                    if collapsed_edges[edge_index] == 1: #Been collapsed
                        continue

                    curr_edge = self.edges[edge_index]
                    curr_rel = curr_edge.label
                    child_index = curr_edge.tail
                    if not curr_edge.is_coref:
                        stack.append((child_index, curr_rel, depth+1, False))
                    else:
                        stack.append((child_index, curr_rel, depth+1, True))
        return sequence

    def extract_entities(self):
        entity_frags = []
        visited_nodes = set()
        stack = [(self.root, None, 0, False)]

        while stack:
            curr_node_idx, par_rel, depth, is_coref = stack.pop()

            if is_coref:
                continue

            visited_nodes.add(curr_node_idx)
            curr_node = self.nodes[curr_node_idx]
            if curr_node.is_named_entity():
                frag = self.build_entity_fragment(curr_node)
                wiki_label = curr_node.getWiki()
                entity_frags.append((frag, wiki_label, "NER"))
            elif curr_node.is_date_entity():
                frag = self.build_entity_fragment(curr_node, "DATE")
                # assert frag is not None, str(self)
                entity_frags.append((frag, None, "DATE"))

            if len(curr_node.v_edges) > 0:
                for edge_index in reversed(curr_node.v_edges):  #depth first search
                    curr_edge = self.edges[edge_index]
                    curr_rel = curr_edge.label
                    child_index = curr_edge.tail
                    if not curr_edge.is_coref:
                        stack.append((child_index, curr_rel, depth+1, False))
                    else:
                        stack.append((child_index, curr_rel, depth+1, True))
        return entity_frags

    def matchedTuples(self, node_idx, subgraph):
        node = self.nodes[node_idx]

        root_label = node.node_str()
        assert root_label in subgraph
        tuple_map = {}
        for edge_index in node.v_edges:
            curr_edge = self.edges[edge_index]
            rel = curr_edge.label
            tail_index = curr_edge.tail
            tail_concept = self.nodes[tail_index].node_str()
            tuple_map[(rel, tail_concept)] = (node_idx, edge_index, tail_index)

        matched_tuples = []

        for (rel, tail_concept) in subgraph[root_label].items():
            if (rel, tail_concept) in tuple_map:
                matched_tuples.append(tuple_map[(rel, tail_concept)])
            else:
                return []

        return matched_tuples

    def findMatch(self, node, subgraph):
        root_label = node.node_str()
        assert root_label in subgraph
        tuple_set = set([(self.edges[edge_index].label, self.nodes[self.edges[edge_index].tail].node_str()) for edge_index in node.v_edges])
        ex_rels = []

        for (rel, tail_concept) in subgraph[root_label].items():
            if (rel, tail_concept) in tuple_set:
                ex_rels.append(rel)
            else:
                return []

        return sorted(ex_rels)

    def matchSubgraph(self, subgraph):
        visited_nodes = set()
        stack = [(self.root, None, False)]

        matched_frags = []

        while stack:
            curr_node_idx, par_rel, is_coref = stack.pop()

            if is_coref:
                continue

            visited_nodes.add(curr_node_idx)
            curr_node = self.nodes[curr_node_idx]

            if curr_node.node_str() in subgraph:
                matched_tuples = self.matchedTuples(curr_node_idx, subgraph)

                if matched_tuples:
                    matched_frags.append(matched_tuples)

            if len(curr_node.v_edges) > 0:
                for edge_index in reversed(curr_node.v_edges):  #depth first search
                    curr_edge = self.edges[edge_index]
                    curr_rel = curr_edge.label
                    child_index = curr_edge.tail
                    if not curr_edge.is_coref:
                        stack.append((child_index, curr_rel, False))
                    else:
                        stack.append((child_index, curr_rel, True))
        return matched_frags

    #Try breadth-first search to find unaligned edges
    #For all edges coming out of the same node, add additional processing for args
    #When storing the fragments, be careful with the external nodes
    def unaligned_nodes(self, aligned_nodes):

        visited_nodes = set()
        unaligned_nodes = []
        stack = deque([self.root])
        while stack:
            curr_node_idx = stack.popleft()

            if curr_node_idx in visited_nodes:
                continue

            visited_nodes.add(curr_node_idx)
            curr_node = self.nodes[curr_node_idx]

            if curr_node_idx not in aligned_nodes:
                unaligned_nodes.append(curr_node)

            for curr_edge_index in curr_node.v_edges:
                curr_edge = self.edges[curr_edge_index]
                tail_node_idx = curr_edge.tail
                stack.append(tail_node_idx)

        return unaligned_nodes

    def build_entity_fragment(self, node, category="NER"):
        n_nodes = len(self.nodes)
        n_edges = len(self.edges)

        frag = AMRFragment(n_edges, n_nodes, self)

        curr_node = node
        curr_node_idx = self.node_dict[curr_node]

        frag.set_root(curr_node_idx)

        c_edge_index = curr_node.c_edge
        frag.set_edge(c_edge_index)

        for curr_edge_index in curr_node.v_edges:
            curr_edge = self.edges[curr_edge_index]
            if category == "NER":
                if curr_edge.label == 'wiki' or curr_edge.label == 'name':
                    frag.set_edge(curr_edge_index)
                    tail_node_idx = curr_edge.tail
                    frag.set_node(tail_node_idx)
                    tail_node = self.nodes[tail_node_idx]
                    frag.set_edge(tail_node.c_edge)
                    # assert len(tail_node.v_edges) == 0 and len(tail_node.p_edges) == 1

                    if curr_edge.label == 'name':
                        for grand_edge_index in tail_node.v_edges:
                            curr_edge = self.edges[grand_edge_index]
                            if 'op' not in curr_edge.label:
                                continue

                            frag.set_edge(grand_edge_index)
                            tail_node_idx = curr_edge.tail
                            frag.set_node(tail_node_idx)
                            curr_tail_node = self.nodes[tail_node_idx]
                            frag.set_edge(curr_tail_node.c_edge)
                            assert len(curr_tail_node.v_edges) == 0 and len(curr_tail_node.p_edges) == 1
            elif category == "DATE":
                if curr_edge.label in date_relations:
                    frag.set_edge(curr_edge_index)
                    tail_node_idx = curr_edge.tail
                    frag.set_node(tail_node_idx)
                    tail_node = self.nodes[tail_node_idx]
                    frag.set_edge(tail_node.c_edge)
                    # Outlier: "early morning".
                    # if len(tail_node.v_edges) != 0:
                    #     return None

        frag.build_ext_list()
        frag.build_ext_set()
        return frag

    #The concepts are all unary or split at the last stage
    def retrieve_fragment(self, integer_reps):
        n_nodes = len(self.nodes)
        n_edges = len(self.edges)

        frag = AMRFragment(n_edges, n_nodes, self)

        integer_concepts = sorted(integer_reps.split('+')) #newly added by xpeng
        integer_concepts = [x.split('.') for x in integer_concepts] #Each concept is a list of integers memorizing a path from the root to the concept
        integer_concepts = [[(int)(x) for x in y] for y in integer_concepts]

        n_concepts = len(integer_concepts)
        n_identified_concepts = 0

        #curr_node = self.root
        curr_num = 0
        lengths = [len(x) for x in integer_concepts]
        max_len = max(lengths)

        n_nodes = len(lengths)
        ext_set = set()
        curr_node_idx = self.retrieve_first_concept(integer_concepts[0])
        if curr_node_idx != self.root:
            ext_set.add(curr_node_idx)

        #index = self.node_dict[curr_node]
        frag.set_root(curr_node_idx) #The structure of the smallest grained fragment should be rooted structure
        #frag.set_node(index)
        curr_node = self.nodes[curr_node_idx]

        c_edge_index = curr_node.c_edge
        #edge_index = self.edge_dict[edge]
        frag.set_edge(c_edge_index)

        if curr_node_idx == self.root:
            #try:
            #    assert len(curr_node.p_edges) == 0, 'The root node has some parent nodes'
            #except AssertionError:
            #    #print str(self)
            #    #logger.writeln(str(self))
            #    #sys.exit(-1)
            #    return None

            length_two = len([None for x in lengths if (x == 2)])
            if len(curr_node.v_edges) > length_two:
                ext_set.add(curr_node_idx)

        if n_nodes == 1:
            frag.set_ext_set(ext_set)
            return frag

        par_node_idx = curr_node_idx
        curr_depth = len(integer_concepts[0])

        #Starting from the node identified in the first step, retrieve the rest of the nodes and the relations that connect them
        for i in range(1, n_nodes):
            curr_len = lengths[i]
            try:
                assert len(integer_concepts[i]) == curr_depth+1, 'A relation has across more than 1 layer in just one step'
            except:
                print(integer_concepts)
                print(integer_concepts[i])
                print(curr_depth)
                #sys.exit(-1)
                #logger.writeln(integer_concepts)
                return None
            (curr_edge_index, curr_node_idx) = self.retrieve_one_concept(integer_concepts[i][curr_depth], par_node_idx)
            if curr_edge_index == -1:
                return None
            frag.set_edge(curr_edge_index)
            curr_node = self.nodes[curr_node_idx]
            if len(curr_node.p_edges) > 1:
                ext_set.add(curr_node_idx)
            const_edge_index = curr_node.c_edge
            frag.set_node(curr_node_idx)
            frag.set_edge(const_edge_index)

            if curr_len < max_len: #Not the leaf of the tree fragment
                if i < n_nodes-1 and curr_len == lengths[i+1]:
                    if len(curr_node.v_edges) > 0:
                        ext_set.add(curr_node_idx)
                else:
                    n_next_len = len([None for x in lengths if (x == curr_len+1)])
                    if len(curr_node.v_edges) > n_next_len:
                        ext_set.add(curr_node_idx)
                    par_node_idx = curr_node_idx
                    curr_depth = len(integer_concepts[i])
            else:
                if len(curr_node.v_edges) > 0:
                    ext_set.add(curr_node_idx)

        frag.set_ext_set(ext_set)
        return frag

    #Newly added, to retrieve the smallest alignment
    def get_concept_relation(self, s_rep):
        #print "s_rep: {}".format(s_rep)
        #print "self.nodes: {}".format([(n.node_label(), n.node_str()) for n in self.nodes])
        #print "e.head {}".format(self.edges[0].head)
        #print "e.tail {}".format(self.edges[0].tail)
        #print "self.edges: {}".format([(self.nodes[e.head].node_label(), self.nodes[e.tail].node_label() if e.tail else "NONE") for e in self.edges])
        path = s_rep.split('.')
        for (i, curr_index) in enumerate(path):
            curr_index = int(curr_index)
            if i == 0:
                assert curr_index == 0 #The root
                curr_node_idx = self.root
                curr_node = self.nodes[curr_node_idx]
            else:
                try:
                    #print curr_index
                    #if curr_index >= len(curr_node.v_edges):
                    #  print "This index is too big!!!! WHY!!!!!"
                    #  print "curr_index {}".format(curr_index)
                    #  print "curr_node.v_edges {}".format(curr_node.v_edges)
                    #  curr_index = len(curr_node.v_edges) - 1
                    #print curr_node
                    #print curr_node.v_edges
                    curr_edge_index = curr_node.v_edges[curr_index]
                except:
                    print("Exception in amr_graph.py get_concept_relation()")
                    print(s_rep)
                    sys.exit(1)
                if i + 1 != len(path) and path[i+1] == 'r': #It's a relation
                    return ('r', curr_edge_index)
                curr_edge = self.edges[curr_edge_index]
                curr_node_idx = curr_edge.tail
                if i + 1 != len(path):
                    curr_node = self.nodes[curr_node_idx]
                    #print("curr_index {}".format(curr_index))
                    #print("curr_node: {}".format(curr_node))


        return ('c', curr_node_idx)

    def retrieve_first_concept(self, i_path):
        if len(i_path) == 1:
            assert i_path[0] == 0
            return self.root

        curr_node_idx = self.root
        curr_node = self.nodes[curr_node_idx]
        for curr_depth in range(1, len(i_path)):
            v_edges = curr_node.v_edges
            num = 0
            curr_index = i_path[curr_depth]
            for i in range(len(v_edges)): #search for non-coref nodes
                curr_edge_index = v_edges[i]
                curr_edge = self.edges[curr_edge_index]
                if not curr_edge.is_coref:
                    num += 1
                if num == curr_index+1:
                    curr_node_idx = curr_edge.tail
                    curr_node = self.nodes[curr_node_idx]
                    break
        return curr_node_idx

    def retrieve_one_concept(self, child_num, par_node_idx):
        par_node = self.nodes[par_node_idx]
        v_edges = par_node.v_edges
        curr_index = 0
        curr_edge_index = -1
        curr_node_idx = -1
        for i in range(len(v_edges)): #search for non-coref nodes
            curr_edge_index = v_edges[i]
            curr_edge = self.edges[curr_edge_index]
            if not curr_edge.is_coref:
                curr_index += 1
            if child_num+1 == curr_index:
                curr_node_idx = curr_edge.tail
                break
        return (curr_edge_index, curr_node_idx)

    #Do a depth-first traversal of the graph, print the amr format
    #Especially careful with the re-entrance structure
    def __str__(self):
        s = ""
        node_sequence = self.dfs()
        #print node_sequence
        assert node_sequence[0][2] == None, 'The parent relation of the root should be none, %s' % node_sequence[0][2]

        dep_rec = 0 #record the current depth
        for curr_node_idx, is_coref, par_rel, depth in node_sequence:
            curr_node = self.nodes[curr_node_idx]
            #print str(curr_node)

            curr_c_edge = self.edges[curr_node.c_edge]
            curr_node_label = str(curr_c_edge)

            if par_rel == None:   #There is no relation going into this node
                if not is_coref and curr_node_label in self.dict.keys():
                    s += "(%s / %s" % (curr_node_label, self.dict[curr_node_label])
                else:
                    s += "(%s" % curr_node_label
            else:
                if depth < dep_rec:  #If the current layer is smaller than the current depth, then the previous few variables have finished traverse, print out the corresponding ) as finish
                    s += "%s" % ((dep_rec- depth) * ')')
                dep_rec = depth

                if curr_node.is_leaf():
                    if not is_coref and curr_node_label in self.dict.keys(): #In this case, current node is variable and is visited for the first time. Leaf variable
                        s += "\n%s:%s (%s / %s)"  % (depth*"\t", par_rel, curr_node_label, self.dict[curr_node_label])
                    else:
                        if curr_node_label not in self.dict.keys() and curr_node.use_quote:
                            s += '\n%s:%s "%s"' % (depth*"\t", par_rel, curr_node_label)
                        else:
                            s += "\n%s:%s %s" % (depth*"\t", par_rel, curr_node_label)
                else:
                    if not is_coref and curr_node_label in self.dict.keys(): #In this case, current node is variable and is visited for the first time. Not leaf variable
                        s += "\n%s:%s (%s / %s"  % (depth*"\t", par_rel, curr_node_label, self.dict[curr_node_label])
                    else:
                        s += "\n%s:%s %s" % (depth*"\t", par_rel, curr_node_label)
        if dep_rec != 0:
            s += "%s" % (dep_rec * ')')
        return s

    def statistics(self):
        named_entity_nums = defaultdict(int)
        entity_nums = defaultdict(int)
        predicate_nums = defaultdict(int)
        variable_nums = defaultdict(int)
        const_nums = defaultdict(int)
        reentrancy_nums = 0

        stack = [(self.root, None)]

        while stack:
            curr_node_idx, par_edge = stack.pop()
            curr_node = self.nodes[curr_node_idx]

            exclude_rels = []

            if par_edge and par_edge.is_coref:
                reentrancy_nums += 1
                continue

            if self.is_named_entity(curr_node):
                entity_name = curr_node.node_str()
                named_entity_nums[entity_name] += 1
                exclude_rels = ['name', 'wiki']
            elif self.is_entity(curr_node):
                entity_name = curr_node.node_str()
                entity_nums[entity_name] += 1
            elif self.is_predicate(curr_node):
                pred_name = curr_node.node_str()
                predicate_nums[pred_name] += 1
            elif self.is_const(curr_node):
                const_name = curr_node.node_str()
                const_nums[const_name] += 1
            else:
                variable_name = curr_node.node_str()
                variable_nums[variable_name] += 1

            for edge_index in curr_node.v_edges:
                curr_edge = self.edges[edge_index]
                edge_label = curr_edge.label
                if edge_label not in exclude_rels:
                    stack.append((curr_edge.tail, curr_edge))

        return (named_entity_nums, entity_nums, predicate_nums, variable_nums, const_nums, reentrancy_nums)

    #Do a depth-first traversal of the graph, print the amr format
    #Collapsing some fragments into single node repr
    def collapsed_form(self, root2fragment, root2entitynames):

        s = ""
        node_sequence = self.collapsed_dfs(root2fragment)
        assert node_sequence[0][2] == None, 'The parent relation of the root should be none, %s' % node_sequence[0][2]

        dep_rec = 0 #record the current depth
        for curr_node_idx, is_coref, par_rel, depth in node_sequence:
            curr_node = self.nodes[curr_node_idx]

            curr_c_edge = self.edges[curr_node.c_edge]
            curr_node_label = str(curr_c_edge)

            if curr_node_idx in root2fragment: #Collapse the fragment at current node

                entity_tag = root2entitynames[curr_node_idx]
                assert curr_node_label in self.dict
                #assert not curr_node.is_leaf()  #Can be unknow date-entity

                entity_frag = root2fragment[curr_node_idx]
                is_leaf_entity = True

                for edge_index in curr_node.v_edges:
                    if entity_frag.edges[edge_index] != 1:
                        is_leaf_entity = False

                if not par_rel:
                    if not is_coref: #First visit of this node, use the fragment name repr
                        if is_leaf_entity:
                            s += "(%s)" % entity_tag
                        else:
                            s += "(%s" % entity_tag #Single entity
                    else:
                        s += "%s_copy" % entity_tag
                        print("impossible here!")
                else:
                    if depth < dep_rec:  #If the current layer is smaller than the current depth, then the previous few variables have finished traverse, print out the corresponding ) as finish
                        s += "%s" % ((dep_rec- depth) * ')')
                    dep_rec = depth

                    if not is_coref:
                        if is_leaf_entity:
                            s += " :%s (%s)"  % (par_rel, entity_tag)
                        else:
                            s += " :%s (%s"  % (par_rel, entity_tag)
                    else:
                        s += " :%s %s_copy" % (par_rel, entity_tag)

            else:
                if not par_rel:   #There is no relation going into this node
                    if not is_coref and curr_node_label in self.dict.keys():
                        s += "(%s / %s" % (curr_node_label, self.dict[curr_node_label])
                    else:
                        s += "(%s" % curr_node_label
                else:
                    if depth < dep_rec:  #If the current layer is smaller than the current depth, then the previous few variables have finished traverse, print out the corresponding ) as finish
                        s += "%s" % ((dep_rec- depth) * ')')
                    dep_rec = depth

                    if curr_node.is_leaf():
                        if not is_coref and curr_node_label in self.dict.keys(): #In this case, current node is variable and is visited for the first time. Leaf variable
                            s += "\n%s:%s (%s / %s)"  % (depth*"\t", par_rel, curr_node_label, self.dict[curr_node_label])
                        else:
                            if curr_node_label not in self.dict.keys() and curr_node.use_quote:
                                s += '\n%s:%s "%s"' % (depth*"\t", par_rel, curr_node_label)
                            else:
                                s += "\n%s:%s %s" % (depth*"\t", par_rel, curr_node_label)
                    else:
                        if not is_coref and curr_node_label in self.dict.keys(): #In this case, current node is variable and is visited for the first time. Not leaf variable
                            s += "\n%s:%s (%s / %s"  % (depth*"\t", par_rel, curr_node_label, self.dict[curr_node_label])
                        else:
                            s += "\n%s:%s %s" % (depth*"\t", par_rel, curr_node_label)
        if dep_rec != 0:
            s += "%s" % (dep_rec * ')')
        return s

    def print_variables(self):
        s = ''
        for node in self.nodes:
            var = str(node)
            s += str(node)
            s += '/'
            s += self.dict[var]
            s += ' '
        print(s)

#unaligned_words are formatted as tuples of (position, word)
def match_word(label, unaligned_words, lemma_map, stop_words):
    for (pos, word) in unaligned_words:
        if word.lower() in stop_words:
            continue
        lem_w = None
        if word in lemma_map:
            lem_w = list(lemma_map[word])[0]

        if len(label) > 4:
            if word[:3] == label[:3] or (lem_w and lem_w[:3] == label[:3]):
                return (pos, word)
        else:
            if word == label or (lem_w and lem_w == label):
                return (pos, word)
    return (None, None)

def concept_label(label):
    concept_label = label
    if '/' in concept_label:
        concept_label = concept_label.split('/')[1].strip()
    elif concept_label[0] == '"':
        concept_label = delete_pattern(concept_label, '~e\.[0-9]+(,[0-9]+)*')
        assert concept_label[-1] == '"', 'weird constant %s' % concept_label
        concept_label = concept_label[1:-1]
    return concept_label

def is_root_arc(edge_label):
    return (edge_label[:3] == 'ARG' and 'of' not in edge_label) or edge_label[:2] == 'op' or edge_label[:3] == 'snt'

if __name__ == '__main__':
    amr_line = '(f / foolish :condition (d / do-02  :ARG0 i) :domain (i / i))'
    #amr_line = '(e / establish-01:ARG1 (m / model:mod (i / innovate-01:ARG1 (i2 / industry))))'
    g = AMRGraph(amr_line)
    print(str(g))

