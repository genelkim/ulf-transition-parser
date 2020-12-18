#!/usr/bin/python
# -*- coding:utf-8 -*-


#A hypergraph representation for amr.

#author: Chuan Wang
#since: 2013-11-20

from collections import defaultdict
from .util import *
import re,sys
from optparse import OptionParser
#from DependencyGraph import *

# Error definitions
class LexerError(Exception):
    pass
class ParserError(Exception):
    pass

class Node():

    #node_id = 0     #static counter, unique for each node
    #mapping_table = {}  # old new index mapping table

    def __init__(self, parent, trace, node_label, firsthit, leaf, depth, seqID):
        """
        initialize a node in the graph
        here a node keeps record of trace i.e. from where the node is reached (the edge label)
        so nodes with same other attributes may have different trace
        """
        self.parent = parent
        self.trace = trace
        self.node_label = node_label
        self.firsthit = firsthit
        self.leaf = leaf
        self.depth = depth
        self.children = []
        self.seqID = seqID
        #Node.node_id += 1
        #self.node_id = node_id

    def __str__(self):
        return str((self.trace, self.node_label, self.depth, self.seqID))

    def __repr__(self):
        return str((self.trace, self.node_label, self.depth, self.seqID))

class AMR(defaultdict):
    """
    An abstract meaning representation.
    Basic idea is based on bolinas' hypergraph for amr.

    Here one AMR is a rooted, directed, acyclic graph.
    We also use the edge-label style in bolinas.
    """
    def __init__(self,*args, **kwargs):

        defaultdict.__init__(self,ListMap,*args,**kwargs)
        self.roots = []
        self.external_nodes = {}

        # attributes to be added
        self.node_to_concepts = {}
        self.align_to_sentence = None

        self.reentrance_triples = []

    def get_variable(self,posID):
        """return variable given postition ID"""
        reent_var = None
        seq = self.dfs()[0]
        for node in seq:
            if node.seqID == posID:
                return node.node_label
        return None

    def get_match(self, subgraph):
        """find the subgraph"""
        def is_match(dict1, dict2):
            rel_concept_pairs = []
            for rel, cpt in dict2.items():
                rel_concept_pairs.append(rel+'@'+cpt)
                if not (rel in dict1 and cpt in dict1[rel]):
                    return None
            return rel_concept_pairs
        subroot = subgraph.keys()[0] # sub root's concept
        concepts_on_the_path = []

        for v in self.node_to_concepts:
            if v[0] == subroot[0] and self.node_to_concepts[v] == subroot:
                concepts_on_the_path = [subroot]
                rcp = is_match(self[v], subgraph[subroot])
                if rcp is not None: return v, concepts_on_the_path+rcp
                #for rel, cpt in subgraph[subroot].items():
                #    if rel in self[v] and cpt in self[v][rel]:
                #        concepts_on_the_path.append(rel+'@'+cpt)
        return None, None

    def get_pid(self,var):
        seq = self.dfs()[0]
        for node in seq:
            if node.node_label == var:
                return node.seqID
        return None
        '''
        posn_queue = posID.split('.')
        var_list = self.roots
        past_pos_id = []
        while posn_queue:
            posn = int(posn_queue.pop(0))
            past_pos_id.append(posn)
            print(var_list,past_pos_id,posn,visited_var)
            variable = var_list[posn]
            var_list = []
            vars = [v[0] for v in self[variable].values()]
            i = 0
            while i < len(vars):
                k = vars[i]
                if k not in visited_var:
                    var_list.append(k)
                elif isinstance(k,(StrLiteral,Quantity)):
                    var_list.append(k)
                else:
                    if visited_var[k] == '.'.join(str(j) for j in past_pos_id+[i]):
                        var_list.append(k)
                    else:
                        vars.pop(i)
                        i -= 1

                i += 1

        '''
        return variable

    def _add_reentrance(self,parent,relation,reentrance):
        if reentrance:
            self.reentrance_triples.append((parent,relation,reentrance[0]))

    def _add_triple(self, parent, relation, child, warn=None):
        """
        Add a (parent, relation, child) triple to the DAG.
        """
        if type(child) is not tuple:
            child = (child,)
        if parent in child:
            #raise Exception('self edge!')
            #sys.stderr.write("WARNING: Self-edge (%s, %s, %s).\n" % (parent, relation, child))
            if warn: warn.write("WARNING: Self-edge (%s, %s, %s).\n" % (parent, relation, child))
            #raise ValueError, "Cannot add self-edge (%s, %s, %s)." % (parent, relation, child)
        for c in child:
            x = self[c]
            for rel, test in self[c].items():
                if parent in test:
                   if warn:
                       warn.write("WARNING: (%s, %s, %s) produces a cycle with (%s, %s, %s)\n" % (parent, relation, child, c, rel, test))
                       #ATTENTION:maybe wrong, test may not have only one element, deal with it later
                       concept1 = self.node_to_concepts[parent]
                       concept2 = self.node_to_concepts[test[0]]
                       #print(concept1,concept2)
                       if concept1 != concept2:
                           warn.write("ANNOTATION ERROR: concepts %s and %s have same node label %s!" % (concept1, concept2, parent))

                    #raise ValueError,"(%s, %s, %s) would produce a cycle with (%s, %s, %s)" % (parent, relation, child, c, rel, test)

        self[parent].append(relation, child)

    def set_alignment(self,alignment):
        self.align_to_sentence = alignment

    def print_triples(self):
        result = ''
        amr_triples = self.bfs()[1]
        for rel,parent,child in amr_triples:
            if not isinstance(child,(Quantity,Polarity,Interrogative,StrLiteral)):
                result += "%s(%s,%s)\n"%(rel,self.node_to_concepts[parent],self.node_to_concepts[child])
            else:
                result += "%s(%s,%s)\n"%(rel,self.node_to_concepts[parent],child)
        return result

    def dfs(self):
        """
        depth first search for the graph
        return dfs ordered nodes and edges
        TO-DO: this visiting order information can be obtained
        through the reading order of amr strings; modify the class
        to OrderedDefaultDict;
        """
        visited_nodes = set()
        visited_edges = []
        sequence = []

        #print('roots', self.roots)
        #for i, r in enumerate(self.roots):
        #    print("what:", i, r)
        for i,r in enumerate(self.roots):
            seqID = str(i)
            #print('root info:', i, r)
            stack = [((r,),None,None,0,seqID)] # record the node, incoming edge, parent, depth and unique identifier

            #all_nodes = []
            while stack:
                next,rel,parent,depth,seqID = stack.pop()
                for n in next:
                    if self.reentrance_triples:
                        firsthit = (parent,rel,n) not in self.reentrance_triples
                    else:
                        firsthit = n not in visited_nodes
                    leaf = False if self[n] else True

                    node = Node(parent, rel, n, firsthit, leaf, depth, seqID)

                    #print(self.node_to_concepts)
                    sequence.append(node)

                    # same StrLiteral/Quantity/Polarity should not be revisited
                    if self.reentrance_triples: # for being the same with the amr string readed in
                        if n in visited_nodes or (parent,rel,n) in self.reentrance_triples:
                            continue
                    else:
                        if n in visited_nodes:
                            continue

                    visited_nodes.add(n)
                    # TODO: add relation order here...
                    p = len([child for rel,child in self[n].items() if (n,rel,child[0]) not in self.reentrance_triples]) - 1
                    for rel, child in reversed(self[n].items()):
                        #print(rel,child)
                        if not (rel, n, child[0]) in visited_edges:
                            #if child[0] not in visited_nodes or isinstance(child[0],(StrLiteral,Quantity)):
                            visited_edges.append((rel,n,child[0]))
                            if (n,rel,child[0]) not in self.reentrance_triples:
                                stack.append((child,rel,n,depth+1,seqID+'.'+str(p)))
                                p -= 1
                            else:
                                stack.append((child,rel,n,depth+1,None))
                        elif isinstance(child[0],(StrLiteral,Quantity)):
                            stack.append((child,rel,n,depth+1,seqID+'.'+str(p)))
                            p -= 1
                        else:
                            pass


        return (sequence, visited_edges)

    def replace_node(self, h_idx, idx):
        """for coreference, replace all occurrence of node idx to h_idx"""
        visited_nodes = set()
        visited_edges = set()

        for i,r in enumerate(self.roots[:]):
            stack = [((r,),None,None)] #node,incoming edge and preceding node

            while stack:
                next, rel, previous = stack.pop()
                for n in next:
                    if n == idx:
                        if previous == None: # replace root
                            self.roots[i] = h_idx
                            break
                        self[previous].replace(rel,(h_idx,))
                    if n in visited_nodes:
                        continue
                    visited_nodes.add(n)
                    for rel, child in reversed(self[n].items()):
                        if not (n, rel, child) in visited_edges:
                            if child in visited_nodes:
                                stack.append((child,rel,n))
                            else:
                                visited_edges.add((n,rel,child))
                                stack.append((child,rel,n))

    def find_rel(self,h_idx,idx):
        """find the relation between head_idx and idx"""
        rels = []
        for rel,child in self[h_idx].items():
            #print(child,idx)
            if child == (idx,):
                rels.append(rel)
        return rels

    def replace_head(self,old_head,new_head,KEEP_OLD=True):
        """change the focus of current sub graph"""
        for rel,child in self[old_head].items():
            if child != (new_head,):
                self[new_head].append(rel,child)
        del self[old_head]
        if KEEP_OLD:
            foo = self[old_head]
            self[new_head].append('NA',(old_head,))

    def replace_rel(self,h_idx,old_rel,new_rel):
        """replace the h_idx's old_rel to new_rel"""
        for v in self[h_idx].getall(old_rel):
            self[h_idx].append(new_rel,v)
        del self[h_idx][old_rel]
    '''
    def rebuild_index(self, node, sent_index_mapping=None):
        """assign non-literal node a new unique node label; replace the
           original index with the new node id or sentence offset;
           if we have been provided the sentence index mapping, we use the
           sentence offsets as new node label instead of the serialized node id.
        """
        if sent_index_mapping is None:
            if node.node_label in self.node_to_concepts and self.node_to_concepts[node.node_label] is not None:
                #update the node_to_concepts table
                self.node_to_concepts[Node.node_id] = self.node_to_concepts[node.node_label]
                del self.node_to_concepts[node.node_label]
                Node.mapping_table[node.node_label] = Node.node_id
                node.node_label = Node.node_id

            elif self.node_label not in node_to_concepts and self.node_label in Node.mapping_table:
                new_label = Node.mapping_table[self.node_label]
                self.node_label = new_label
            else:
                #print(Node.node_id,self.node_label)
                node_to_concepts[Node.node_id] = self.node_label
                self.node_label = Node.node_id

    '''

    def is_named_entity(self, var):
        edge_label_set = self[var].keys()
        if 'name' in edge_label_set:
            try:
                assert 'wiki' in edge_label_set
            except:
                print('ill-formed entity found')
                print(self.to_amr_string())
                return False
            return True
        return False

    def is_entity(self, var):
        if var in self.node_to_concepts:
            var_concept = self.node_to_concepts[var]
            return var_concept.endswith('-entity') or var_concept.endswith('-quantity') or var_concept.endswith('-organization') or var_concept == 'amr-unknown'
        return False

    def is_predicate(self, var):
        if var in self.node_to_concepts:
            return re.match('.*-[0-9]+',self.node_to_concepts[var]) is not None
        return False

    def is_const(self, var):
        return var not in self.node_to_concepts

    def statistics(self):
        #sequence = self.dfs()[0]
        named_entity_nums = defaultdict(int)
        entity_nums = defaultdict(int)
        predicate_nums = defaultdict(int)
        variable_nums = defaultdict(int)
        const_nums = defaultdict(int)
        reentrancy_nums = 0

        stack = [(self.roots[0],None,None,0)]

        while stack:
            cur_var, rel, parent, depth = stack.pop()
            exclude_rels = []
            if (parent, rel, cur_var) in self.reentrance_triples: # reentrancy here
                reentrancy_nums += 1
                continue
            if self.is_named_entity(cur_var):
                entity_name = self.node_to_concepts[cur_var]
                named_entity_nums[entity_name] += 1

                exclude_rels = ['name','wiki']
            elif self.is_entity(cur_var): # entity does not have name relation
                entity_name = self.node_to_concepts[cur_var]
                entity_nums[entity_name] += 1

            elif self.is_predicate(cur_var):
                pred_name = self.node_to_concepts[cur_var]
                predicate_nums[pred_name] += 1

            elif self.is_const(cur_var):
                const_nums[cur_var] += 1
            else:
                variable_name = self.node_to_concepts[cur_var]
                variable_nums[variable_name] += 1

            for rel, var in self[cur_var].items():
                if rel not in exclude_rels:
                    stack.append((var[0], rel, cur_var, depth+1))
        return named_entity_nums,entity_nums,predicate_nums,variable_nums,const_nums,reentrancy_nums

    def to_amr_string(self, indent_size=4):

        indent = " " * indent_size
        amr_string = ""

        seq = self.dfs()[0]
        #print(seq)

        if len(seq) == 0:
            return "(e / EMPTY-AMR)"

        #always begin with root
        assert seq[0].trace == None
        dep_rec = 0

        for (i, node) in enumerate(seq):

            #in case of multiple roots, finish the previous one
            if i != 0 and node.depth == 0:
                if dep_rec != 0:
                    amr_string += "%s"%((dep_rec)*')')
                else:
                    amr_string += ')'
                dep_rec = 0

            if node.trace == None:
                # Start a new line if this is not the first root.
                if i != 0:
                    amr_string += "\n"
                if node.firsthit and node.node_label in self.node_to_concepts:
                    amr_string += "(%s / %s"%(node.node_label,self.node_to_concepts[node.node_label])
                else:
                    amr_string += "(%s"%(node.node_label)
            else:
                if node.depth >= dep_rec:
                    dep_rec = node.depth
                else:
                    amr_string += "%s"%((dep_rec-node.depth)*')')
                    dep_rec = node.depth


                if not node.leaf:
                    if node.firsthit and node.node_label in self.node_to_concepts:
                        amr_string += "\n%s:%s (%s / %s"%(node.depth*indent,node.trace,node.node_label,self.node_to_concepts[node.node_label])
                    else:
                        amr_string += "\n%s:%s %s"%(node.depth*indent,node.trace,node.node_label)

                else:
                    if node.firsthit and node.node_label in self.node_to_concepts:
                        amr_string += "\n%s:%s (%s / %s)"%(node.depth*indent,node.trace,node.node_label,self.node_to_concepts[node.node_label])
                    else:
                        if isinstance(node.node_label,StrLiteral):
                            amr_string += '\n%s:%s "%s"'%(node.depth*indent,node.trace,node.node_label)
                        else:
                            amr_string += "\n%s:%s %s"%(node.depth*indent,node.trace,node.node_label)


        if dep_rec != 0:
            amr_string += "%s"%((dep_rec)*')')
        else:
            amr_string += ')'

        return amr_string

    def __reduce__(self):
        t = defaultdict.__reduce__(self)
        return (t[0], ()) + (self.__dict__,) + t[3:]

if __name__ == "__main__":

    opt = OptionParser()
    opt.add_option("-v", action="store_true", dest="verbose")

    (options, args) = opt.parse_args()

    s = '''(a / and :op1(恶化 :ARG0(它) :ARG1(模式 :mod(开发)) :time (已经)) :op2(t / 堵塞 :ARG0(它) :ARG1(交通 :mod(局部)) :location(a / around :op1(出口))))'''
    s1 = '''(a  /  and :op1 (c  /  change-01 :ARG0 (i  /  it) :ARG1 (p  /  pattern :mod (d  /  develop-02)) :ARG2 (b  / bad :degree (m  /  more))) :op2 (c2  /  cause-01 :ARG0 i :ARG1 (c3  /  congest-01 :ARG1 (a2  /  around :op1 (e  /  exit :poss i)) :ARG2 (t  /  traffic) :ARG1-of (l2  /  localize-01))) :time (a3  /  already))'''
    s = s.decode('utf8')
    #amr_ch = AMR.parse_string(s)
    amr_en = AMR.parse_string(s1)

    #print(str(amr_en))
    #print(amr_en.dfs())
    print(amr_en.to_amr_string())
    #print(amr_ch)
    #print(amr_ch.dfs())
    #print(amr_ch.to_amr_string())
