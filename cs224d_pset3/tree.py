import random

class Node:
    def __init__(self, label, word=None):
        self.label = label
        self.word = word
        self.parent = None
        self.left = None
        self.right = None
    
        self.isLeaf = False


class Tree:
    def __init__(self, treeString, openChar='(', closeChar=')'):
        tokens = []
        self.open = '('
        self.close = ')'
        for toks in treeString.strip().split():
            tokens += list(toks)
        self.root = self.parse(tokens)
        self.labels = get_labels(self.root)
        self.num_words = len(self.labels)

    def parse(self, tokens, parent=None):
        assert tokens[0] == self.open, 'Malformed tree'
        assert tokens[-1] == self.close, 'Malformed tree'

        split = 2
        countOpen = countClose = 0
        if tokens[split] == self.open:
            countOpen += 1
            split += 1
 
        while countOpen != countClose:
            if tokens[split] == self.open:
                countOpen += 1
            if tokens[split] == self.close:
                countClose += 1
            split += 1 
  
        node = Node(int(tokens[1]))
        node.parent = parent
    
        if countOpen == 0:
            node.word = ''.join(tokens[2:-1]).lower()
            node.isLeaf = True
            return node
        
        node.left = self.parse(tokens[2:split], parent = node)
        node.right = self.parse(tokens[split:-1], parent = node)
  
        return node
  
    def get_words(self):
        leaves = getLeaves(self.root) 
        words = [node.word for node in leaves]
        return words


def leftTraverse(node, nodeFn=None, args=None):
    if node is None:
        return
    leftTraverse(node.left, nodeFn, args)
    leftTraverse(node.right, nodeFn, args)
    nodeFn(node, args)
    

def getLeaves(node):
    if node is None:
        return []
    if node.isLeaf:
        return [node]
    else:
        return getLeaves(node.left) + getLeaves(node.right)

def get_labels(node):
    if node is None:
        return []
    return get_labels(node.left) + get_labels(node.right) + [node.label]

def loadTrees(dataSet = 'train'):
    file = 'trees/%s.txt' % dataSet
    print "loading %s trees.." % dataSet
    with open(file, 'r') as fid:
        trees = [Tree(l) for l in fid.readlines()]
    return trees

def simplified_data(num_train, num_dev, num_test):
    rndstate = random.getstate()
    random.seed(0)
    trees = loadTrees('train') + loadTrees('dev') + loadTrees('test')
    
    #print '-===============-'
    #print len(trees)
    #print trees[0].get_words()
    
    pos_trees = [t for t in trees if t.root.label == 4]
    neg_trees = [t for t in trees if t.root.label == 0]
    
    binarize_labels(pos_trees)
    binarize_labels(neg_trees)
    
    #print len(pos_trees), len(neg_trees)
    #print pos_trees[0].labels
   
    pos_trees = sorted(pos_trees, key=lambda t: len(t.get_words()))
    neg_trees = sorted(neg_trees, key=lambda t: len(t.get_words()))
    num_train/=2
    num_dev/=2 
    num_test/=2
    train = pos_trees[:num_train] + neg_trees[:num_train]
    dev = pos_trees[num_train : num_train+num_dev] + neg_trees[num_train : num_train+num_dev]
    test = pos_trees[num_train+num_dev : num_train+num_dev+num_test] + neg_trees[num_train+num_dev : num_train+num_dev+num_test]
    random.shuffle(train)
    random.shuffle(dev)
    random.shuffle(test)
    random.setstate(rndstate)

    return train, dev, test  
    
    
def binarize_labels(trees):
    def binarize_node(node, _):
        if node.label<2:
            node.label = 0
        elif node.label>2:
            node.label = 1
    for tree in trees:
        leftTraverse(tree.root, binarize_node, None)
        tree.labels = get_labels(tree.root)

"""
train, dev, test = simplified_data(700,100,200)
print len(train)
print len(dev)
print len(test)
"""
