
# RNN-T decode

class Node:
    def __init__(self, label, embed=None, states=None, node=None):
        self.label = label
        self.embed = embed
        self.states = states
        self.score = 0.
        self.prefix = []
        self.score = 0.
        
# initialize by blank
embed, states = decoder(Tensor(blank), h=None)
node = Node(blank, embed, states)
open_nodes.append(Node(blank))
for i in range(T):
    candidates = []
    for nd in open_nodes:
        ytu = joint(x[i], nd.embed)
        out = F.log_softmax(ytu, dim=vocab_dim) # vocab dim
        probs, preds = torch.topk(out, K, dim=vocab_dim)
        for k in range(len(probs)):
            node.prev = node.label
            node.label = int(preds[k])
            node.score = nd.score + float(torch.log(probs[k]))
            node.embed = nd.embed
            node.states = nd.states
            node.prefix = nd.prefix
            node.prefix.append(int(preds[k]))
            candidates.append(node)
        candidates = sorted(candidates, key=lambda o: o.score)
    candidates = candidates[:Nbest]
    finals = [ n for n in candidates if n.label == eos_token ]
    open_nodes = [ n for n in candidates if n.label != eos_token ]
    if len(open_nodes) == 0:
        break
    for nd in open_nodes:
        if nd.label != blank and nd.prev != nd.label:
            embed, states = decoder(Tensor(nd.label), nd.states)
            nd.states = states
            nd.embed = embed
            
