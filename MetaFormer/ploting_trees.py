from ete3 import Tree, TreeStyle


def build_taxa_dict(taxonomies, abundances):
    """
    Returns a nested dictionary, merging shared prefixes but *without* summing abundances.
    Each leaf node has:
      {
        "_is_leaf": True,
        "_abundance": <float>
      }
    For example, if one path stops at 'f__X' with abundance 0.1
    and another path continues down to 'g__Y' with abundance 0.2,
    'f__X' becomes a leaf in the dictionary, and 'g__Y' is a sub-node.
    """
    tree_dict = {}

    for tax_str, ab in zip(taxonomies, abundances):
        if not tax_str or tax_str == "<pad>":
            # Skip empty or padding
            continue
        
        ranks = tax_str.split(";")  # e.g. ['d__Bacteria', 'p__Firmicutes', ...]
        node = tree_dict

        for i, rank in enumerate(ranks):
            if rank not in node:
                node[rank] = {}  # create child
            # If this is the final rank (a leaf in this path)
            if i == len(ranks) - 1:
                node[rank]["_is_leaf"] = True
                node[rank]["_abundance"] = ab
            # Move deeper
            node = node[rank]

    return tree_dict


def dict_to_ete3(tree_dict, node_name="Root", parent=None):
    """
    Recursively build an ete3 Tree from our prefix-tree dictionary.
    - If a node has '_is_leaf': True, we attach 'abundance' to it.
    - Otherwise, it's just an internal node.
    """
    if parent is None:
        # This is the root
        t = Tree()
        n = t
        n.name = node_name
    else:
        # Create a child node
        n = parent.add_child(name=node_name)

    # If this node is a leaf, attach abundance
    is_leaf = tree_dict.get("_is_leaf", False)
    if is_leaf:
        n.add_feature("abundance", tree_dict["_abundance"])

    # For each sub-rank (child key), recurse
    for key, subdict in tree_dict.items():
        if key.startswith("_"):
            # skip special fields
            continue
        dict_to_ete3(subdict, node_name=key, parent=n)

    return t if parent is None else None


### to show abundances 
from ete3 import TreeStyle, NodeStyle, TextFace, add_face_to_node

def my_layout_fn(node):
    """
    Show node.name normally. If the node has an 'abundance' feature, display it too.
    """
    if hasattr(node, "abundance"):
        # This is a leaf node with a known abundance
        label = f"{node.name} ({node.abundance:.3f})"
    else:
        # Just show the rank label
        label = node.name

    face = TextFace(label, fsize=10)
    # Put the label on the right side of the branch
    add_face_to_node(face, node, column=0, position="branch-right")