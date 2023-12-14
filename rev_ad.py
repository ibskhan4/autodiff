from .rev_variable import *
from collections import defaultdict

def find_gradients(var):

    # Helper function for defaultdict such that 0 is returned for non-present keys
    def return_zero():
        return 0

    # Creating the dictionary to store the gradients for all the nodes in the graph
    gradients_dict = defaultdict(return_zero)

    # Creating a set to eventually store the leaf nodes for which we ultimately wish to store the partial derivatives with respect to
    leafs = set()

    # Recursively computing the gradients for each child node
    def child_gradients(var, path_val):
        """
        Recursively computes local derivatives for each child and updates gradient with respect to each child; multiplies local derivates along same path
        and sums different paths leading to same child node
        """
        # Captures the leaf nodes (i.e. our input variables) such that we can retrieve the final partial derivative calculations with respect to these
        # variables once the recursive iterations have been fully run
        if not var.local_gradients:
            leafs.add(var)

        # Loops through each child node for a given variable
        for child, local_gradient in var.local_gradients:
            # Multiply edges of path to compute summative value of given path to child
            val_path_to_child = path_val * local_gradient

            # Sum all paths to child
            gradients_dict[child] += val_path_to_child

            # Recursively move through graph to compute all gradients
            child_gradients(child, val_path_to_child)

    # path_val is 1 to begin with since initial value is derivative of node with respect to itself, which is 1
    child_gradients(var, 1)

    # Creating list to store partial derivatives with respect to the input ReverseVariables
    leaf_gradients = []

    # Looping through stored leafs and appending to above list
    for leaf in leafs:
        leaf_gradients.append(gradients_dict[leaf])

    return leaf_gradients

def rev_jacobian(fns):
    """
    Returns the Jacobian given a function or a list of functions in the form of ReverseVariable objects

    Parameters
    ----------
    fns: NumPy array of ReverseVariable objects, list of ReverseVariable objects, or a ReverseVariable object

    Returns
    -------
    rev_jacobian: NumPy array representing the Jacobian
    """

    # Check what the input parameters' types are to return the appropriate result
    # If the user inputs a single function fns
    if isinstance(fns, np.ndarray) or isinstance(fns, list):
        if not all([isinstance(f, ReverseVariable) for f in fns]):
            raise TypeError("Parameter elements must all be ReverseVariable objects")

        return np.vstack([find_gradients(f) for f in fns])
    elif isinstance(fns, ReverseVariable):
        return find_gradients(fns)
    else:
        raise TypeError("Parameter must be a NumPy array of ReverseVariable objects, a list of ReverseVariable objects, or a ReverseVariable object")
