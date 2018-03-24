Code consists of:

class GSOM_Node 		- which initializes nodes with weights, no neighbouring nodes, and iterations
				- method adjust_weights which updates weights according to formula on the wiki page (refer https://en.wikipedia.org/wiki/Growing_self-organizing_map )
				- method is_boundary which checks if the node is at boundary of map, and returns boolean value


method _distance 		- takes a matrix and it's weights as inputs
		 		- returns summation of squares of differences of input values. used by _find_bmu

method _find_bmu 		- takes a matrix as input
		 		- returns the winning node, that is, node having minimum distance

method _find_similar_boundary   - takes a node as input and returns a node with similar boundary, or False

method _grow			- the winning node is provided as input
				- the node inserts a node in all 4 possible directions if it does not exist
				- returns entire list of nodes
Input:

150 rows, 4 columns

Algorithm ran for 100 iterations and produced the following results. Initially 4 nodes are placed at (10,10), (10,11), (11,11), (11,10).
Ending positions of all columns ('sepal length', 'sepal width', 'petal length', 'petal width') are completely random and can be viewed(from graph, some are overlapped also, take a look at individual clustering for clarity if needed) from the clustering performed.

Output:

Column 1: 12816 neurons generated 
Column 2: 13117 neurons generated
Column 3: 12937 neurons generated
Column 4: 12710 neurons generated

The neurons generated can also be removed from the map if it finds a different BMU.

How to Run:
1. Install Jupyter notebook with a Python 3 Kernel.
2. Download notebook from Github link
3. Click on Run > Run all.
