# distutils: language = c++
from libcpp.vector cimport vector


cdef struct box2d:
    double xmin, xmax, ymin, ymax


cdef bint boxes_intersect_2d(box2d box1, box2d box2):
    '''
    Test if there is any intersection between `box1` and `box2`. An
    intersection is detected if any part of the boxes touch, even at a
    single point.
    '''
    if box1.xmin > box2.xmax:
        return False
    elif box2.xmin > box1.xmax:
        return False
    elif box1.ymin > box2.ymax:
        return False
    elif box2.ymin > box1.ymax:
        return False
    else:
        return True


cdef class _QuadNode:
    cdef readonly:
        # the bounds for the node
        box2d bounds
        # the node that generated this node (which can be None for the
        # root Node)
        _QuadNode parent
        # the depth of the current node
        long depth
        # the greatest possible depth for this tree
        long max_depth
        # the direct descendants for this node
        tuple children
        # the boxes contained in this node and their corresponding
        # indices. A vector is like a list except it is more efficient
        # because its items are typed
        vector[(long, box2d)] boxes
        # the total number of boxes that have been added to the node
        # and its childen. This is not necessarily equal to len(boxes)
        long box_count

    def __init__(self,
                 box2d bounds,
                 _QuadNode parent,
                 long depth,
                 long max_depth):
        self.max_depth = max_depth
        self.depth = depth
        self.bounds = bounds
        self.parent = parent
        self.box_count = 0
        self.children = ()
        self.boxes = [] # this automatically gets coerced to a vector

    def __repr__(self):
        out = ('< QuadNode: x=(%s, %s), y=(%s, %s), depth=%s/%s, '
               'box count=%s >' %
               (self.bounds.xmin, self.bounds.xmax,
                self.bounds.ymin, self.bounds.ymax,
                self.depth, self.max_depth,
                self.box_count))
        return out

    cdef void subdivide_nodes(self):
        '''
        Create children for the node. This should only be called if
        children do not already exist.
        '''
        cdef:
            double dx, dy
            box2d bounds1, bounds2, bounds3, bounds4
            _QuadNode child1, child2, child3, child4

        # if we have reached the max depth, then do not make any
        # children
        if self.depth == self.max_depth:
            return

        dx = self.bounds.xmax - self.bounds.xmin
        dy = self.bounds.ymax - self.bounds.ymin

        bounds1.xmin = self.bounds.xmin
        bounds1.xmax = self.bounds.xmin + dx/2
        bounds1.ymin = self.bounds.ymin
        bounds1.ymax = self.bounds.ymin + dy/2

        bounds2.xmin = self.bounds.xmin + dx/2
        bounds2.xmax = self.bounds.xmin + dx
        bounds2.ymin = self.bounds.ymin
        bounds2.ymax = self.bounds.ymin + dy/2

        bounds3.xmin = self.bounds.xmin
        bounds3.xmax = self.bounds.xmin + dx/2
        bounds3.ymin = self.bounds.ymin + dy/2
        bounds3.ymax = self.bounds.ymin + dy

        bounds4.xmin = self.bounds.xmin + dx/2
        bounds4.xmax = self.bounds.xmin + dx
        bounds4.ymin = self.bounds.ymin + dy/2
        bounds4.ymax = self.bounds.ymin + dy

        child1 = _QuadNode(
            bounds1, self, self.depth + 1, self.max_depth)
        child2 = _QuadNode(
            bounds2, self, self.depth + 1, self.max_depth)
        child3 = _QuadNode(
            bounds3, self, self.depth + 1, self.max_depth)
        child4 = _QuadNode(
            bounds4, self, self.depth + 1, self.max_depth)
        self.children = (child1, child2, child3, child4)

    cdef bint contains_box(self, box2d bx):
        '''
        Identifies whether the node completely contains `bx`.
        '''
        # if the lower bounds of `self` are not below `bx`, then
        # return false
        if self.bounds.xmin >= bx.xmin:
            return False
        elif self.bounds.ymin >= bx.ymin:
            return False

        # if the upper bounds of `self` are not above `bx`, then
        # return false
        elif self.bounds.xmax <= bx.xmax:
            return False
        elif self.bounds.ymax <= bx.ymax:
            return False
        else:
            return True

    cdef void add_box(self, long idx, box2d bx):
        '''
        Adds a box to the smallest possible bounding node. If the
        smallest possible bounding node does not yet exist, then it
        will be created.

        If the box does not fit in any of the child nodes, then the
        box will be added to `self`, regardless of whether the box is
        contained in `self`. To avoid erroneous behavior, this method
        should only be called from the top-level node or if it has
        been verified that the box is indeed contained in `self`.

        The box should be uniquely identifiable by `idx`
        '''
        cdef:
            _QuadNode child
            (long, box2d) item = (idx, bx)
            
        # the box will either be added to the current node or one of
        # its children so increment `box_count`
        self.box_count += 1
        # To test whether an item belongs in a node, we need to know
        # if the item can fit in any of the nodes children. If the
        # node has no children, then give it some.
        if not self.children:
            self.subdivide_nodes()

        for child in self.children:
            if child.contains_box(bx):
                child.add_box(idx, bx)
                return

        # if we reach this point, then the item is contained in no
        # children, so we add the item to `self`. `push_back` is like
        # `append` for vectors
        self.boxes.push_back(item)
        return

    cdef _QuadNode smallest_bounding_node(self, box2d bx):
        '''
        Find the smallest existing node that completely contains
        the box. This will not create any new nodes.

        If the box does not fit in any of the child nodes, then `self`
        will be returned, regardless of whether the box is contained
        in `self`. To avoid erroneous behavior, this method should
        only be called from the top-level node or if it has been
        verified that `bx` is indeed contained in `self`.
        '''
        cdef:
            _QuadNode child

        for child in self.children:
            if child.contains_box(bx):
                return child.smallest_bounding_node(bx)

        # if we reach this point, then the item is contained in no
        # children, so `self` is the smallest bounding node.
        return self

    cdef list descendant_nodes(self):
        '''
        Returns a list of all the nodes that descended from `self`
        '''
        cdef:
            _QuadNode child
            list nodes = []

        nodes.extend(self.children)
        for child in self.children:
            nodes.extend(child.descendant_nodes())

        return nodes

    cdef list ancestor_nodes(self):
        '''
        Returns a list of all the ancestors for `self`
        '''
        cdef:
            list nodes = []

        if self.parent is None:
            return nodes

        nodes.append(self.parent)
        nodes.extend(self.parent.ancestor_nodes())
        return nodes

    cpdef void prune(self):
        '''
        Recursively removes a nodes children if the children have no
        children of their own and if none of the children have boxes.
        '''
        cdef:
            _QuadNode child
            bint remove_children = True

        for child in self.children:
            child.prune()
            if child.children:
                remove_children = False
            
            elif child.boxes.size() != 0:
                remove_children = False
        
        if remove_children:
            self.children = ()
            
        return


cdef class QuadTree(_QuadNode):
    '''
    The top-level QuadTree class. This is used to efficiently find
    which boxes, among a large collection of boxes, intersect a query
    box. This class is functionally similar to RTree, except the build
    time is significantly shorter, while the query time is longer.

    Parameters
    ----------
    bounds : (4,) float array
        The bounds for the quad tree as (xmin, ymin, xmax, ymax). This
        defines the area that is recursively bisected. The data does
        not need to fall within these bounds, although the algorithm
        will be less efficient if they do not.

    max_depth : int, optional

    '''
    def __init__(self, double[:] bounds, long max_depth=5):
        cdef:
            box2d bx

        bx.xmin = bounds[0]
        bx.ymin = bounds[1]
        bx.xmax = bounds[2]
        bx.ymax = bounds[3]
        super().__init__(bx, None, 1, max_depth)

    def add_boxes(self, double[:, :] boxes):
        '''
        Adds the boxes to the QuadTree

        Parameters
        ----------
        boxes : (n, 4) float array
            Boxes described as (xmin, ymin, xmax, ymax)

        '''

        cdef:
            long i
            box2d bx

        for i in range(boxes.shape[0]):
            bx.xmin = boxes[i, 0]
            bx.ymin = boxes[i, 1]
            bx.xmax = boxes[i, 2]
            bx.ymax = boxes[i, 3]
            self.add_box(self.box_count, bx)

    def intersections(self, double[:] box):
        '''
        Finds all the boxes that intersect `box`.

        Parameters
        ----------
        box : (4,) float array
            Query box described as (xmin, ymin, xmax, ymax)

        Returns
        -------
        (k,) list of ints
            The indices of the boxes that intersect `box`. The indices
            correspond to the order in which the boxes were added to
            the quad tree.

        '''
        cdef:
            long i
            box2d bx1, bx2
            _QuadNode node, member
            list family = []
            vector[long] indices = []

        bx1.xmin = box[0]
        bx1.ymin = box[1]
        bx1.xmax = box[2]
        bx1.ymax = box[3]
        # find the smallest node that `box` fits in, and then test
        # whether that box intersects any boxes in related nodes
        node = self.smallest_bounding_node(bx1)
        family.append(node)
        family.extend(node.ancestor_nodes())
        family.extend(node.descendant_nodes())
        for member in family:
            for i, bx2 in member.boxes:
                if boxes_intersect_2d(bx1, bx2):
                    indices.push_back(i)

        return indices
