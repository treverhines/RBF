
cdef struct box3d:
    double xmin, xmax, ymin, ymax, zmin, zmax


cdef bint boxes_intersect_3d(box3d box1, box3d box2):
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
    elif box1.zmin > box2.zmax:
        return False
    elif box2.zmin > box1.zmax:
        return False
    else:
        return True


cdef class _OctNode:
    cdef readonly:
        # the bounds for the node
        box3d bounds
        # the node that generated this node (which can be None for the
        # root Node)
        _OctNode parent
        # the depth of the current node
        long depth
        # the greatest possible depth for this tree
        long max_depth
        # the direct descendants for this node
        list children
        # the boxes contained in this node and their corresponding
        # indices
        list boxes
        # the total number of boxes that have been added to the node
        # and its childen. This is not necessarily equal to len(boxes)
        long box_count

    def __init__(self,
                 box3d bounds,
                 _OctNode parent,
                 long depth,
                 long max_depth):
        self.max_depth = max_depth
        self.depth = depth
        self.bounds = bounds
        self.parent = parent
        self.box_count = 0
        self.children = []
        self.boxes = []

    def __repr__(self):
        out = ('< OctNode: x=(%s, %s), y=(%s, %s), z=(%s, %s), '
               'depth=%s/%s, box count=%s >' %
               (self.bounds.xmin, self.bounds.xmax,
                self.bounds.ymin, self.bounds.ymax,
                self.bounds.zmin, self.bounds.zmax,
                self.depth, self.max_depth,
                self.box_count))
        return out

    cdef void subdivide_nodes(self):
        '''
        Create children for the node. This should only be called if
        children do not already exist.
        '''
        cdef:
            double dx, dy, dz
            box3d bounds1, bounds2, bounds3, bounds4
            box3d bounds5, bounds6, bounds7, bounds8
            _OctNode child1, child2, child3, child4
            _OctNode child5, child6, child7, child8

        # if we have reached the max depth, then do not make any
        # children
        if self.depth == self.max_depth:
            return

        dx = self.bounds.xmax - self.bounds.xmin
        dy = self.bounds.ymax - self.bounds.ymin
        dz = self.bounds.zmax - self.bounds.zmin

        bounds1.xmin = self.bounds.xmin
        bounds1.xmax = self.bounds.xmin + dx/2
        bounds1.ymin = self.bounds.ymin
        bounds1.ymax = self.bounds.ymin + dy/2
        bounds1.zmin = self.bounds.zmin
        bounds1.zmax = self.bounds.zmin + dz/2

        bounds2.xmin = self.bounds.xmin + dx/2
        bounds2.xmax = self.bounds.xmin + dx
        bounds2.ymin = self.bounds.ymin
        bounds2.ymax = self.bounds.ymin + dy/2
        bounds2.zmin = self.bounds.zmin
        bounds2.zmax = self.bounds.zmin + dz/2

        bounds3.xmin = self.bounds.xmin
        bounds3.xmax = self.bounds.xmin + dx/2
        bounds3.ymin = self.bounds.ymin + dy/2
        bounds3.ymax = self.bounds.ymin + dy
        bounds3.zmin = self.bounds.zmin
        bounds3.zmax = self.bounds.zmin + dz/2

        bounds4.xmin = self.bounds.xmin + dx/2
        bounds4.xmax = self.bounds.xmin + dx
        bounds4.ymin = self.bounds.ymin + dy/2
        bounds4.ymax = self.bounds.ymin + dy
        bounds4.zmin = self.bounds.zmin
        bounds4.zmax = self.bounds.zmin + dz/2

        bounds5.xmin = self.bounds.xmin
        bounds5.xmax = self.bounds.xmin + dx/2
        bounds5.ymin = self.bounds.ymin
        bounds5.ymax = self.bounds.ymin + dy/2
        bounds5.zmin = self.bounds.zmin + dz/2
        bounds5.zmax = self.bounds.zmin + dz

        bounds6.xmin = self.bounds.xmin + dx/2
        bounds6.xmax = self.bounds.xmin + dx
        bounds6.ymin = self.bounds.ymin
        bounds6.ymax = self.bounds.ymin + dy/2
        bounds6.zmin = self.bounds.zmin + dz/2
        bounds6.zmax = self.bounds.zmin + dz

        bounds7.xmin = self.bounds.xmin
        bounds7.xmax = self.bounds.xmin + dx/2
        bounds7.ymin = self.bounds.ymin + dy/2
        bounds7.ymax = self.bounds.ymin + dy
        bounds7.zmin = self.bounds.zmin + dz/2
        bounds7.zmax = self.bounds.zmin + dz

        bounds8.xmin = self.bounds.xmin + dx/2
        bounds8.xmax = self.bounds.xmin + dx
        bounds8.ymin = self.bounds.ymin + dy/2
        bounds8.ymax = self.bounds.ymin + dy
        bounds8.zmin = self.bounds.zmin + dz/2
        bounds8.zmax = self.bounds.zmin + dz

        child1 = _OctNode(
            bounds1, self, self.depth + 1, self.max_depth)
        child2 = _OctNode(
            bounds2, self, self.depth + 1, self.max_depth)
        child3 = _OctNode(
            bounds3, self, self.depth + 1, self.max_depth)
        child4 = _OctNode(
            bounds4, self, self.depth + 1, self.max_depth)
        child5 = _OctNode(
            bounds5, self, self.depth + 1, self.max_depth)
        child6 = _OctNode(
            bounds6, self, self.depth + 1, self.max_depth)
        child7 = _OctNode(
            bounds7, self, self.depth + 1, self.max_depth)
        child8 = _OctNode(
            bounds8, self, self.depth + 1, self.max_depth)

        self.children = [child1, child2, child3, child4,
                         child5, child6, child7, child8]

    cdef bint contains_box(self, box3d bx):
        '''
        Identifies whether the node completely contains `bx`.
        '''
        # if the lower bounds of `self` are not below `bx`, then
        # return false
        if self.bounds.xmin >= bx.xmin:
            return False
        elif self.bounds.ymin >= bx.ymin:
            return False
        elif self.bounds.zmin >= bx.zmin:
            return False

        # if the upper bounds of `self` are not above `bx`, then
        # return false
        elif self.bounds.xmax <= bx.xmax:
            return False
        elif self.bounds.ymax <= bx.ymax:
            return False
        elif self.bounds.zmax <= bx.zmax:
            return False
        else:
            return True

    cdef void add_box(self, long idx, box3d bx):
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
            _OctNode child

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
        # children, so we add the item to `self`.
        self.boxes.append((idx, bx))
        return

    cdef _OctNode smallest_bounding_node(self, box3d bx):
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
            _OctNode child

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
            _OctNode child
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
        Removes nodes that have no boxes and no children. This should
        be called only when all the boxes have been added.
        '''
        cdef:
            _OctNode child
            list kept_children = []

        for child in self.children:
            child.prune()
            # if, after pruning, the child has children of its own or
            # it has boxes, then keep it
            if child.children:
                kept_children.append(child)
            elif child.boxes:
                kept_children.append(child)

        self.children = kept_children
        return


cdef class OctTree(_OctNode):
    '''
    The top-level OctTree class. This is used to efficiently find
    which boxes, among a large collection of boxes, intersect a query
    box. This class is functionally similar to RTree, except the build
    time is significantly shorter, while the query time is longer.

    Parameters
    ----------
    bounds : (6,) float array
        The bounds for the oct-tree as (xmin, ymin, zmin, xmax, ymax,
        zmax). This defines the area that is recursively bisected. The
        data does not need to fall within these bounds, although the
        algorithm will be less efficient if they do not.

    max_depth : int, optional

    '''
    def __init__(self, double[:] bounds, long max_depth=5):
        cdef:
            box3d bx

        bx.xmin = bounds[0]
        bx.ymin = bounds[1]
        bx.zmin = bounds[2]
        bx.xmax = bounds[3]
        bx.ymax = bounds[4]
        bx.zmax = bounds[5]
        super().__init__(bx, None, 1, max_depth)

    def add_boxes(self, double[:, :] boxes):
        '''
        Adds the boxes to the OctTree

        Parameters
        ----------
        boxes : (n, 6) float array
            Boxes described as (xmin, ymin, zmin, xmax, ymax, zmax)

        '''

        cdef:
            long i
            long nboxes = boxes.shape[0]
            box3d bx

        for i in range(nboxes):
            bx.xmin = boxes[i, 0]
            bx.ymin = boxes[i, 1]
            bx.zmin = boxes[i, 2]
            bx.xmax = boxes[i, 3]
            bx.ymax = boxes[i, 4]
            bx.zmax = boxes[i, 5]
            self.add_box(self.box_count, bx)

    def intersections(self, double[:] box):
        '''
        Finds all the boxes that intersect `box`.

        Parameters
        ----------
        box : (6,) float array
            Query box described as (xmin, ymin, zmin, xmax, ymax,
            zmax)

        Returns
        -------
        (k,) list of ints
            The indices of the boxes that intersect `box`. The indices
            correspond to the order in which the boxes were added to
            the oct-tree.

        '''
        cdef:
            long i
            box3d bx1, bx2
            _OctNode node, member
            list family = []
            list indices = []

        bx1.xmin = box[0]
        bx1.ymin = box[1]
        bx1.zmin = box[2]
        bx1.xmax = box[3]
        bx1.ymax = box[4]
        bx1.zmax = box[5]
        # find the smallest node that `box` fits in, and then test
        # whether that box intersects any boxes in related nodes
        node = self.smallest_bounding_node(bx1)
        family.append(node)
        family.extend(node.ancestor_nodes())
        family.extend(node.descendant_nodes())
        for member in family:
            for i, bx2 in member.boxes:
                if boxes_intersect_3d(bx1, bx2):
                    indices.append(i)

        return indices
