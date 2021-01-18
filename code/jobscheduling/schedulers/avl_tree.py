class Node:
    def __init__(self, key, data, note=None):
        """
        AVL Tree Node class
        :param key: type obj
            Key of the node
        :param data: type obj
            Data stored in the node
        :param note: type obj
            Note to use when modifying other nodes
        """
        self.key = key
        self.data = data
        self.note = note
        self.left = None
        self.right = None


class AVLTree:
    def __init__(self):
        """
        Modified AVL Tree class that realizes the desired AVL tree functionality from CEDF
        """
        self.node = None
        self.height = -1
        self.balance = 0

    def height(self):
        """
        Obtains the height of a tree
        :return: type int
            Tree height
        """
        if self.node:
            return self.node.height
        else:
            return 0

    def is_leaf(self):
        """
        Checks if the AVLTree is only one node
        :return: type bool
            True/False
        """
        return (self.height == 0)

    def is_empty(self):
        """
        Checks if the AVLTree is empty
        :return: type bool
            True/False
        """
        return self.node is None

    def insert(self, key, data, note=0):
        """
        Inserts a node into the AVL Tree
        :param key: type obj
            Key to use for the node
        :param data: type obj
            Data stored within the node
        :param note: type obj
            Note to use when modifying other AVL Tree nodes
        :return: None
        """
        tree = self.node

        newnode = Node(key, data, note)

        if tree is None:
            self.node = newnode
            self.node.left = AVLTree()
            self.node.right = AVLTree()

        elif key < tree.key:
            self.node.left.insert(key, data, note)

        elif key > tree.key:
            self.node.note = max(self.node.note, note)
            self.node.right.insert(key, data, note)

        self.rebalance()

    def rebalance(self):
        '''
        Rebalance a particular (sub)tree
        '''
        # key inserted. Let's check if we're balanced
        self.update_heights(False)
        self.update_balances(False)
        while self.balance < -1 or self.balance > 1:
            if self.balance > 1:
                if self.node.left.balance < 0:
                    self.node.left.lrotate()  # we're in case II
                    self.update_heights()
                    self.update_balances()
                self.rrotate()
                self.update_heights()
                self.update_balances()

            if self.balance < -1:
                if self.node.right.balance > 0:
                    self.node.right.rrotate()  # we're in case III
                    self.update_heights()
                    self.update_balances()
                self.lrotate()
                self.update_heights()
                self.update_balances()

    def rrotate(self):
        '''
        Does a right rotation of the subtree
        '''
        # Rotate left pivoting on self
        A = self.node
        B = self.node.left.node
        T = B.right.node

        self.node = B
        B.right.node = A
        A.left.node = T

    def lrotate(self):
        '''
        Does a left rotation of the subtree
        '''
        # Rotate left pivoting on self
        A = self.node
        B = self.node.right.node
        T = B.left.node

        self.node = B
        B.left.node = A
        A.right.node = T

    def update_heights(self, recurse=True):
        """
        Updates the heights of the subtrees in the AVL tree
        :param recurse:
        :return:
        """
        if self.node is not None:
            if recurse:
                if self.node.left is not None:
                    self.node.left.update_heights()
                if self.node.right is not None:
                    self.node.right.update_heights()

            self.height = max(self.node.left.height,
                              self.node.right.height) + 1
        else:
            self.height = -1

    def update_balances(self, recurse=True):
        """
        Updates internal balances of left and right subtrees
        :param recurse: bool
            Whether recursion into subtrees should be performed
        """
        if self.node is not None:
            if recurse:
                if self.node.left is not None:
                    self.node.left.update_balances()
                if self.node.right is not None:
                    self.node.right.update_balances()

            self.balance = self.node.left.height - self.node.right.height
        else:
            self.balance = 0

    def delete(self, key):
        """
        Deletes a node in the AVL Tree with specified key
        :param key: type obj
            Key to search for
        :return None
        """
        if self.node is not None:
            if self.node.key == key:
                if self.node.left.node is None and self.node.right.node is None:
                    self.node = None  # leaves can be killed at will
                # if only one subtree, take that
                elif self.node.left.node is None:
                    self.node = self.node.right.node
                elif self.node.right.node is None:
                    self.node = self.node.left.node

                # worst-case: both children present. Find logical successor
                else:
                    replacement = self.logical_successor(self.node)
                    if replacement is not None:  # sanity check
                        self.node.key = replacement.key
                        self.node.data = replacement.data
                        self.node.note = replacement.note

                        # replaced. Now delete the key from right child
                        self.node.right.delete(replacement.key)

                self.rebalance()
                return
            elif key < self.node.key:
                self.node.left.delete(key)
            elif key > self.node.key:
                self.node.right.delete(key)

            self.rebalance()
        else:
            return

    def logical_predecessor(self, node):
        '''
        Find the biggest valued node in LEFT child
        '''
        node = node.left.node
        if node is not None:
            while node.right is not None:
                if node.right.node is None:
                    return node
                else:
                    node = node.right.node
        return node

    def logical_successor(self, node):
        '''
        Find the smallest valued node in RIGHT child
        '''
        node = node.right.node
        if node is not None:  # just a sanity check

            while node.left is not None:
                if node.left.node is None:
                    return node
                else:
                    node = node.left.node
        return node

    def check_balanced(self):
        """
        Checks if the tree is balanced
        :return: bool
            True/False
        """
        if self is None or self.node is None:
            return True

        # We always need to make sure we are balanced
        self.update_heights()
        self.update_balances()
        return ((abs(self.balance) < 2) and self.node.left.check_balanced() and self.node.right.check_balanced())

    def inorder_traverse(self):
        """
        Traverses the AVL tree in order
        :return:
        """
        if self.node is None:
            return []

        inlist = []
        l = self.node.left.inorder_traverse()
        for i in l:
            inlist.append(i)

        inlist.append((self.node.key, self.node.data))

        l = self.node.right.inorder_traverse()
        for i in l:
            inlist.append(i)

        return inlist

    def minimum(self):
        """
        Obtains the minimum valued node in the AVL Tree
        :return: type Node
            The Node that is minimum
        """
        if self.node is None:
            return None

        # Apply notes
        if self.node.note:
            # Only update subtree notes if sj_max > s_max_note
            if self.node.data[1] > self.node.note:
                if self.node.left.node is not None:
                    # If no previous note, set it
                    if self.node.left.node.note is not None:
                        self.node.left.note = self.node.note
                    # Otherwise merge and set to the minimum note
                    else:
                        self.node.left.note = min(self.node.left.node.note, self.node.note)

                if self.node.right.node is not None:
                    # If no previous note, set it
                    if self.node.right.node.note is not None:
                        self.node.right.note = self.node.note
                    # Otherwise merge and set to the minimum note
                    else:
                        self.node.right.note = min(self.node.right.node.note, self.node.note)

            # If the note is present we update sj_max for this node
            self.node.data[1] = min(self.node.data[1], self.node.note)

        if self.node and self.node.left.node is None:
            return self.node

        else:
            return self.node.left.minimum()
