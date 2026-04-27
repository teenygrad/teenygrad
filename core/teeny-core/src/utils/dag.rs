/*
 * Copyright (c) 2026 Teenygrad.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

use alloc::vec::Vec;

/// A node in the DAG. Carries a boxed trait object and a list of children.
pub struct Node<V> {
    pub value: V,
    pub children: Vec<usize>, // indices into Dag.nodes
    pub parents: Vec<usize>,  // indices into Dag.nodes (needed for topological sort, etc)
}

pub struct Dag<V> {
    nodes: Vec<Node<V>>,
}

impl<V> Default for Dag<V> {
    fn default() -> Self {
        Self { nodes: Vec::new() }
    }
}

impl<V> Dag<V> {
    /// Create an empty DAG.
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds a node to the graph, returns index of the node.
    pub fn add_node(&mut self, value: V) -> usize {
        let idx = self.nodes.len();
        self.nodes.push(Node {
            value,
            children: Vec::new(),
            parents: Vec::new(),
        });
        idx
    }

    /// Adds a directed edge from `from` to `to`.
    /// Panics if indices are out of bounds or would cause a self-loop.
    /// Ignores duplicate edges.
    pub fn add_edge(&mut self, from: usize, to: usize) {
        assert!(from != to, "self-loop detected");
        assert!(
            from < self.nodes.len() && to < self.nodes.len(),
            "invalid node index"
        );
        // Check for duplicate edges
        if !self.nodes[from].children.contains(&to) {
            self.nodes[from].children.push(to);
        }
        if !self.nodes[to].parents.contains(&from) {
            self.nodes[to].parents.push(from);
        }
        // NOT checking for cycles for no_std simplicity
    }

    /// Returns a reference to the node at the given index.
    pub fn node(&self, idx: usize) -> &Node<V> {
        &self.nodes[idx]
    }

    /// Returns a mutable reference to the node at the given index.
    pub fn node_mut(&mut self, idx: usize) -> &mut Node<V> {
        &mut self.nodes[idx]
    }

    /// Returns the number of nodes in the graph.
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Returns true if the graph contains no nodes.
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Perform a topological sort. Returns a list of node indices.
    /// Panics if the graph contains a cycle.
    pub fn topological_sort(&self) -> Vec<usize> {
        let mut in_degree = alloc::vec![0; self.nodes.len()];
        for node in &self.nodes {
            for &child in &node.children {
                in_degree[child] += 1;
            }
        }

        let mut stack = Vec::new();
        for (i, &deg) in in_degree.iter().enumerate() {
            if deg == 0 {
                stack.push(i);
            }
        }

        let mut order = Vec::with_capacity(self.nodes.len());
        let mut remaining_in_degree = in_degree.clone();

        while let Some(n) = stack.pop() {
            order.push(n);
            for &child in &self.nodes[n].children {
                remaining_in_degree[child] -= 1;
                if remaining_in_degree[child] == 0 {
                    stack.push(child);
                }
            }
        }

        assert!(
            order.len() == self.nodes.len(),
            "cycle detected in DAG (not acyclic)"
        );
        order
    }
}

impl<V> IntoIterator for Dag<V> {
    type Item = Node<V>;
    type IntoIter = alloc::vec::IntoIter<Node<V>>;

    fn into_iter(self) -> Self::IntoIter {
        self.nodes.into_iter()
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use super::*;

    #[test]
    fn test_empty_dag() {
        let dag: Dag<i32> = Dag { nodes: Vec::new() };
        assert_eq!(dag.len(), 0);
        assert!(dag.is_empty());
        assert_eq!(dag.topological_sort(), Vec::<usize>::new());
    }

    #[test]
    fn test_single_node() {
        let node = Node {
            value: 42,
            parents: Vec::new(),
            children: Vec::new(),
        };
        let dag = Dag { nodes: vec![node] };
        assert_eq!(dag.len(), 1);
        assert!(!dag.is_empty());
        assert_eq!(dag.topological_sort(), vec![0]);
    }

    #[test]
    fn test_chain() {
        // 0 -> 1 -> 2
        let n2 = Node {
            value: 2,
            children: Vec::new(),
            parents: Vec::new(),
        };
        let n1 = Node {
            value: 1,
            children: vec![2],
            parents: Vec::new(),
        };
        let n0 = Node {
            value: 0,
            children: vec![1],
            parents: Vec::new(),
        };
        let dag = Dag {
            nodes: vec![n0, n1, n2],
        };
        let order = dag.topological_sort();
        // The only valid topo order is [0,1,2]
        assert_eq!(order, vec![0, 1, 2]);
    }

    #[test]
    fn test_branch() {
        //   0
        //  / \
        // 1   2
        //  \ /
        //   3
        let n3 = Node {
            value: 3,
            children: Vec::new(),
            parents: Vec::new(),
        };
        let n2 = Node {
            value: 2,
            children: vec![3],
            parents: Vec::new(),
        };
        let n1 = Node {
            value: 1,
            children: vec![3],
            parents: Vec::new(),
        };
        let n0 = Node {
            value: 0,
            children: vec![1, 2],
            parents: Vec::new(),
        };
        let dag = Dag {
            nodes: vec![n0, n1, n2, n3],
        };
        let order = dag.topological_sort();
        // 0 before 1 and 2, 1/2 before 3
        let pos = |n| order.iter().position(|&x| x == n).unwrap();
        assert!(pos(0) < pos(1) && pos(0) < pos(2));
        assert!(pos(1) < pos(3));
        assert!(pos(2) < pos(3));
    }

    #[test]
    #[should_panic(expected = "cycle detected in DAG")]
    fn test_cycle_panics() {
        // 0 -> 1 -> 2 -> 0 (cycle)
        let n0 = Node {
            value: 0,
            children: vec![1],
            parents: Vec::new(),
        };
        let n1 = Node {
            value: 1,
            children: vec![2],
            parents: Vec::new(),
        };
        let n2 = Node {
            value: 2,
            children: vec![0],
            parents: Vec::new(),
        };
        let dag = Dag {
            nodes: vec![n0, n1, n2],
        };
        let _ = dag.topological_sort(); // Should panic
    }
}
