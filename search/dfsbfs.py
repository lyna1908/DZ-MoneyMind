def dfs(problem):
    start_node = SpendingNode(problem.initial_state)
    stack = [start_node]
    visited = set()

    while stack:
        node = stack.pop()
        state_tuple = tuple(node.state.items())  # Convert dict to hashable tuple

        if state_tuple in visited:
            continue

        visited.add(state_tuple)
        print(node.state)

        if problem.is_goal_state(node.state):
          
            return node  # Return the goal node

        # Expand children and add to stack
        children = problem.expand_node(node)
        stack.extend(children)  # DFS = LIFO
        from collections import deque

def bfs(problem):
    start_node = SpendingNode(problem.initial_state)
    queue = deque([start_node])
    visited = set()

    while queue:
        node = queue.popleft()
        state_tuple = tuple(node.state.items())

        if state_tuple in visited:
            continue

        visited.add(state_tuple)
        print(node.state)

        if problem.is_goal_state(node.state):
            return node

        children = problem.expand_node(node)
        queue.extend(children)  # BFS = FIFO