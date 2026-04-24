from collections import deque

# 1. SET: Unique Numbers
# Sets automatically remove duplicates.
numbers = {1, 2, 3}
numbers.add(2)  # 2 is already there, so it remains {1, 2, 3}
print(f"Unique Set: {numbers}")

# 2. STACK: LIFO (Last-In, First-Out)
# Lists are used as stacks with .append() and .pop().
stack = []
stack.append(10)  # Push 10
stack.append(20)  # Push 20
last_in = stack.pop()  # Removes 20 (the last one added)
print(f"Popped from Stack: {last_in}")

# 3. QUEUE: FIFO (First-In, First-Out)
# Deque is used for efficient queues with .append() and .popleft().
queue = deque([100, 200])
queue.append(300)      # Enqueue 300 at the end
first_in = queue.popleft()  # Removes 100 (the first one added)
print(f"Dequeued from Queue: {first_in}")
