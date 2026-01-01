class Stack:
    def __init__(self):
        """Инициализация пустого стека."""
        self._items = []

    def push(self, item):
        """Добавляет элемент на вершину стека."""
        self._items.append(item)

    def pop(self):
        """Удаляет и возвращает элемент с вершины стека.
        
        Вызывает IndexError, если стек пуст.
        """
        if self.is_empty():
            raise IndexError("pop from empty stack")
        return self._items.pop()

    def peek(self):
        """Возвращает элемент с вершины стека без его удаления.
        
        Вызывает IndexError, если стек пуст.
        """
        if self.is_empty():
            raise IndexError("peek from empty stack")
        return self._items[-1]

    def is_empty(self):
        """Возвращает True, если стек пуст, иначе False."""
        return len(self._items) == 0

    def size(self):
        """Возвращает количество элементов в стеке."""
        return len(self._items)

    def __repr__(self):
        """Возвращает строковое представление стека."""
        return f"Stack({self._items})"


s = input().strip()
stack = Stack()
matching = {')': '(', ']': '[', '}': '{'}

for char in s:
    if char in '([{':
        stack.push(char)
    elif char in ')]}':
        if stack.is_empty():
            print("No")
            exit()
        if stack.pop() != matching[char]:
            print("No")
            exit()

print("Yes" if stack.is_empty() else "No")