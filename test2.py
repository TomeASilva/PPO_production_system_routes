from collections import deque

class Part:
    def __init__(self, _type):
        self.type = _type


my_deque = deque()
part_list = []
for i in range(4):
    my_part = Part(i + 1)
    part_list.append(my_part)
    my_deque.append(my_part)


first_elment = my_deque[0]
my_deque.remove(part_list[0])

if first_elment == my_deque[0]:
    print("The system failed to respect FIfO")
else:
    print("System respected FIFO")

for part in my_deque:
    print(part.type)
