

my_list = [121.43781, 0.0, 51.47832]

def cap(list_values, max_value, min_value):
    wip_cap = []
    for value in list_values:
        wip_cap.append(int(max(min(value, max_value), min_value)))
    return wip_cap
 
print(cap(my_list, 100, 0))