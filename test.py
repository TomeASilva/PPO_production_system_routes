
def summarize_performance(path):

    wip = np.loadtxt (f"{path}/WIP.csv", delimiter=";", unpack=False)
    
    wip = np.mean(wip[:,[1, 3, 5]], axis=0)
    wip_total = np.mean(wip)
    
    parts_produced = np.loadtxt(f"{path}/PartsProduced.csv", delimiter=";", unpack=False)
    parts_produced = parts_produced[-1, 1:]
    
    flow_time = np.loadtxt (f"{path}/flow_time.csv", delimiter=";", unpack=False)
    mean_flow_time = np.mean(flow_time[:, -1])
    mean_cycle_time = np.mean(flow_time[:, -2])

    part_0 = flow_time[flow_time[:, 0] == 0 ][:, 2:]
    part_1 = flow_time[flow_time[:, 0] == 1 ][:, 2:]  
    part_2 = flow_time[flow_time[:, 0] == 2 ][:, 2:]  
    
    flow_time_parts = [np.mean(part[: , -1], axis=0) for part in [part_0, part_1, part_2]]
    cycle_time_parts = [np.mean(part[:, -2], axis=0) for part in [part_0, part_1, part_2]]
    
    
    print(f"\033[0;31mAverage WIP: {wip_total}\033[0m")
    for i in range(3):
        print(f"Route_{i}: {wip[i]}")

    print(f"\033[0;31mParts Produced Total: {parts_produced[0]}\033[0m") 
    for i in range(1, 4):
        print(f"Part_{i-1}: {parts_produced[i]}")
    
    print(f"\033[0;31mParts Cycle Time: {mean_cycle_time}\033[0m")
    for i in range (3):
        print(f"Part{i}: {cycle_time_parts[i]}")
        
    print(f"\033[0;31mParts Flow Time: {mean_flow_time}\033[0m")
    for i in range (3):
        print(f"Part{i}: {flow_time_parts[i]}")
         
  
summarize_performance("2020_05_04 12_10_21 WIP set to [3, 3, 3]")