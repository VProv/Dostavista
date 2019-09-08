import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tqdm import tqdm
import sys
from scipy.spatial import distance_matrix

def filter_orders(orders):
    orders = orders[(orders['pickup_to']>=360) 
                    & (orders['dropoff_to'] >=360) 
                    & (orders['payment'] > 0) 
                    & (orders['dropoff_to'] >= orders['pickup_to'])]
    return orders


def manh_distance(x0,y0,x1,y1):
    return np.abs(x0 - x1) + np.abs(y0-y1)


def courier_distance(x0,y0,x1,y1):
    return 10 + manh_distance(x0,y0,x1,y1)


def greedy_courier(start_position, visited_nodes, money, end2start, end2start_time, drop_coordinates, pick_coordinates, time_from_start_to_end, time_from_ends_to_starts, time_from_ends_to_end, metric_from_ends_to_end, drop_from, drop_to, pick_from, pick_to, PROFIT_THRESHOLD=50):
    """
    Need a lot of global variables
    """
    path = []
    pick_time_first = 10 + distance_matrix([start_position], pick_coordinates, p=1).reshape((-1,))
    drop_time_first = pick_time_first + time_from_start_to_end
    metric_first = money - (pick_time_first + drop_time_first) * 2
    current_time = 360
    
    pick_times = current_time + pick_time_first
    drop_times = pick_times + time_from_start_to_end
    need_to_wait_mask = (pick_from - pick_times) > 0
    wait_pick_array = pick_from - pick_times
    wait_pick_array[np.invert(need_to_wait_mask)] = 0
    metric_first -= wait_pick_array * 2
    new_pick_time = pick_times + wait_pick_array
    new_drop_time = new_pick_time + time_from_start_to_end
    # Drop
    wait_drop_array = drop_from - new_drop_time
    wait_drop_array_mask = wait_drop_array > 0
    wait_drop_array[np.invert(wait_drop_array_mask)] = 0
    metric_first -= wait_drop_array * 2
    new_drop_time = new_drop_time + wait_drop_array
    
    sorted_metric = np.argsort(metric_first)[::-1]
    # first step
    for j, id_ in enumerate(sorted_metric):
        if id_ in visited_nodes:
            continue
        pick_time = new_pick_time[id_]
        drop_time = new_drop_time[id_]
        if  pick_time < pick_to[id_] and drop_time < drop_to[id_]:
            path.append(id_)
            current_time = drop_time
            metric_from_ends_to_end[:, id_] = -10e5
            break
            
        if metric_first[id_] < -100:
            break
        #elif pick_time < pick_from[id_]: # Let's wait
        #    wait_pick = pick_from[id_] - pick_time
        #    pick_time = pick_from[id_]
        #    new_drop_time = pick_time + time_from_start_to_end[id_]
        #    if new_drop_time < drop_to[id_]: # Can drop
        #        if drop_from[id_] <= new_drop_time: # In time!
        #            if (metric_first[id_] - wait_pick * 2) > -50:
        #                path.append(id_)
        #                current_time = new_drop_time
        #                metric_from_ends_to_end[:, id_] = -10e5
        #                break
        #        else: # Should wait drop
        #            wait_drop = drop_from[id_] - new_drop_time
        #            if (metric_first[id_] - wait_pick * 2 - wait_drop * 2) > -50:
        #                path.append(id_)
        #                current_time = new_drop_time + wait_drop
        #                metric_from_ends_to_end[:, id_] = -10e5
        #                break
    STOP = False
    if len(path) == 0:
        STOP = True
    else:
        visited_nodes.add(path[0])
    while not STOP:
        current_metric_array = metric_from_ends_to_end[path[-1], :]
        # [id_] Pick
        pick_times = current_time + time_from_ends_to_starts[path[-1]]
        drop_times = pick_times + time_from_start_to_end
        need_to_wait_mask = (pick_from - pick_times) > 0
        wait_pick_array = pick_from - pick_times
        wait_pick_array[np.invert(need_to_wait_mask)] = 0
        current_metric_array -= wait_pick_array * 2
        new_pick_time = pick_times + wait_pick_array
        new_drop_time = new_pick_time + time_from_start_to_end
        # Drop
        wait_drop_array = drop_from - new_drop_time
        wait_drop_array_mask = wait_drop_array > 0
        wait_drop_array[np.invert(wait_drop_array_mask)] = 0
        current_metric_array -= wait_drop_array * 2
        new_drop_time = new_drop_time + wait_drop_array
        # Sort
        sorted_metric = np.argsort(current_metric_array)[::-1]
        for j, id_ in enumerate(sorted_metric):
            pick_time = new_pick_time[id_]
            drop_time = new_drop_time[id_]
            if  current_metric_array[id_] < 0 or current_time >=1439:
                STOP = True
            if  pick_time < pick_to[id_] and drop_time < drop_to[id_] and id_ not in visited_nodes:
                path.append(id_)
                visited_nodes.add(id_)
                current_time = drop_time
                metric_from_ends_to_end[:, id_] = -10e5
                break  
            #elif pick_time < pick_from[id_]: # Let's wait
            #    wait_pick = pick_from[id_] - pick_time
            #    pick_time = pick_from[id_]
            #    new_drop_time = pick_time + time_from_start_to_end[id_]
            #    if new_drop_time < drop_to[id_]: # Can drop
            #        if drop_from[id_] <= new_drop_time: # In time!
            #            if (current_metric_array[id_] - wait_pick * 2) > PROFIT_THRESHOLD:
            #                path.append(id_)
            #                visited_nodes.add(id_)
            #                current_time = new_drop_time
            #                metric_from_ends_to_end[:, id_] = -10e5
            #                break
            #        else: # Should wait drop
            #            wait_drop = drop_from[id_] - new_drop_time
            #            if (current_metric_array[id_] - wait_pick * 2 - wait_drop * 2) > PROFIT_THRESHOLD:
            #                path.append(id_)
            #                visited_nodes.add(id_)
            #                current_time = new_drop_time + wait_drop
            #                metric_from_ends_to_end[:, id_] = -10e5
            #                break
    return path, current_time, visited_nodes


def main(orders_path, output_path):
    with open(orders_path) as f:
        contest_data = json.load(f)
    couriers = pd.DataFrame(contest_data['couriers'])
    orders = pd.DataFrame(contest_data['orders'])
    depots = pd.DataFrame(contest_data['depots'])
    orders = filter_orders(orders)

    drop_x = orders['dropoff_location_x'].values
    drop_y = orders['dropoff_location_y'].values
    pick_x = orders['pickup_location_x'].values
    pick_y = orders['pickup_location_y'].values

    money = orders['payment'].values
    end2start = np.zeros((orders.shape[0], orders.shape[0]))
    end2start_time = np.zeros((orders.shape[0], orders.shape[0]))
    drop_coordinates = orders[['dropoff_location_x', 'dropoff_location_y']].values
    pick_coordinates = orders[['pickup_location_x', 'pickup_location_y']].values
    time_from_start_to_end = np.array(courier_distance(pick_x, pick_y, drop_x, drop_y))
    # [end_id, start_id]
    time_from_ends_to_starts = 10 + distance_matrix(drop_coordinates, pick_coordinates, p=1).astype('float')
    # Avoid to go to myself
    time_from_ends_to_starts += np.eye(time_from_ends_to_starts.shape[0]) * 10e6
    # end0 -> start1 -> end1

    time_from_ends_to_end = time_from_start_to_end + time_from_ends_to_starts
    #
    metric_from_ends_to_end = np.array(money) - time_from_ends_to_end * 2

    drop_from = orders['dropoff_from'].values
    drop_to = orders['dropoff_to'].values
    pick_from = orders['pickup_from'].values
    pick_to = orders['pickup_to'].values
    
    
    # CENTER
    CENTER = [210,130]
    couriers_distance_from_center = manh_distance(couriers['location_x'],couriers['location_y'], CENTER[0], CENTER[1]).values
    argsort = np.argsort(couriers_distance_from_center)[::-1]
    
    paths, times = [], []
    visited_nodes = set()
    for start_position in tqdm(couriers[['location_x', 'location_y']].values[argsort]):
        path, time, visited_nodes = greedy_courier(start_position, visited_nodes, 
                                                  money, end2start, end2start_time, drop_coordinates, pick_coordinates, time_from_start_to_end, time_from_ends_to_starts, time_from_ends_to_end, metric_from_ends_to_end, drop_from, drop_to, pick_from, pick_to)
        paths.append(path)
        times.append(time)
    
    order_ids = orders['order_id'].values
    pickup_point_ids = orders['pickup_point_id'].values
    dropoff_point_ids = orders['dropoff_point_id'].values
    actions = []
    for path, courier_id in zip(paths, couriers['courier_id'].values[argsort]):
        for id_ in path:
            actions.append([courier_id, "pickup", order_ids[id_], pickup_point_ids[id_]])
            actions.append([courier_id, "dropoff", order_ids[id_], dropoff_point_ids[id_]])
            
    result = pd.DataFrame(actions, columns=['courier_id', 'action', 'order_id', 'point_id'])

    result.to_json(output_path, orient='records')
    
    
if __name__ == '__main__':
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    main(input_file, output_file)
    