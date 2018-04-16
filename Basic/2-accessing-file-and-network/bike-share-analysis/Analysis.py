import matplotlib.pyplot as plt
import csv
import numpy as np
import networkx as nx


def get_data_by_time(filename):
    """
    This function reads in a file with trip data and return data dividing by user type.
    The return value is a dictionary. For week, the first index is the day of week. The value is a list,
    for which the first is numbers of trips and the second is the duration in this day.
    For hour list, the index is the time, the value is the numbers.
    """
    with open(filename, 'r') as f_in:
        # set up csv reader object
        reader = csv.DictReader(f_in)
        result = {}
        result['n_week'] = [0] * 7
        result['d_week'] = [0] * 7
        result['cus_hour'] = [0] * 24
        result['sub_hour'] = [0] * 24
        for data in reader:
            duration = float(data['duration'])
            if data['day_of_week'] == 'Sunday':
                result['n_week'][0] += 1
                result['d_week'][0] += duration
            elif data['day_of_week'] == 'Monday':
                result['n_week'][1] += 1
                result['d_week'][1] += duration
            elif data['day_of_week'] == 'Tuesday':
                result['n_week'][2] += 1
                result['d_week'][2] += duration
            elif data['day_of_week'] == 'Wednesday':
                result['n_week'][3] += 1
                result['d_week'][3] += duration
            elif data['day_of_week'] == 'Thursday':
                result['n_week'][4] += 1
                result['d_week'][4] += duration
            elif data['day_of_week'] == 'Friday':
                result['n_week'][5] += 1
                result['d_week'][5] += duration
            else:
                result['n_week'][6] += 1
                result['d_week'][6] += duration

            hour = int(data['hour'])
            if data['user_type'] == 'Customer':
                result['cus_hour'][hour] += 1
            else:
                result['sub_hour'][hour] += 1
        return result


def plot_data_by_day(data):
    index = np.arange(7)
    width = 0.3

    fig = plt.figure()
    host = fig.add_subplot(111)
    part = host.twinx()

    host.set_ylim(0, 50000)
    part.set_ylim(0, 700000)
    host.set_xlabel('Day of Week')
    host.set_ylabel('Numbers')
    part.set_ylabel('Durations(min)')

    p1 = host.bar(index + 0.1, data['n_week'], width, color='b')
    p2 = part.bar(index + 0.4, data['d_week'], width, color='g')

    plt.legend((p1[0], p2[0]), ('Numbers', 'Durations'), prop={'size': 8})
    host.yaxis.label.set_color('b')
    part.yaxis.label.set_color('g')
    plt.xticks(index + 0.25, ('Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'))
    plt.show()


def plot_data_by_hour(data):
    index = np.arange(24)
    p1 = plt.plot(index, data['cus_hour'], 'bo', index, data['cus_hour'], 'k', color='green')
    p2 = plt.plot(index, data['sub_hour'], 'bo', index, data['sub_hour'], 'k', color='blue')
    plt.legend((p1[1], p2[1]), ('Customer', 'Subscribe'))
    plt.xlabel('hour')
    plt.xlim([0, 24])
    plt.ylabel('numbers')
    ticks = [0] * 24
    for i in index:
        ticks[i] = str(i)
    plt.xticks(index, ticks)
    plt.show()


def get_station_graph(start_station_id, end_station_list):
    """
    return a set from start station to first ten end station
    """
    start_station_graph = []
    for i in range(10):
        if end_station_list[i] is not None:
            start_station_graph.append((start_station_id, end_station_list[i]))
    return start_station_graph


def get_three_largest_stations_graph(filename):
    """
    find the three largest stations with the most connection with others and find the destinations for them.
    """
    with open(filename) as f_in:
        reader = csv.DictReader(f_in)
        station = {}  # This is a {station-id: station-name} dictionary. It is more efficient by using id.
        start_station_number = {}  # This is a {station-id: number of connections} dictionary.
        start_station_route = {}  # This is a {start-id: {end_id: number of connections}} dictionary.

        largest_station_id = 0
        largest_station_times = 0
        second_largest_station_id = 0
        second_largest_station_times = 0
        third_largest_station_id = 0
        third_largest_station_times = 0
        for row in reader:
            start_id = row['start station id']
            end_id = row['end station id']
            if station.get(start_id) is None:
                station[start_id] = row['start station name']
            if station.get(end_id) is None:
                station[end_id] = row['start station name']
            if start_station_route.get(start_id) is None:
                start_station_route[start_id] = {}
                start_station_route[start_id][end_id] = 1
                start_station_number[start_id] = 1
            else:
                start_station_number[start_id] += 1
                if start_station_route[start_id].get(end_id) is None:
                    start_station_route[start_id][end_id] = 1
                else:
                    start_station_route[start_id][end_id] += 1

            times = start_station_number[start_id]
            if times > third_largest_station_times:
                if times >= second_largest_station_times:
                    if times >= largest_station_times:
                        # If this one is the largest one, only adding the largest by one
                        if start_id != largest_station_id:
                            third_largest_station_id = second_largest_station_id
                            third_largest_station_times = second_largest_station_times
                            second_largest_station_id = largest_station_id
                            second_largest_station_times = largest_station_times
                            largest_station_id = start_id
                        largest_station_times += 1
                    else:
                        # If this one is the second largest one, only adding the second largest by one
                        if start_id != second_largest_station_id:
                            third_largest_station_id = second_largest_station_id
                            third_largest_station_times = second_largest_station_times
                            second_largest_station_id = start_id
                        second_largest_station_times = times
                else:
                    third_largest_station_id = start_id
                    third_largest_station_times = times

    # print the largest three stations information
    largest_station = station[largest_station_id]
    second_largest_station = station[second_largest_station_id]
    third_largest_station = station[third_largest_station_id]
    print("The largest three stations in NYC are {}, {}, and {}."
          .format(largest_station, second_largest_station, third_largest_station))
    print("{} has {} connections with {} stations.".
          format(largest_station, largest_station_times, len(start_station_route[largest_station_id])))
    print("{} has {} connections with {} stations.".
          format(second_largest_station, second_largest_station_times,
                 len(start_station_route[second_largest_station_id])))
    print("{} has {} connections with {} stations.".
          format(third_largest_station, third_largest_station_times,
                 len(start_station_route[third_largest_station_id])))

    # sort the station_route by numbers of connections and get the first ten start-end connections
    largest_station_graph = get_station_graph(largest_station_id,
                                              sort_end_station_list(start_station_route[largest_station_id]))
    second_largest_station_graph = get_station_graph(second_largest_station_id, sort_end_station_list(
        start_station_route[second_largest_station_id]))
    third_largest_station_graph = get_station_graph(third_largest_station_id, sort_end_station_list(
        start_station_route[third_largest_station_id]))

    # convert the station-id back to station-name
    largest_station_graph = get_station_name(largest_station_graph, station)
    second_largest_station_graph = get_station_name(second_largest_station_graph, station)
    third_largest_station_graph = get_station_name(third_largest_station_graph, station)

    return largest_station_graph, second_largest_station_graph, third_largest_station_graph


def sort_end_station_list(start_station_route):
    end_station_list = sorted(start_station_route.items(), key=lambda d: d[1], reverse=True)
    return end_station_list


def get_station_name(station_graph, station):
    station_name_graph = []
    for graph in station_graph:
        id = graph[1][0]
        station_name_graph.append(((station[graph[0]], "{}({})".format(station[id], id)), graph[1][1]))
    return station_name_graph


def plot_graph(station_graph):
    """
    plot the connection between start-stations and end-stations by graph
    The node is the station name, and the number in edge is the riding times between two stations. The largest is in red.
    """
    G = nx.DiGraph()
    edge_labels = {graph[0]: graph[1] for graph in station_graph}
    node_labels = {graph[0]: graph[0][1] for graph in station_graph}
    for graph in station_graph:
        G.add_edge(graph[0][0], graph[0][1])
    red_edges = [station_graph[0][0]]
    blue_edges = [edge for edge in G.edges() if edge not in red_edges]
    pos = nx.spring_layout(G)
    nx.draw_networkx_nodes(G, pos, node_color='green', node_size=200)
    nx.draw_networkx_labels(G, pos, node_labels=node_labels)
    nx.draw_networkx_edges(G, pos, edgelist=red_edges, edge_color='r', arrows=True)
    nx.draw_networkx_edges(G, pos, edgelist=blue_edges, edge_color='b', arrows=True, arrowsize=10)
    nx.draw_networkx_edge_labels(G, pos=pos, edge_labels=edge_labels)
    plt.show()


if __name__ == '__main__':
    time_data = get_data_by_time('./data/NYC-2016-Summary.csv')
    plot_data_by_hour(time_data)

    largest_station_graph, second_largest_station_graph, third_largest_station_graph = \
        get_three_largest_stations_graph('./data/NYC-CitiBike-2016.csv')
    plot_graph(largest_station_graph)
    plot_graph(second_largest_station_graph)
    plot_graph(third_largest_station_graph)
