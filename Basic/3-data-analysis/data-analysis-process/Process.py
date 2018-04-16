import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def get_league_data(df, league_id, season):
    columns = ['id', 'league_id', 'season', 'match_api_id', 'home_team_api_id', 'away_team_api_id', 'home_team_goal', 'away_team_goal']
    df_match = df[(df['league_id'] == league_id) & (df['season'] == season)]
    return df_match.loc[:, columns]


# get total goals in a league
def get_league_mean_goals(df):
    matches = len(df.index)
    total_scores = df['home_team_goal'].sum() + df['away_team_goal'].sum()
    return float("{0:.2f}".format(float(total_scores) / matches))


def process_float(data):
    return float("{0:.2f}".format(data))


# get points according to the rule three points for a win
# https://en.wikipedia.org/wiki/Three_points_for_a_win
def get_best_team_result(df):
    team_points = {}
    for row in df.itertuples():
        home_team = row[8]
        away_team = row[9]
        home_team_goal = row[10]
        away_team_goal = row[11]
        result = home_team_goal - away_team_goal

        # this four-element list means numbers of win, tie, lose and total points
        if team_points.get(home_team) is None:
            home_team_result = [0] * 4
            team_points[home_team] = home_team_result
        if team_points.get(away_team) is None:
            away_team_result = [0] * 4
            team_points[away_team] = away_team_result

        # home win
        if result > 0:
            team_points[home_team][0] += 1
            team_points[away_team][2] += 1
        # tie
        elif result == 0:
            team_points[home_team][1] += 1
            team_points[away_team][1] += 1
        # away win
        else:
            team_points[away_team][0] += 1
            team_points[home_team][2] += 1

    for team in team_points.keys():
        team_points[team][3] = 3 * team_points[team][0] + team_points[team][1]

    team_points = sorted(team_points.items(), key=lambda d: d[1][3], reverse=True)
    matches = team_points[0][1][0] + team_points[0][1][1] + team_points[0][1][2]
    # win rate, tie rate, lose rate
    win_rate = process_float(team_points[0][1][0] / matches)
    tie_rate = process_float(team_points[0][1][1] / matches)
    lose_rate = process_float(team_points[0][1][2] / matches)
    return {team_points[0][0]: [win_rate, tie_rate, lose_rate]}


# find the most mean goals and the team. Assume they have the same number of matches
def get_team_mean_goals(df):
    # For the team_goals, the key is team_id, the value is three-element list.
    # In the list, the first element is goals, the second element is fumbles, the third is number of matches
    team_goals = {}
    for row in df.itertuples():
        home_team = row[5]
        away_team = row[6]
        home_team_goal = row[7]
        away_team_goal = row[8]

        if team_goals.get(home_team) is None:
            team_goals[home_team] = [0] * 3
            team_goals[home_team][0] = home_team_goal
            team_goals[home_team][1] = away_team_goal
            team_goals[home_team][2] = 1
        else:
            team_goals[home_team][0] += home_team_goal
            team_goals[home_team][1] += away_team_goal
            team_goals[home_team][2] += 1

        if team_goals.get(away_team) is None:
            team_goals[away_team] = [0] * 3
            team_goals[away_team][0] = away_team_goal
            team_goals[away_team][1] += home_team_goal
            team_goals[away_team][2] = 1
        else:
            team_goals[away_team][0] += away_team_goal
            team_goals[away_team][1] += home_team_goal
            team_goals[away_team][2] += 1
    team_goals = sorted(team_goals.items(), key=lambda d: d[1][0], reverse=True)
    max_goals_team = team_goals[0][0]
    goals = team_goals[0][1][0]
    fumbles = team_goals[0][1][1]
    matches = float(team_goals[0][1][2])
    max_mean_goals = float("{0:.2f}".format(goals / matches))
    mean_fumbles = float("{0:.2f}".format(fumbles / matches))
    return max_goals_team, max_mean_goals, mean_fumbles


def plot_goals_bar(league_list, league_goals_list, team_list, team_goals_list, team_fumbles_list):
    fig = plt.figure(figsize=(12, 6))
    l = len(league_list)
    ind = np.arange(l)
    league_list = [league_list[i] + "\n" + str(team_list[i]) for i in range(l)]
    league = fig.add_subplot(111)
    team = league.twinx()

    league.set_ylim(2, 3.2)
    team.set_ylim(0, 4)
    league.set_xlabel('Country')
    league.set_ylabel('League Mean Goals')
    team.set_ylabel('Top Team Mean Goals and Fumbles')
    #     plt.bar(league_list, league_goals_list)
    league.bar(ind, league_goals_list)
    plt.xticks(ind, league_list)
    p1 = plt.plot(ind, team_goals_list, 'bo', ind, team_goals_list, 'k', color='g')
    p2 = plt.plot(ind, team_fumbles_list, 'bo', ind, team_fumbles_list, 'k', color='r')

    plt.legend((p1[0], p2[0]), ('goals', "fumbles"), loc=9)
    plt.show()


def plot_team_line(team_list, win_list, tie_list, lose_list):
    # fig = plt.figure(figsize=(12, 6))
    ind = np.arange(len(team_list))
    win_list = np.array(win_list)
    tie_list = np.array(tie_list)
    lose_list = np.array(lose_list)
    p1 = plt.bar(ind, lose_list, width=0.5, color='r')
    p2 = plt.bar(ind, tie_list, width=0.5, bottom=lose_list, color='b')
    p3 = plt.bar(ind, win_list, width=0.5, bottom=tie_list+lose_list, color='g')
    plt.ylim([0, 1])
    plt.legend((p1[0], p2[0], p3[0]), ('lose', 'tie', 'win'), loc=1)
    plt.xticks(ind, team_list)
    plt.show()


def plot_team_attributes(mean_team_attribute, fastest_team_attribute):
    ind = np.arange(len(mean_team_attribute))
    plt.figure(figsize=(14, 6))
    p1 = plt.plot(ind, mean_team_attribute, 'bo', ind, mean_team_attribute, 'k', color='g')
    p2 = plt.plot(ind, fastest_team_attribute, 'bo', ind, fastest_team_attribute, 'k', color='b')
    plt.legend((p1[0], p2[0]), ('Mean', 'Fast'))
    plt.xticks(ind, new_labels)
    plt.xlabel('Team Attributes', fontsize=12, weight='bold')
    plt.ylabel('Index', fontsize=12, weight='bold')
    plt.show()


if __name__ == "__main__":
    df_match = pd.read_csv('./data/Match.csv')
    columns = ['id', 'league_id', 'season', 'match_api_id', 'home_team_api_id', 'away_team_api_id', 'home_team_goal',
               'away_team_goal']
    df_league = pd.read_csv('./data/League.csv')
    df_team_attribute = pd.read_csv('./data/Team_Attributes.csv')

    # Answer1:
    season = '2015/2016'
    league_id_list = list(df_league['id'])
    league_country_list = []
    league_mean_goals_list = []
    team_list = []
    team_goals_list = []
    team_fumbles_list = []
    for league_id in league_id_list:
        league = df_league[df_league['id'] == league_id]
        # mean_goals-country
        league_country = str(league['name']).split()[1]
        league_country_list.append(league_country)

        # df_match = df_match[(df_match['season'] == season) & (df_match['league_id'] == league_id)]
        df_league_data = get_league_data(df_match, league_id, season)
        league_mean_goals = get_league_mean_goals(df_league_data)
        league_mean_goals_list.append(league_mean_goals)

        team, goals, fumbles = get_team_mean_goals(df_league_data)
        team_list.append(team)
        team_goals_list.append(goals)
        team_fumbles_list.append(fumbles)
    plot_goals_bar(league_country_list, league_mean_goals_list, team_list, team_goals_list, team_fumbles_list)


    # Answer2:
    leagues_list = {1729: 'England', 7809: 'Germany', 10257: 'Italy', 21518: 'Spain'}
    best_teams = []
    team_list = []
    win_list = []
    tie_list = []
    lose_list = []
    for league_id in leagues_list.keys():
        df_league_match = df_match[(df_match['season'] == season) & (df_match['league_id'] == league_id)]
        best_team = get_best_team_result(df_league_match)
        key = leagues_list[league_id] + "\n" + str(list(best_team.keys())[0])
        team_list.append(key)
        value = list(best_team.values())
        win_list.append(value[0][0])
        tie_list.append(value[0][1])
        lose_list.append(value[0][2])
    plot_team_line(team_list, win_list, tie_list, lose_list)

    # Answer3:
    new_labels = ['buildUpPlay\nSpeed', 'buildUpPlay\nDribbling', 'buildUpPlay\nPassing',
              'chanceCreation\nPassing', 'chanceCreation\nCrossing', 'chanceCreation\nShooting',
              'defence\nPressure', 'defence\nAggression', 'defence\nTeamWidth']
    df_team_attribute = df_team_attribute.iloc[:, np.r_[4, 6, 8, 11, 13, 15, 18, 20, 22]]
    df_team_attribute.columns = new_labels
    mean_team_attributes = df_team_attribute.mean()
    fastest_team_mean_attributes = df_team_attribute[
        df_team_attribute['buildUpPlay\nSpeed'] == df_team_attribute['buildUpPlay\nSpeed'].max()].mean()
    plot_team_attributes(mean_team_attributes, fastest_team_mean_attributes)
