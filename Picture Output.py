import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
import re
import matplotlib.pyplot as plt

# 1. 数据加载 - 加载两个赛季的数据并合并
data_2223 = pd.read_csv('22-23 Premier League Data Final.csv')
data_2324 = pd.read_csv('23-24 Premier League Data Final.csv')

# 给每个赛季的数据添加一个标识列
data_2223['Season'] = '22-23'
data_2324['Season'] = '23-24'

# 合并两个赛季的数据集
data = pd.concat([data_2223, data_2324], ignore_index=True,sort=False)

# 2. 特征准备和预处理
def convert_results_to_numeric(results):
    return results.apply(lambda x: 1 if x == 'W' else (0 if x == 'D' else -1))

data['HomePrevious'] = convert_results_to_numeric(data['HomePrevious'])
data['AwayPrevious'] = convert_results_to_numeric(data['AwayPrevious'])

data['Month'] = data['MatchDate'].apply(lambda date: int(re.search(r'(?<=\d{4}-)(\d{2})(?=-\d{2})', date).group(1)))

def calculate_match_result(row):
    if row['HomeGoals'] > row['AwayGoals']:
        return 1
    elif row['HomeGoals'] < row['AwayGoals']:
        return 2
    else:
        return 0

data['MatchResult'] = data.apply(calculate_match_result, axis=1)

# 模型训练与预测函数
def train_and_predict(data, months, season):
    global teams_points

    train_data = data[(data['Month'].isin(months)) & (data['Season'] == season)]
    test_data = data[(data['Month'] > months[-1]) & (data['Season'] == season)]

    if train_data.empty or test_data.empty:
        print(f"No data available for months: {months} in season: {season}")
        return None, None

    X_train = train_data[['HomeTeamRating', 'AwayTeamRating', 'HomePrevious', 'AwayPrevious']]
    y_train = train_data['MatchResult']
    X_test = test_data[['HomeTeamRating', 'AwayTeamRating', 'HomePrevious', 'AwayPrevious']]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    gnb_model = GaussianNB()
    gnb_model.fit(X_train_scaled, y_train)
    y_pred_gnb = gnb_model.predict(X_test_scaled)

    return y_pred_gnb, test_data

# 积分榜初始化
teams_points = {}
rankings_history = {}

# 3. 计算真实积分榜
def calculate_final_standings(data):
    points = {}
    for index, row in data.iterrows():
        home_team = row['HomeTeam']
        away_team = row['AwayTeam']
        home_goals = row['HomeGoals']
        away_goals = row['AwayGoals']

        if home_team not in points:
            points[home_team] = 0
        if away_team not in points:
            points[away_team] = 0

        if home_goals > away_goals:
            points[home_team] += 3
        elif home_goals < away_goals:
            points[away_team] += 3
        else:
            points[home_team] += 1
            points[away_team] += 1

    return points

# 真实赛季末积分榜的排名
final_standings = calculate_final_standings(data)
sorted_final_standings = sorted(final_standings.items(), key=lambda x: x[1], reverse=True)
real_ranking = {team: rank for rank, (team, points) in enumerate(sorted_final_standings, start=1)}

# 主程序循环，预测所有比赛并输出最终积分榜
for season in ['22-23', '23-24']:
    rankings_history[season] = {team: [] for team in data['HomeTeam'].unique()}
    months = sorted(data[data['Season'] == season]['Month'].unique())

    for i in range(len(months)):
        current_months = months[:i + 1]
        teams_points.clear()

        real_points = calculate_final_standings(data[(data['Month'].isin(current_months)) & (data['Season'] == season)])

        predictions, test_data = train_and_predict(data, current_months, season)

        if predictions is not None and test_data is not None:
            for team, points in real_points.items():
                teams_points[team] = points

            for i in range(len(predictions)):
                team1, team2 = test_data.iloc[i]['HomeTeam'], test_data.iloc[i]['AwayTeam']
                result = predictions[i]

                if team1 not in teams_points:
                    teams_points[team1] = 0
                if team2 not in teams_points:
                    teams_points[team2] = 0

                if result == 1:
                    teams_points[team1] += 3
                elif result == 2:
                    teams_points[team2] += 3
                elif result == 0:
                    teams_points[team1] += 1
                    teams_points[team2] += 1

            predicted_sorted_teams = sorted(teams_points.items(), key=lambda x: x[1], reverse=True)
            predicted_ranking = {team: rank for rank, (team, points) in enumerate(predicted_sorted_teams, start=1)}

            # 只考虑前5名和后5名的准确率
            top_5_real = set(team for team, rank in real_ranking.items() if rank <= 5)
            bottom_5_real = set(team for team, rank in real_ranking.items() if rank > len(real_ranking) - 5)
            top_bottom_5_real = top_5_real.union(bottom_5_real)

            correct_top_bottom_predictions = sum(1 for team in top_bottom_5_real if predicted_ranking.get(team) in top_bottom_5_real)
            ranking_accuracy_score = (correct_top_bottom_predictions / 10) * 100


            for rank, (team, points) in enumerate(predicted_sorted_teams, start=1):
                rankings_history[season][team].append(rank)


# 可视化排名历史
for season in ['22-23', '23-24']:
    plt.figure(figsize=(12, 8))
    for team, ranks in rankings_history[season].items():
        plt.plot(range(1, len(ranks) + 1), ranks, marker='o', label=team)

    plt.gca().invert_yaxis()
    plt.xticks(range(1, len(months) + 1), [f"M{i}" for i in range(1, len(months) + 1)])
    plt.xlabel('Month')
    plt.ylabel('Ranking')
    plt.title(f'Premier League Team Rankings Over Time ({season} Season)')
    plt.legend(loc='upper left')
    plt.grid(True)
    plt.show()
