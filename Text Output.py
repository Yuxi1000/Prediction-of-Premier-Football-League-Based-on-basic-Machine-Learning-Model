#Text Output.py
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
data = pd.concat([data_2223, data_2324], ignore_index=True, sort=False)

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

        if home_goals > away_goals:  # 主队胜
            points[home_team] += 3
        elif home_goals < away_goals:  # 客队胜
            points[away_team] += 3
        else:  # 平局
            points[home_team] += 1
            points[away_team] += 1

    return points

# 主程序循环，输出最终积分榜和准确率
for season in ['22-23', '23-24']:
    rankings_history[season] = {team: [] for team in data['HomeTeam'].unique()}
    months = sorted(data[data['Season'] == season]['Month'].unique())

    for i in range(len(months)):
        current_months = months[:i + 1]
        teams_points.clear()

        # 首先计算真实积分榜
        real_points = calculate_final_standings(data[(data['Month'].isin(current_months)) & (data['Season'] == season)])

        predictions, test_data = train_and_predict(data, current_months, season)

        if predictions is not None and test_data is not None:
            # 更新积分榜，首先添加真实比赛的积分
            for team, points in real_points.items():
                teams_points[team] = points

            # 然后添加预测结果的积分
            for i in range(len(predictions)):
                team1, team2 = test_data.iloc[i]['HomeTeam'], test_data.iloc[i]['AwayTeam']
                result = predictions[i]

                if team1 not in teams_points:
                    teams_points[team1] = 0
                if team2 not in teams_points:
                    teams_points[team2] = 0

                if result == 1:  # 主队胜
                    teams_points[team1] += 3
                elif result == 2:  # 客队胜
                    teams_points[team2] += 3
                elif result == 0:  # 平局
                    teams_points[team1] += 1
                    teams_points[team2] += 1

            # 输出当前的最终积分榜（基于所有可用数据）
            sorted_teams = sorted(teams_points.items(), key=lambda x: x[1], reverse=True)

            print(f"\nBased on the results of the previous {len(current_months)} months - Final Standings:")
            print("Rank\tTeam\tPoints")

            for rank, (team, points) in enumerate(sorted_teams, start=1):
                print(f"{rank}\t{team}\t{points}")

            # 输出准确率（只考虑前5名和后5名）
            real_sorted_standings = sorted(real_points.items(), key=lambda x: x[1], reverse=True)
            real_top_5 = set(team for team, _ in real_sorted_standings[:5])
            real_bottom_5 = set(team for team, _ in real_sorted_standings[-5:])

            predicted_sorted_standings = sorted(teams_points.items(), key=lambda x: x[1], reverse=True)
            predicted_top_5 = set(team for team, _ in predicted_sorted_standings[:5])
            predicted_bottom_5 = set(team for team, _ in predicted_sorted_standings[-5:])

            correct_top_bottom_predictions_count = len(real_top_5.intersection(predicted_top_5)) + len(real_bottom_5.intersection(predicted_bottom_5))
            total_comparisons_count = len(real_top_5) + len(real_bottom_5)

            accuracy_score = (correct_top_bottom_predictions_count / total_comparisons_count) * 100 if total_comparisons_count > 0 else None

            print(f"Top and Bottom Ranking Prediction Accuracy for Season {season}: {accuracy_score:.2f}%" if accuracy_score is not None else "No valid results available for accuracy calculation.")

# 输出两个赛季的真实积分榜
for season in ['22-23', '23-24']:
    final_standings_season = calculate_final_standings(data[data['Season'] == season])
    sorted_final_standings_season = sorted(final_standings_season.items(), key=lambda x: x[1], reverse=True)

    print(f'\nFinal LeaderBoard for Season {season}:')
    print('Rank\tTeam\tPoints')

    for rank, (team, score) in enumerate(sorted_final_standings_season, start=1):
        print(f'{rank}\t{team}\t{score}')