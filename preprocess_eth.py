import pandas as pd
import numpy as np
import os


def detect_separator(file_path):
    with open(file_path, 'r') as file:
        first_line = file.readline()
        for sep in ['\t', ',', ' ', ';']:
            if sep in first_line:
                return sep
    return None

# 폴더 내 모든 파일명 받아오기
# folder_path = './val'
folder_path = './test'

file_names = sorted(os.listdir(folder_path)) # 파일명을 알파벳 순서로 정렬
dir_len = len(file_names)
print(f"file_names: {file_names}")

idx=1
# 파일별로 데이터 처리
for file_name in file_names:
    
    file_path = os.path.join(folder_path, file_name)
    
    # 구분자 감지
    separator = detect_separator(file_path)
    print(f"separator:{separator}")
    
    if not separator:
        print(f"Could not detect separator for {file_name}. Skipping this file.")
        continue

    # 데이터셋 읽기
    df = pd.read_csv(file_path, sep=separator, header=None)
    df.columns = ['frame_id', 'agent_id', 'x', 'y']

    # 시간 간격 (frame interval)
    dt = 10  # 프레임 간격이 10임

    # 에이전트 별로 속도와 가속도 계산
    agents = df['agent_id'].unique()
    results = []

    for agent in agents:
        agent_data = df[df['agent_id'] == agent].sort_values(by='frame_id')
        agent_data['x_diff'] = agent_data['x'].diff()
        agent_data['y_diff'] = agent_data['y'].diff()
        agent_data['distance'] = np.sqrt(agent_data['x_diff']**2 + agent_data['y_diff']**2)
        agent_data['velocity'] = (agent_data['distance'] / dt).round(4)
        agent_data['velocity_diff'] = agent_data['velocity'].diff()
        agent_data['acceleration'] = (agent_data['velocity_diff'] / dt).round(4)
        results.append(agent_data)

    # 결과 병합
    result_df = pd.concat(results)

    # NaN 값을 0으로 대체
    result_df.fillna(0, inplace=True)

    # 불필요한 컬럼 제거
    result_df = result_df.drop(columns=['x_diff', 'y_diff', 'distance', 'velocity_diff'])

    # frame_number 기준으로 정렬
    result_df = result_df.sort_values(by=['frame_id', 'agent_id'])

    result_path = folder_path + '2'

    # 결과 저장
    result_file_path = os.path.join(result_path, f'{file_name}2')
    result_df.to_csv(result_file_path, sep=',', index=False, header=False)

    print(f"{idx}/{dir_len}번째 파일의 결과가 저장되었습니다:", result_file_path)
    idx+=1