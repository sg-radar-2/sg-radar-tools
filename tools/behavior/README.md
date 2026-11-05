# 행동 분류기 (초안)

트랙 CSV를 읽어서 사람마다 **접근(APPROACH) / 이탈(LEAVE) / 체류(DWELL) / 배회(LOITER)**를 판정하고, 행동이 시작되고 끝날 때 JSON 이벤트를 만듭니다. 규칙 기반이라 판정 기준이 전부 `Config`에 숫자로 나와 있고, 녹화를 보면서 이 숫자를 조정해 나가는 방식입니다.

```
.pcr ──(tools/pcr/to_numpy_gui.pyw)──▶ <이름>_tracks_raw.csv ──(classify_csv.py)──▶ <이름>_events.json
                                                                                 └▶ <이름>_labels.csv
```

## 파일

| 파일 | 하는 일 |
|---|---|
| `behavior_classifier.py` | 분류기 본체. 프레임을 하나씩 넣으면 이벤트를 돌려주는 구조라 실시간에도 그대로 씁니다 |
| `classify_csv.py` | CSV를 프레임 순서대로 분류기에 넣어서 결과를 파일로 저장 (검증용) |
| `test_behavior_classifier.py` | 인공 궤적(직선 접근, 옆으로 지나가기, 서 있기, 원 돌기, ID 바뀜)으로 기본 동작 확인 |

필요한 건 Python 3.9 이상뿐입니다 (외부 패키지 없음).

## 실행

```bat
cd sg-radar-tools

:: 1) .pcr을 CSV로: tools\pcr\to_numpy_gui.pyw 실행 → "트랙 CSV도 저장" 켜고 변환

:: 2) 분류
python tools\behavior\classify_csv.py recordings\20261104_back_forth_walk_run_tracks_raw.csv
python tools\behavior\classify_csv.py recordings\*_tracks_raw.csv --out-dir results

:: 기준값을 바꿔서 돌려보기
python tools\behavior\classify_csv.py --dump-config my_config.json
:: (my_config.json을 고친 뒤)
python tools\behavior\classify_csv.py recordings\a_tracks_raw.csv --config my_config.json

:: 테스트
python tools\behavior\test_behavior_classifier.py
```

실행하면 화면에 이벤트 요약이 나오고, CSV 옆에 두 파일이 생깁니다.

- `<이름>_events.json`: 서버로 보낼 이벤트 목록. 행동마다 `phase: "start"`와 `phase: "end"`가 같은 `event_id`로 한 쌍씩 나옵니다.
- `<이름>_labels.csv`: 입력 행마다 평활화된 위치, 문까지 거리, 그 순간의 판정(`motion`), 진행 중인 행동, 계산에 쓴 값(속도, 방향 비율, 경로 길이 등). 판정이 이상할 때 이 파일을 열어 어느 값 때문인지 확인하면 됩니다.

```json
{"event_id": "Session #0-T0-APPROACH-3", "type": "APPROACH", "phase": "end",
 "session": "Session #0", "track_id": 0, "target_ids": [9], "time_s": 13.7,
 "x": -0.1, "y": 0.75, "distance_m": 0.76, "start_s": 9.2, "end_s": 13.7, "duration_s": 4.5,
 "start_distance_m": 3.33, "end_distance_m": 0.76, "reason": "motion_changed"}
```

## 판정 방법

좌표는 레이더 기준(x 좌우, y 정면)이고, **"거리"는 문까지의 거리**입니다. 지금은 레이더가 문에 붙어 있다고 보고 문 위치를 (0, 0)으로 두었습니다. 설치 위치가 정해지면 `door_x`, `door_y`를 바꾸세요.

1. **입력 거르기**: 포인트가 30개 미만인 관측은 위치를 믿기 어려워서 버립니다. 트래킹을 놓쳤을 때 마지막 위치에 멈춰 있는 값이 여기서 걸러집니다.
2. **트랙 잇기**: 사람이 사라진 지 4초 안에, 마지막 위치에서 1.5m 안에 새 레이더 ID가 나타나면 같은 사람으로 이어 붙입니다.
3. **짧은 판정**: 최근 1초 위치로 속도를 구해서 네 가지 중 하나로 정합니다. 0.5초 이상 유지돼야 확정합니다.
   - `still`: 속력 0.3 m/s 미만
   - `approaching` / `receding`: 문 쪽 속도가 0.3 m/s 이상이고, 움직임의 85% 이상이 문 방향일 때. 도플러가 같은 방향을 가리키면 0.15 m/s로도 인정
   - `moving`: 그 밖의 움직임 (옆으로 지나가기 등)
4. **행동 판정**

| 행동 | 조건 (기본값) |
|---|---|
| APPROACH | `approaching`이 이어지는 동안 거리가 0.4m 이상 줄면 시작, `approaching`이 끝나면 종료 |
| LEAVE | `receding`이 이어지는 동안 거리가 0.4m 이상 늘면 시작 |
| DWELL | 문 앞 1.5m 안에서, 3초 동안 순이동이 0.5m 미만인 상태가 10초 이상 이어짐 |
| LOITER | 최근 30초 중 80% 이상 3.5m 안에 있었고, 절반 이상 움직였고, 경로가 10m 이상인데, 직진도 0.35 이하이거나 접근↔이탈을 2번 이상 반복. 배회 중에는 접근/이탈 이벤트를 따로 내지 않음 |

숫자는 전부 `behavior_classifier.py`의 `Config`에 설명과 함께 있습니다.

## 지금 녹화로 확인한 결과

| 녹화 | 기대 | 결과 |
|---|---|---|
| 앞뒤 걷기/뛰기 | 접근·이탈 반복 | 접근 5, 이탈 5 (거리 변화 예: 3.33 → 0.76m 접근) |
| 좌우 걷기/뛰기 | 이벤트 없음 (지나가기) | 없음 |
| 둥글게 걷기 (54초) | 배회 | 처음 30초는 접근/이탈 반복, 30초 지점부터 배회 |
| sample-dataset 12 (3m 앞에 서서 팔 동작) | 이벤트 없음 | 없음 |

옆으로 지나가기와 앞뒤 걷기를 구분하는 기준(`min_radial_ratio`, `approach_min_dr_m`)은 이 녹화들로 맞춘 값입니다. 녹화가 네 개뿐이라 앞으로 바뀔 수 있습니다.

## 아직 정해야 하거나 부족한 것

- **문 위치와 문 앞 구역 크기**: 레이더 설치 위치가 정해지면 `door_x/door_y`, `zone_radius_m`, `watch_radius_m`부터 맞춰야 합니다.
- **체류·배회 녹화가 없음**: 문 앞에 10초 이상 서 있다 가기, 1분 이상 서성이기를 녹화해서 확인해야 합니다. 녹화할 때 정답 구간(몇 초부터 몇 초까지 무슨 행동)을 같이 적어두면 정확도와 오탐·미탐을 셀 수 있습니다.
- **배회 판정에 30초가 걸림**: 그 전까지는 접근/이탈로 보입니다. 더 빨리 잡고 싶으면 `loiter_window_s`를 줄이되, 오탐이 늘어나는지 같이 봐야 합니다.
- **문 바로 앞(1m 이내)을 옆으로 지나가기**: 거리 변화가 대부분 문 방향이라 접근으로 보일 수 있습니다. 녹화로 확인이 필요합니다.
- **여러 사람**: 한 사람 녹화로만 확인했습니다. 두 사람이 가까이 있으면 트랙 잇기가 엉킬 수 있습니다.

## 실시간으로 쓸 때 (라즈베리파이 / 서버)

CSV 대신 레이더에서 받은 프레임을 바로 넣으면 됩니다. `update()`가 돌려주는 이벤트를 그대로 JSON으로 보내면 됩니다.

```python
from behavior_classifier import BehaviorClassifier, Config, TargetObs
import requests

clf = BehaviorClassifier(Config(), session="door-1")

def on_frame(t, targets):          # 레이더 프레임 콜백 (t: 초)
    obs = [TargetObs(tg.target_id, tg.x, tg.y, n_points=tg.n_points, doppler=tg.doppler_mean)
           for tg in targets]
    for event in clf.update(t, obs):
        requests.post("http://SERVER/events", json=event, timeout=2)
```

---

# 문 앞 상황 판정 뷰어 (결정 트리)

접근·이탈 같은 동작을 넘어서 **문 앞에서 무슨 상황인지**(통과, 귀가, 외출, 방문/배달, 장시간 체류, 배회, 반복 접근, 도어락 조작 시도)를 결정 트리로 판정하고, 이상 상황이면 **카메라 촬영** 이벤트를 내는 과정을 화면으로 보여줍니다.

```bat
pip install numpy matplotlib scikit-learn
:: situation_viewer.pyw 더블클릭  (처음 한 번은 결정 트리 학습에 10초 정도)
```

| 파일 | 하는 일 |
|---|---|
| `scenario_sim.py` | 8가지 상황 + '멀리 서 있기'(해당 없음)의 가짜 궤적을 만들고, 레이더처럼 흔들림·격자·끊김을 넣어 `tracks_raw.csv` 형식으로 저장 (`python scenario_sim.py`) |
| `situation_core.py` | 특징 계산, 결정 트리 두 개 학습·저장(`models/`), 판정, 카메라 촬영 이벤트 |
| `situation_viewer.pyw` | 재생 화면 |

**결정 트리 두 개**

1. 움직임 트리 (프레임마다): 속력, 문 쪽 속도, 문 방향 비율, 도플러 → 다가옴 / 멀어짐 / 지나감 / 정지
2. 상황 트리 (0.5초마다, 지금까지의 흐름): 처음 잡힌 거리, 가장 가까웠던 거리, 연속 정지 시간, 문 바로 앞 정지 시간, 정지 중 손 움직임(doppler_std), 접근·이탈 횟수, 경로 길이, 직진도, 옆으로 이동한 비율, 앉음 비율, 사라졌는지 → 8가지 상황 또는 '판단 중'

상황 트리가 이상 상황(장시간 체류, 배회, 반복 접근, 도어락 조작 시도)을 60% 이상 확률로 1초 넘게 유지하면 촬영합니다.

**화면**

- 재생: 위에서 본 궤적과 문 앞 구역, 오른쪽에 6가지 관찰값 · 상황 판정 · **판정 근거(트리가 지나온 질문)**, 아래에 판정/정답 타임라인과 촬영 시점(▼)
- 시나리오 결과: 세션마다 정답과 최종 판정, 촬영 여부를 표로 비교
- 결정 트리: 두 트리 전체(글자 형태)와 검증 정확도, 혼동 행렬

**알아둘 점**

- 지금 결정 트리는 **가짜 데이터로 학습**했습니다. 흔들림 크기는 실제 서 있는 녹화(sample-dataset)의 속도 분포에 맞췄지만, 실제 성능은 정답을 붙인 녹화로 다시 학습·검증해야 합니다. '시나리오 결과'의 정확도도 같은 생성기로 만든 데이터라 실제보다 높게 나옵니다.
- 실제 녹화 4개에 돌려본 결과: 앞뒤 걷기 → 반복 접근(촬영), 둥글게 걷기 → 배회(촬영), 3m 앞에 서 있기 → 판단 중, 좌우 걷기 → 대부분 통과·판단 중이지만 마지막 사람(문 앞 1m를 19초 동안 세 번 왕복)은 반복 접근 → 배회로 촬영됨. 이런 경우를 이상으로 볼지는 팀에서 정해야 합니다.
- 트랙은 레이더 ID 그대로 씁니다 (behavior_classifier.py의 ID 이어 붙이기는 아직 안 들어감).
- 녹화에 정답을 붙이면 `scenario_sim.training_episodes()` 대신 그 녹화로 `situation_core.train()`을 돌리면 됩니다.
