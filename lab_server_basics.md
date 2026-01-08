# lab_server

**1단계: 내 방(폴더) 만들기 (저장소 정리)**

서버에 접속하면 텅 빈 화면일 겁니다. 가장 먼저 데이터를 어디에 둘지
정해야 합니다. 보내주신 새로운 규칙(SSD 활용)에 맞춰 정리해 드릴게요.

- **/storage (창고):** 용량이 크고 안전합니다. 원본 데이터나 중요한
결과물은 여기에 둡니다.
- **/data (작업대):** Node1, Node3에 달려있는 빠른 SSD입니다. 실험할
때는 **창고에서 작업대로 데이터를 복사해와서** 쓰는 게 가장 빠릅니다.

**[할 일]** 터미널에 아래 명령어를 쳐서 본인 폴더를 만드세요.

Bash

# 1. 창고에 내 폴더 만들기 (한 번만 하면 됨)

mkdir -p /storage/connectome/gunwo610

# 2. (나중에 실험할 때) 작업대에도 내 폴더 만들기

# 주의: 이건 Node1이나 Node3에 접속했을 때만 보입니다.

mkdir -p /data/gunwo610

⇒ 이미 만들어져 있음.

**2단계: 도구 세팅 (가상환경 만들기)**

서버에는 파이썬만 덩그러니 있습니다. pytorch 같은 도구 상자를 만들어야
합니다.

**[할 일]**

Bash

# 1. 아나콘다(도구함 관리자) 깨우기

source /usr/anaconda3/etc/profile.d/conda.sh

# 2. 내 가상환경 만들기 (이름: my_lab) - 처음에 한 번만!

conda create -n my_lab python=3.8

# 3. 가상환경 접속 (활성화)

conda activate my_lab

# 4. 필요한 거 설치 (예시)

conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c
pytorch -c nvidia

**3단계: 리허설 해보기 (Interactive Mode) ⭐중요**

코드를 짰는데 서버에서 잘 돌아가는지 모르잖아요? **주문서(Batch)를 넣기
전에 직접 들어가서 테스트**해보는 단계입니다.

**[할 일]**

1. **가상 컴퓨터 빌리기:** 터미널에 아래 명령어를 칩니다. (Node1의 GPU
1개를 1시간만 빌리겠다는 뜻)

> Bash
> 
> 
> srun --nodes=1 --nodelist=node1 --gpus=1 --time=01:00:00 --pty
> bash -i
> 
1. 명령어를 치면 프롬프트가 gunwo610@node1으로 바뀝니다. (Node1에
들어온 것!)
2. 이제 python test.py 처럼 코드를 돌려보세요. 에러가 안 나나요?
3. 잘 되면 exit를 쳐서 나옵니다.

**4단계: 진짜 주문서 작성 (train_job.sh 만들기)**

이제 질문하신 **train_job**이 나옵니다. 이건 **"나 퇴근할 건데,
밤새 이대로 실행해줘"**라고 적은 **메모장 파일**입니다.

VS Code에서 새 파일을 만들고, 이름을 run_experiment.sh (이게 train_job
파일입니다)라고 저장하세요. 그리고 아래 내용을 복사해 붙여넣으세요.

Bash

#!/bin/bash

#SBATCH -J my_experiment # 1. 작업 이름 (내 맘대로)

#SBATCH -t 12:00:00 # 2. 최대 실행 시간 (12시간)

#SBATCH -N 1 # 노드 개수

#SBATCH --nodelist=node1 # 3. 사용할 노드 (SSD 있는 node1 선택)

#SBATCH --gpus-per-node=1 # GPU 개수

#SBATCH -o %x_%j.out # 실행 기록 저장할 파일명

#SBATCH -e %x_%j.err # 에러 기록 저장할 파일명

# --- 여기부터가 진짜 실행할 명령어들입니다 ---

# 1. 환경 설정

source /usr/anaconda3/etc/profile.d/conda.sh

conda activate my_lab

# 2. 데이터 복사 (창고 -> 작업대) : 속도를 위해!

# /storage에 있는 내 데이터를 빠른 /data(SSD)로 복사함

cp -r /storage/connectome/gunwo610/my_data /data/gunwo610/

# 3. 파이썬 코드 실행

# (데이터 경로는 이제 /data/gunwo610/my_data 를 쓰면 됨)

srun python main.py

# 4. (선택) 결과물 백업 (작업대 -> 창고)

# SSD인 /data는 임시 공간이라 날아갈 수 있으니 중요한 결과는 /storage로
옮겨둠

cp -r /data/gunwo610/results /storage/connectome/gunwo610/

**5단계: 주문서 제출 (Batch 실행)**

이제 작성한 주문서를 서버 관리인(Slurm)에게 제출하고 퇴근하면 됩니다.

**[할 일]** 터미널에 딱 한 줄만 치세요.

Bash

sbatch run_experiment.sh

- **성공하면:** Submitted batch job 12345 같은 숫자가 뜹니다.
- **확인:** squeue -u gunwo610 을 쳐보면 내 작업이 돌고 있는지(R) 대기
중인지(PD) 나옵니다.

**돌리면서 해보기(01.07)**

cd, ls

df -h ->전체 저장공간 보는 거

1. squeue (전체 대기열 확인): 서버 전체에서 누가 작업을 돌리고 있고,
누가 기다리고 있는지 보여줍니다.

◦ squeue: 현재 실행 중(R)이거나 대기 중(PD)인 모든 작업 목록 출력

2. nvidia-smi (현재 노드 GPU 확인): 지금 접속한 노드(컴퓨터)의 GPU
상태를 보여줍니다.

◦ nvidia-smi: GPU 온도, 메모리 사용량, 현재 돌아가는 프로세스 확인

Interactive Mode가 GPU를 뺏어오나요?

아니요, 뺏어오지 않습니다. 안심하셔도 됩니다.

- Slurm의 역할: Slurm은 '교통정리'를 해주는 스케줄러입니다. srun이나
salloc으로 Interactive Mode를 요청하면, Slurm이 빈 GPU가 있는지
확인합니다.
- 빈 자리가 있으면: 즉시 할당해 줍니다.
- 빈 자리가 없으면: 대기 상태(Pending, PD)로 들어가서 앞사람이 끝날
때까지 줄을 서서 기다립니다.

따라서 다른 사람의 작업을 강제로 중단시키지 않으니 걱정 말고 명령어를
입력하셔도 됩니다

입력하신 명령어 srun --nodes=1 --gres=gpu:1 --pty bash -i의 각 부분
의미는 다음과 같습니다.

1. srun: Slurm 스케줄러에게 작업을 실시간으로 실행해달라고 요청하는
명령어입니다.

2. --nodes=1: 컴퓨터(노드)를 1대 빌려달라는 뜻입니다.

3. --gres=gpu:1: 해당 노드에 있는 GPU를 1개 사용할 수 있게 해달라는
요청입니다 (Generic Resource).

4. --pty: 터미널(Pseudo-Terminal) 모드로 실행하여, 마치 내 컴퓨터
터미널처럼 입출력을 주고받을 수 있게 합니다.

5. bash -i: 실행할 명령어로, interactive(대화형) 모드의 bash 쉘을
켜라는 뜻입니다.

다음으로 해야할 것은

1. Conda 환경 구축하기.
2. Interactive 말고 job 예약해두고 가기.

conda create -n {환경 이름} python=3.9 -y

conda activate quick_test

squeue -u 본인ID : 할당된 qpu가 있는지 확인