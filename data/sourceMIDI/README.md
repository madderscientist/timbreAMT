# 收集到的MIDI文件
gitignore中忽略了所有文件。包括本来是包括如下文件夹的：
```
├─AkPnBcht
├─AkPnBsdf
├─AkPnCGdD2
├─flutetunes
├─giantmidi-piano
├─ground_truth
├─maestro
├─midiData
└─mine
```

以下数据来自MAPS数据集
- AkPnBcht
- AkPnBsdf
- AkPnCGdD2

以下数据来自https://github.com/bytedance/GiantMIDI-Piano/tree/master/midis_for_evaluation
- giantmidi-piano
- ground_truth
- maestro

以下数据来自https://github.com/adlaneKadri/Music-Generation-RNNs
- midiData

以下数据来自https://www.flutetunes.com/titles.php
- flutetunes

flutetunes网站的爬虫如下，由于脚本问题只爬了solo的一部分：
```py
import requests
from bs4 import BeautifulSoup
import tqdm

def downloadFile(url, filename):
    response = requests.get(url)
    with open(filename, 'wb') as file:
        file.write(response.content)

# 有bug，因为文件命名不一定都是这样
# 有些下载下来用不了
def flutetone():
    """
    example:
    https://www.flutetunes.com/tunes/tis-so-sweet-to-trust-in-jesus.mid
    https://www.flutetunes.com/tunes/79ths-farewell-to-gibraltar.mid
    """
    url = "https://www.flutetunes.com/tunes.php?instr=1"
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    table = soup.find('table', class_='tunes')
    tunes = [row.find_all('td')[0].text for row in table.find_all('tr')[1:]]
    for index, title in enumerate(tqdm.tqdm(tunes, desc="Downloading MIDI files"), start=1):
        _title = title.lower()
        _title = ''.join(char for char in _title if char.isalnum() or char.isspace())
        _title = _title.strip().replace(' ', '-')
        try:
            downloadFile(f"https://www.flutetunes.com/tunes/{_title}.mid", f"dataset/flutetunes/{_title}.mid")
        except Exception as e:
            print(f"Error downloading {title} (index {index}): {e}")

flutetone()

# 除去有问题的midi
import mido
import os
files = "flutetunes"
for root, dirs, files in os.walk(files):
    for file in files:
        midi_file = os.path.join(root, file)
        try:
            mido.MidiFile(midi_file)
        except:
            print(f"Removing {midi_file}")
            os.remove(midi_file)
```

## MAPS
MAPS 数据集是为了多音高估计和音乐自动转录而设计的钢琴录音数据库，其结构如下：

### 顶层目录
- **ISOL**：包含孤立音符和单音音乐片段。用于测试单音高估计算法或在需要孤立音符时训练多音高算法。
- **RAND**：包含随机音符组成的和弦。用于在没有任何先验音乐知识的情况下客观评估算法。
- **UCHO**：包含西方音乐中常见的和弦，如爵士乐或古典音乐中的和弦。用于在有先验知识的情况下评估性能。
- **MUS**：包含从互联网上获得的高质量标准 MIDI 文件生成的音乐片段。用于评估算法在真实音乐场景中的性能。

### 子目录结构
每个顶层目录下根据录音条件和使用的乐器模型进一步细分为多个子目录。例如：
- **ISOL** 下有多个子目录，每个子目录代表一种特定的录音条件和乐器模型组合，如 `StbgTGd2`、`AkPnBsdf` 等。
- **RAND**、**UCHO** 和 **MUS** 目录也有类似的子目录结构。

### 文件命名规则
- **ISOL** 文件命名规则：`MAPS_ISOL_ps_i0_Ss_Mm_instrName.wav`
  - `ps`：演奏风格，如 `NO`（正常演奏的2秒长音符）、`LG`（长音符）、`ST`（断奏）、`RE`（逐渐加快的重复音符）等。
  - `i0`：响度，如 `P`（弱）、`M`（中等）、`F`（强）。
  - `Ss`：是否使用延音踏板，`s=1` 表示使用。
  - `Mm`：音高，以 MIDI 代码表示。
  - `instrName`：乐器名称代码，如 `StbgTGd2`、`AkPnBsdf` 等。
- **RAND** 文件命名规则：`MAPS_RAND_Px_Mm1-m2_Ii1-i2_Ss_nn_instrName.wav`
  - `Px`：多音高水平，从2到7。
  - `Mm1-m2`：音高范围，如 `21-108`（全钢琴范围）或 `36-95`（中心5个八度）。
  - `Ii1-i2`：响度范围，如 `60-68`（中等响度）或 `32-96`（从弱到强）。
  - `Ss`：是否使用延音踏板。
  - `nn`：结果索引。
- **UCHO** 文件命名规则：`MAPS_UCHO_Cc1-...-cp_Ii1-i2_Ss_nn_instrName.wav`
  - `Cc1-...-cp`：和弦内容，表示和弦中每个音符与根音的距离（以半音为单位）。
  - `Ii1-i2`：音高范围。
  - `Ss`：是否使用延音踏板。
  - `nn`：结果索引。
- **MUS** 文件命名规则：`MAPS_MUS_description_instrName.wav`
  - `description`：音乐片段的描述。
  - `instrName`：乐器名称代码。

### MIDI 文件
- 每个音频文件都有对应的 MIDI 文件作为真实值（ground truth），用于算法评估。
