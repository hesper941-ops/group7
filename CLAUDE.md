# Claude Code 使用说明

## 基本原则

本地环境只用于阅读和修改代码，不包含完整运行环境。

不要在本地尝试运行、训练、测试或调试项目代码。任何依赖数据集、Python 环境、CUDA、GPU、Linux 路径或服务器依赖的操作，都应在服务器上完成。

## 目录限制

只允许修改当前项目中的 `workplace/` 目录下的内容。

不要修改 `workplace/` 目录外的任何文件，除非用户明确要求。

## Git 操作限制

不要执行任何 Git 操作。

包括但不限于：

```bash
git add
git commit
git push
git pull
git reset
git checkout
git merge
git rebase
git clean
```

所有 Git 操作必须由用户本人完成。

如果需要提交代码，只能告诉用户修改了哪些文件，并建议用户自行执行 Git 命令。

## 服务器路径说明

服务器端项目主目录为：

```bash
/share/home/tm1078571822880000/a904903640/group7
```

该服务器目录与 Claude Code 当前所在的本地项目目录处于同一项目层级。

本地只负责修改代码；服务器负责运行代码。


服务器端数据集主目录为：

```bash
/share/home/tm1078571822880000/a944494510/课程项目
```

该目录下文件树为：
.
|____dataset
| |____.DS_Store
| |____fisheye
| | |____Video_20260131_170142792.avi
| | |____Video_20260131_200029359.avi
| | |____Video_20260227_202553335.avi
| | |____Video_20260227_204121788.avi
| | |____Video_20260131_150734369.avi
| | |____Video_20260131_164304016.avi
| | |____Video_20260131_165208524.avi
| | |____Video_20260131_171253889.avi
| | |____Video_20260131_171648040.avi
| | |____Video_20260227_203348219.avi
| | |____Video_20260301_152459131.avi
| | |____Video_20260306_163434878.avi
| | |____Video_20260306_164407883.avi
| | |____Video_20260131_165614756.avi
| | |____Video_20260306_165839689.avi
| | |____Video_20260131_145524524.avi
| | |____Video_20260131_170539636.avi
| | |____Video_20260131_201556629.avi
| | |____Video_20260227_202953348.avi
| | |____Video_20260227_213001434.avi
| | |____Video_20260301_144803454.avi
| | |____Video_20260131_151916937.avi
| | |____Video_20260131_194854095.avi
| | |____Video_20260301_151942635.avi
| | |____Video_20260306_152340690.avi
| | |____Video_20260306_152721366.avi
| | |____Video_20260306_164902044.avi
| | |____Video_20260306_170449073.avi
| | |____Video_20260131_164745532.avi
| | |____Video_20260131_194205407.avi
| | |____Video_20260227_204553897.avi
| | |____Video_20260301_153037623.avi
| | |____Video_20260131_151559270.avi
| | |____Video_20260131_172143627.avi
| | |____Video_20260131_195202906.avi
| | |____Video_20260227_213404452.avi
| | |____Video_20260301_153434856.avi
| | |____Video_20260131_152410916.avi
| | |____Video_20260131_170919896.avi
| | |____Video_20260227_203753817.avi
| | |____Video_20260306_162401599.avi
| | |____Video_20260306_163105571.avi
| |____imu.csv
| |____AR_Data_Process3.0
| | |____.DS_Store
| | |____get_timestamp.py
| | |____ASR.py
| | |____imu.py
| | |____strong_gesture_test.py
| | |____get_timestamp_test.py
| | |____data
| | | |____.DS_Store
| | | |____._.DS_Store
| | |____models
| | | |____clip_teacher_model
| | | | |____preprocessor_config.json
| | | | |____config (1).json
| | | | |____pytorch_model.bin
| | | | |____tokenizer_config.json
| | | | |____preprocessor_config (1).json
| | | | |____tokenizer.json
| | | | |____config.json
| | | | |____pytorch_model(1).bin
| | | |____all-MiniLM-L6-v2
| | | | |____1_Pooling
| | | | | |____config.json
| | | | |____modules.json
| | | | |____sentence_bert_config.json
| | | | |____vocab.txt
| | | | |____pytorch_model.bin
| | | | |____tokenizer_config.json
| | | | |____special_tokens_map.json
| | | | |____tokenizer.json
| | | | |____config.json
| | |____strong_gesture2.0.py
| | |____._.DS_Store
| | |____mfcc.py
| | |____strong_gesture.py
| |____._imu.csv
| |____HoloLens
| | |____interaction_20260131_065459.mp4
| | |____interaction_20260131_072412.mp4
| | |____interaction_20260131_090917.mp4
| | |____interaction_20260131_120024.mp4
| | |____interaction_20260131_121548.mp4
| | |____interaction_20260227_122952.mp4
| | |____interaction_20260227_123354.mp4
| | |____interaction_20260227_132951.mp4
| | |____interaction_20260301_072503.mp4
| | |____interaction_20260301_073435.mp4
| | |____._interaction_20260131_114156.mp4
| | |____._interaction_20260131_115150.mp4
| | |____._interaction_20260301_073041.mp4
| | |____interaction_20260131_090541.mp4
| | |____interaction_20260131_091249.mp4
| | |____interaction_20260227_133408.mp4
| | |____interaction_20260306_083107.mp4
| | |____interaction_20260131_070722.mp4
| | |____interaction_20260131_085207.mp4
| | |____interaction_20260131_085611.mp4
| | |____interaction_20260131_090139.mp4
| | |____interaction_20260131_114852.mp4
| | |____interaction_20260306_085830.mp4
| | |____interaction_20260131_071552.mp4
| | |____interaction_20260131_092133.mp4
| | |____interaction_20260306_084853.mp4
| | |____._interaction_20260131_070722.mp4
| | |____._interaction_20260131_114852.mp4
| | |____interaction_20260131_084300.mp4
| | |____interaction_20260227_124559.mp4
| | |____interaction_20260301_064753.mp4
| | |____._interaction_20260131_092133.mp4
| | |____interaction_20260131_084732.mp4
| | |____interaction_20260301_071948.mp4
| | |____interaction_20260306_072721.mp4
| | |____._interaction_20260131_065459.mp4
| | |____._interaction_20260131_120024.mp4
| | |____interaction_20260227_122606.mp4
| | |____interaction_20260227_123745.mp4
| | |____interaction_20260227_124133.mp4
| | |____interaction_20260306_082346.mp4
| | |____interaction_20260306_083434.mp4
| | |____interaction_20260306_084406.mp4
| | |____interaction_20260131_071918.mp4
| | |____interaction_20260131_091657.mp4
| | |____interaction_20260131_114156.mp4
| | |____interaction_20260131_115150.mp4
| | |____interaction_20260301_073041.mp4
| | |____interaction_20260306_072344.mp4
| | |____interaction_20260306_090441.mp4
| |____._.DS_Store
|____code
| |____real_scene_utils.py
| |____.DS_Store
| |____feature_extraction
| | |____get_timestamp.py
| | |____ASR.py
| | |____imu.py
| | |____strong_gesture2.0.py
| | |____mfcc.py
| |____train_and_test.py
| |____README.md
| |____baseline_real_scene.py

服务端项目主目录和数据集主目录均包含 `code/` 与 `dataset/` 子目录。其中项目主目录仅包含数据集主目录内全部 Python 脚本，不包含 .bin、.mp4、.avi、.json、.csv 等文件。