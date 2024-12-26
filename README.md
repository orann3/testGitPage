# [RoboMIND: Benchmark on Multi-embodiment Intelligence Normative Data for Robot Manipulation](https://x-humanoid-robomind.github.io/)

[![License](https://img.shields.io/badge/License-Apache_2.0-yellow.svg)](https://opensource.org/licenses/Apache-2.0)
[![Project Page](https://img.shields.io/badge/Project%20Page-RoboMIND-blue.svg)](https://x-humanoid-robomind.github.io/)
[![arXiv](https://badgen.net/badge/icon/arXiv?icon=awesome&label&color=red&style=flat-square)](https://arxiv.org/abs/2412.13877)
[![Dataset](https://img.shields.io/badge/Dataset-TBU-000000.svg)](https://zitd5je6f7j.feishu.cn/share/base/form/shrcnOF6Ww4BuRWWtxljfs0aQqh)



## 📁 数据说明 📁
构建高质量的机器人训练数据集对开发具有良好泛化能力的端到端具身智能大模型至关重要。理想的数据集应涵盖多样化的场景、任务类型和机器人平台，使模型能够适应不同环境并可靠执行各类任务。本团队构建了一个大规模、真实的机器人学习数据集，记录机器人在复杂环境中执行长程任务时的交互数据，从而支持训练出具有通用操作能力的智能模型。

本数据集的部分目录结构示例如下，展示了Franka机器人下单个任务的2条训练轨迹和2条验证轨迹：
```
.
|-- h5_franka_1rgb
|   |-- bread_in_basket
|   |   `-- success_episodes
|   |       |-- train
|   |       |   |-- 1014_144236
|   |       |   |   `-- data
|   |       |   |       `-- trajectory.hdf5
|   |       |   |-- 1014_144602
|   |       |   |   `-- data
|   |       |   |       `-- trajectory.hdf5
|   |       |-- val
|   |       |   |-- 1014_144642
|   |       |   |   `-- data
|   |       |   |       `-- trajectory.hdf5
|   |       |   |-- 1014_151731
|   |       |   |   `-- data
|   |       |   |       `-- trajectory.hdf5
| -- h5_franka_3rgb
| -- h5_tiangong_1rg
| -- h5_ur_1rgb
| -- h5_songling_3rg
| -- h5_simulation
```

## 📊 数据分布 📊
<img src="./static/images/piechart.png" border=0 width=100%>

### 🤖 机器人平台构成 🤖
RoboMIND数据集汇集了多种机器人平台的操作数据，包括19,222条Franka Emika Panda单臂机器人轨迹、9,686条"天工"人形机器人轨迹、8,030条AgileX Cobot Magic V2.0双臂机器人轨迹、6,911条UR-5e单臂机器人轨迹，以及11,783条在仿真环境中采集的Franka机械臂数据。其中，自研的"天工"人形机器人贡献了17.4%的数据量，提供了大量需要高度协调能力的双臂操作任务。

### 🔎 轨迹时长分布 🔎
从轨迹长度来看，不同机器人平台呈现出独特的分布特征。Franka和UR机器人的任务通常具有较短的轨迹，时间步数少于200步，这类数据特别适合用于训练基础操作技能。相比之下，"天工"和AgileX机器人的任务轨迹普遍较长，超过500个时间步，更适合用于训练长时间跨度的任务和复杂技能组合。

### 🚀 任务类型划分 🚀
基于自然语言描述，并考虑物品大小、使用场景和操作技能等因素，我们将数据集中的任务分为基础技能、精准操作、场景理解、柜体操作和协作任务五大类。除了基础操作任务外，数据集还包含了大量复杂任务，为训练通用机器人策略提供了丰富的数据支持。

### 💪 物品多样性 💪
整个数据集包含了61种不同的物品类别，具体如下所示。可以看出，在厨房场景中，数据集不仅包含了常见的食物，如草莓、鸡蛋、香蕉和梨子等，也包括了复杂的可调节物体，如烤箱和面包机。在家庭场景中，数据集既包括了刚性物体，如网球，也包括了可变形物体，如玩具。办公和工业场景则包含了需要精确控制的小物体，如电池和齿轮。这样多样化的物体种类不仅增加了数据集的复杂性，也有助于训练能够在各种环境下执行操作的通用操控策略。

（物品图）

## 数据使用





### 📝 Citation 📝
If you find RoboMIND useful in your research, please consider citing:
```
@article{wu2024robomindbenchmarkmultiembodimentintelligence,
        title={RoboMIND: Benchmark on Multi-embodiment Intelligence Normative Data for Robot Manipulation},
        author={Kun Wu and Chengkai Hou and Jiaming Liu and Zhengping Che and Xiaozhu Ju and Zhuqin Yang and Meng Li and Yinuo Zhao and Zhiyuan Xu and Guang Yang and Zhen Zhao and Guangyu Li and Zhao Jin and Lecheng Wang and Jilei Mao and Xinhua Wang and Shichao Fan and Ning Liu and Pei Ren and Qiang Zhang and Yaoxu Lyu and Mengzhen Liu and Jingyang He and Yulin Luo and Zeyu Gao and Chenxuan Li and Chenyang Gu and Yankai Fu and Di Wu and Xingyu Wang and Sixiang Chen and Zhenyu Wang and Pengju An and Siyuan Qian and Shanghang Zhang and Jian Tang},
        journal={arXiv preprint arXiv:2412.13877},
        year={2024}
      }
```



## 参与讨论
如果您对 RoboMIND 感兴趣，欢迎加入微信群，参与讨论。

<img src="./static/images/piechart.png" border=0 width=50%>

