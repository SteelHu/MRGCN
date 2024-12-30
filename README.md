class TimesBlock(nn.Module):
    def __init__(self, configs):
        # 初始化TimesBlock类，继承自nn.Module
        super(TimesBlock, self).__init__()
        # 调用父类nn.Module的初始化方法

        self.seq_len = configs.seq_len
        # 序列长度，从配置中获取
        self.pred_len = configs.pred_len
        # 预测长度，从配置中获取
        self.k = configs.top_k
        # 保留的周期数，从配置中获取

        # 参数高效的卷积设计
        self.conv = nn.Sequential(
            # 使用nn.Sequential定义一个顺序容器，包含多个子模块
            Inception_Block_V1(configs.d_model, configs.d_ff,
                               num_kernels=configs.num_kernels),
            # 第一个Inception_Block_V1，输入维度为d_model，输出维度为d_ff
            nn.GELU(),
            # GELU激活函数
            Inception_Block_V1(configs.d_ff, configs.d_model,
                               num_kernels=configs.num_kernels)
            # 第二个Inception_Block_V1，输入维度为d_ff，输出维度为d_model
        )

    def forward(self, x):
        # 定义前向传播函数
        B, T, N = x.size()
        # 获取输入x的批次大小B，时间步长T，特征维度N

        period_list, period_weight = FFT_for_Period(x, self.k)
        # 使用FFT_for_Period函数计算周期列表和周期权重

        res = []
        # 初始化结果列表
        for i in range(self.k):
            # 遍历每个周期
            period = period_list[i]
            # 获取当前周期

            # 如果序列长度和预测长度之和不能被当前周期整除，则进行填充
            if (self.seq_len + self.pred_len) % period != 0:
                length = (
                                 ((self.seq_len + self.pred_len) // period) + 1) * period
                # 计算填充后的长度
                padding = torch.zeros([x.shape[0], (length - (self.seq_len + self.pred_len)), x.shape[2]]).to(x.device)
                # 创建填充张量
                out = torch.cat([x, padding], dim=1)
                # 将填充张量与原输入x拼接
            else:
                length = (self.seq_len + self.pred_len)
                # 如果能整除，则直接使用原长度
                out = x
                # 使用原输入x

            # 调整张量形状以适应2D卷积
            out = out.reshape(B, length // period, period,
                              N).permute(0, 3, 1, 2).contiguous()
            # 将张量形状调整为[B, N, length // period, period]

            # 使用2D卷积进行特征提取
            out = self.conv(out)  # [B, N, length // period, period] -> [B, N, length // period, period]

            # 将张量形状恢复原状
            out = out.permute(0, 2, 3, 1).reshape(B, -1, N)
            res.append(out[:, :(self.seq_len + self.pred_len), :])
            # 将结果添加到结果列表中
        res = torch.stack(res, dim=-1)
        # 将结果列表堆叠成一个新的张量

        # 自适应聚合
        period_weight = F.softmax(period_weight, dim=1)
        # 对周期权重进行softmax归一化
        period_weight = period_weight.unsqueeze(
            1).unsqueeze(1).repeat(1, T, N, 1)
        # 调整周期权重的形状以适应广播
        res = torch.sum(res * period_weight, -1)
        # 根据周期权重对结果进行加权求和

        # 残差连接
        res = res + x
        # 将结果与输入x相加，实现残差连接

        return res
        # 返回最终结果
class TimesBlock(nn.Module):
    def __init__(self, configs):
        # 初始化TimesBlock类，继承自nn.Module
        super(TimesBlock, self).__init__()
        # 调用父类nn.Module的初始化方法

        self.seq_len = configs.seq_len
        # 序列长度，从配置中获取
        self.pred_len = configs.pred_len
        # 预测长度，从配置中获取
        self.k = configs.top_k
        # 保留的周期数，从配置中获取

        # 参数高效的卷积设计
        self.conv = nn.Sequential(
            # 使用nn.Sequential定义一个顺序容器，包含多个子模块
            Inception_Block_V1(configs.d_model, configs.d_ff,
                               num_kernels=configs.num_kernels),
            # 第一个Inception_Block_V1，输入维度为d_model，输出维度为d_ff
            nn.GELU(),
            # GELU激活函数
            Inception_Block_V1(configs.d_ff, configs.d_model,
                               num_kernels=configs.num_kernels)
            # 第二个Inception_Block_V1，输入维度为d_ff，输出维度为d_model
        )

    def forward(self, x):
        # 定义前向传播函数
        B, T, N = x.size()
        # 获取输入x的批次大小B，时间步长T，特征维度N

        period_list, period_weight = FFT_for_Period(x, self.k)
        # 使用FFT_for_Period函数计算周期列表和周期权重

        res = []
        # 初始化结果列表
        for i in range(self.k):
            # 遍历每个周期
            period = period_list[i]
            # 获取当前周期

            # 如果序列长度和预测长度之和不能被当前周期整除，则进行填充
            if (self.seq_len + self.pred_len) % period != 0:
                length = (
                                 ((self.seq_len + self.pred_len) // period) + 1) * period
                # 计算填充后的长度
                padding = torch.zeros([x.shape[0], (length - (self.seq_len + self.pred_len)), x.shape[2]]).to(x.device)
                # 创建填充张量
                out = torch.cat([x, padding], dim=1)
                # 将填充张量与原输入x拼接
            else:
                length = (self.seq_len + self.pred_len)
                # 如果能整除，则直接使用原长度
                out = x
                # 使用原输入x

            # 调整张量形状以适应2D卷积
            out = out.reshape(B, length // period, period,
                              N).permute(0, 3, 1, 2).contiguous()
            # 将张量形状调整为[B, N, length // period, period]

            # 使用2D卷积进行特征提取
            out = self.conv(out)  # [B, N, length // period, period] -> [B, N, length // period, period]

            # 将张量形状恢复原状
            out = out.permute(0, 2, 3, 1).reshape(B, -1, N)
            res.append(out[:, :(self.seq_len + self.pred_len), :])
            # 将结果添加到结果列表中
        res = torch.stack(res, dim=-1)
        # 将结果列表堆叠成一个新的张量

        # 自适应聚合
        period_weight = F.softmax(period_weight, dim=1)
        # 对周期权重进行softmax归一化
        period_weight = period_weight.unsqueeze(
            1).unsqueeze(1).repeat(1, T, N, 1)
        # 调整周期权重的形状以适应广播
        res = torch.sum(res * period_weight, -1)
        # 根据周期权重对结果进行加权求和

        # 残差连接
        res = res + x
        # 将结果与输入x相加，实现残差连接

        return res
        # 返回最终结果
# Time Series Library (TSLib)
TSLib is an open-source library for deep learning researchers, especially for deep time series analysis.

We provide a neat code base to evaluate advanced deep time series models or develop your model, which covers five mainstream tasks: **long- and short-term forecasting, imputation, anomaly detection, and classification.**

:triangular_flag_on_post:**News** (2024.10) We have included [[TimeXer]](https://arxiv.org/abs/2402.19072), which defined a practical forecasting paradigm: Forecasting with Exogenous Variables. Considering both practicability and computation efficiency, we believe the new forecasting paradigm defined in TimeXer can be the "right" task for future research.

:triangular_flag_on_post:**News** (2024.10) Our lab has open-sourced [[OpenLTM]](https://github.com/thuml/OpenLTM), which provides a distinct pretrain-finetuning paradigm compared to TSLib. If you are interested in Large Time Series Models, you may find this repository helpful.

:triangular_flag_on_post:**News** (2024.07) We wrote a comprehensive survey of [[Deep Time Series Models]](https://arxiv.org/abs/2407.13278) with a rigorous benchmark based on TSLib. In this paper, we summarized the design principles of current time series models supported by insightful experiments, hoping to be helpful to future research.

:triangular_flag_on_post:**News** (2024.04) Many thanks for the great work from [frecklebars](https://github.com/thuml/Time-Series-Library/pull/378). The famous sequential model [Mamba](https://arxiv.org/abs/2312.00752) has been included in our library. See [this file](https://github.com/thuml/Time-Series-Library/blob/main/models/Mamba.py), where you need to install `mamba_ssm` with pip at first.

:triangular_flag_on_post:**News** (2024.03) Given the inconsistent look-back length of various papers, we split the long-term forecasting in the leaderboard into two categories: Look-Back-96 and Look-Back-Searching. We recommend researchers read [TimeMixer](https://openreview.net/pdf?id=7oLshfEIC2), which includes both look-back length settings in experiments for scientific rigor.

:triangular_flag_on_post:**News** (2023.10) We add an implementation to [iTransformer](https://arxiv.org/abs/2310.06625), which is the state-of-the-art model for long-term forecasting. The official code and complete scripts of iTransformer can be found [here](https://github.com/thuml/iTransformer).

:triangular_flag_on_post:**News** (2023.09) We added a detailed [tutorial](https://github.com/thuml/Time-Series-Library/blob/main/tutorial/TimesNet_tutorial.ipynb) for [TimesNet](https://openreview.net/pdf?id=ju_Uqw384Oq) and this library, which is quite friendly to beginners of deep time series analysis.

:triangular_flag_on_post:**News** (2023.02) We release the TSlib as a comprehensive benchmark and code base for time series models, which is extended from our previous GitHub repository [Autoformer](https://github.com/thuml/Autoformer).

## Leaderboard for Time Series Analysis

Till March 2024, the top three models for five different tasks are:

| Model<br>Ranking | Long-term<br>Forecasting<br>Look-Back-96              | Long-term<br/>Forecasting<br/>Look-Back-Searching     | Short-term<br>Forecasting                                    | Imputation                                                   | Classification                                               | Anomaly<br>Detection                               |
| ---------------- | ----------------------------------------------------- | ----------------------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | -------------------------------------------------- |
| 🥇 1st            | [TimeXer](https://arxiv.org/abs/2402.19072)      | [TimeMixer](https://openreview.net/pdf?id=7oLshfEIC2) | [TimesNet](https://arxiv.org/abs/2210.02186)                 | [TimesNet](https://arxiv.org/abs/2210.02186)                 | [TimesNet](https://arxiv.org/abs/2210.02186)                 | [TimesNet](https://arxiv.org/abs/2210.02186)       |
| 🥈 2nd            | [iTransformer](https://arxiv.org/abs/2310.06625) | [PatchTST](https://github.com/yuqinie98/PatchTST)     | [Non-stationary<br/>Transformer](https://github.com/thuml/Nonstationary_Transformers) | [Non-stationary<br/>Transformer](https://github.com/thuml/Nonstationary_Transformers) | [Non-stationary<br/>Transformer](https://github.com/thuml/Nonstationary_Transformers) | [FEDformer](https://github.com/MAZiqing/FEDformer) |
| 🥉 3rd            | [TimeMixer](https://openreview.net/pdf?id=7oLshfEIC2)          | [DLinear](https://arxiv.org/pdf/2205.13504.pdf)       | [FEDformer](https://github.com/MAZiqing/FEDformer)           | [Autoformer](https://github.com/thuml/Autoformer)            | [Informer](https://github.com/zhouhaoyi/Informer2020)        | [Autoformer](https://github.com/thuml/Autoformer)  |


**Note: We will keep updating this leaderboard.** If you have proposed advanced and awesome models, you can send us your paper/code link or raise a pull request. We will add them to this repo and update the leaderboard as soon as possible.

**Compared models of this leaderboard.** ☑ means that their codes have already been included in this repo.
  - [x] **TimeXer** - TimeXer: Empowering Transformers for Time Series Forecasting with Exogenous Variables [[NeurIPS 2024]](https://arxiv.org/abs/2402.19072) [[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/TimeXer.py)
  - [x] **TimeMixer** - TimeMixer: Decomposable Multiscale Mixing for Time Series Forecasting [[ICLR 2024]](https://openreview.net/pdf?id=7oLshfEIC2) [[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/TimeMixer.py).
  - [x] **TSMixer** - TSMixer: An All-MLP Architecture for Time Series Forecasting [[arXiv 2023]](https://arxiv.org/pdf/2303.06053.pdf) [[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/TSMixer.py)
  - [x] **iTransformer** - iTransformer: Inverted Transformers Are Effective for Time Series Forecasting [[ICLR 2024]](https://arxiv.org/abs/2310.06625) [[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/iTransformer.py).
  - [x] **PatchTST** - A Time Series is Worth 64 Words: Long-term Forecasting with Transformers [[ICLR 2023]](https://openreview.net/pdf?id=Jbdc0vTOcol) [[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/PatchTST.py).
  - [x] **TimesNet** - TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis [[ICLR 2023]](https://openreview.net/pdf?id=ju_Uqw384Oq) [[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/TimesNet.py).
  - [x] **DLinear** - Are Transformers Effective for Time Series Forecasting? [[AAAI 2023]](https://arxiv.org/pdf/2205.13504.pdf) [[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/DLinear.py).
  - [x] **LightTS** - Less Is More: Fast Multivariate Time Series Forecasting with Light Sampling-oriented MLP Structures [[arXiv 2022]](https://arxiv.org/abs/2207.01186) [[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/LightTS.py).
  - [x] **ETSformer** - ETSformer: Exponential Smoothing Transformers for Time-series Forecasting [[arXiv 2022]](https://arxiv.org/abs/2202.01381) [[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/ETSformer.py).
  - [x] **Non-stationary Transformer** - Non-stationary Transformers: Exploring the Stationarity in Time Series Forecasting [[NeurIPS 2022]](https://openreview.net/pdf?id=ucNDIDRNjjv) [[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/Nonstationary_Transformer.py).
  - [x] **FEDformer** - FEDformer: Frequency Enhanced Decomposed Transformer for Long-term Series Forecasting [[ICML 2022]](https://proceedings.mlr.press/v162/zhou22g.html) [[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/FEDformer.py).
  - [x] **Pyraformer** - Pyraformer: Low-complexity Pyramidal Attention for Long-range Time Series Modeling and Forecasting [[ICLR 2022]](https://openreview.net/pdf?id=0EXmFzUn5I) [[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/Pyraformer.py).
  - [x] **Autoformer** - Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting [[NeurIPS 2021]](https://openreview.net/pdf?id=I55UqU-M11y) [[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/Autoformer.py).
  - [x] **Informer** - Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting [[AAAI 2021]](https://ojs.aaai.org/index.php/AAAI/article/view/17325/17132) [[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/Informer.py).
  - [x] **Reformer** - Reformer: The Efficient Transformer [[ICLR 2020]](https://openreview.net/forum?id=rkgNKkHtvB) [[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/Reformer.py).
  - [x] **Transformer** - Attention is All You Need [[NeurIPS 2017]](https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf) [[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/Transformer.py).

See our latest paper [[TimesNet]](https://arxiv.org/abs/2210.02186) for the comprehensive benchmark. We will release a real-time updated online version soon.

**Newly added baselines.** We will add them to the leaderboard after a comprehensive evaluation.
  - [x] **PAttn** - Are Language Models Actually Useful for Time Series Forecasting? [[NeurIPS 2024]](https://arxiv.org/pdf/2406.16964) [[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/PAttn.py)
  - [x] **Mamba** - Mamba: Linear-Time Sequence Modeling with Selective State Spaces [[arXiv 2023]](https://arxiv.org/abs/2312.00752) [[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/Mamba.py)
  - [x] **SegRNN** - SegRNN: Segment Recurrent Neural Network for Long-Term Time Series Forecasting [[arXiv 2023]](https://arxiv.org/abs/2308.11200.pdf) [[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/SegRNN.py).
  - [x] **Koopa** - Koopa: Learning Non-stationary Time Series Dynamics with Koopman Predictors [[NeurIPS 2023]](https://arxiv.org/pdf/2305.18803.pdf) [[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/Koopa.py).
  - [x] **FreTS** - Frequency-domain MLPs are More Effective Learners in Time Series Forecasting [[NeurIPS 2023]](https://arxiv.org/pdf/2311.06184.pdf) [[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/FreTS.py).
  - [x] **MICN** - MICN: Multi-scale Local and Global Context Modeling for Long-term Series Forecasting [[ICLR 2023]](https://openreview.net/pdf?id=zt53IDUR1U)[[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/MICN.py).
  - [x] **Crossformer** - Crossformer: Transformer Utilizing Cross-Dimension Dependency for Multivariate Time Series Forecasting [[ICLR 2023]](https://openreview.net/pdf?id=vSVLM2j9eie)[[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/Crossformer.py).
  - [x] **TiDE** - Long-term Forecasting with TiDE: Time-series Dense Encoder [[arXiv 2023]](https://arxiv.org/pdf/2304.08424.pdf) [[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/TiDE.py).
  - [x] **SCINet** - SCINet: Time Series Modeling and Forecasting with Sample Convolution and Interaction [[NeurIPS 2022]](https://openreview.net/pdf?id=AyajSjTAzmg)[[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/SCINet.py).
  - [x] **FiLM** - FiLM: Frequency improved Legendre Memory Model for Long-term Time Series Forecasting [[NeurIPS 2022]](https://openreview.net/forum?id=zTQdHSQUQWc)[[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/FiLM.py).
  - [x] **TFT** - Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting [[arXiv 2019]](https://arxiv.org/abs/1912.09363)[[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/TemporalFusionTransformer.py). 
 
## Usage

1. Install Python 3.8. For convenience, execute the following command.

```
pip install -r requirements.txt
```

2. Prepare Data. You can obtain the well pre-processed datasets from [[Google Drive]](https://drive.google.com/drive/folders/13Cg1KYOlzM5C7K8gK8NfC-F3EYxkM3D2?usp=sharing) or [[Baidu Drive]](https://pan.baidu.com/s/1r3KhGd0Q9PJIUZdfEYoymg?pwd=i9iy), Then place the downloaded data in the folder`./dataset`. Here is a summary of supported datasets.

<p align="center">
<img src=".\pic\dataset.png" height = "200" alt="" align=center />
</p>

3. Train and evaluate model. We provide the experiment scripts for all benchmarks under the folder `./scripts/`. You can reproduce the experiment results as the following examples:

```
# long-term forecast
bash ./scripts/long_term_forecast/ETT_script/TimesNet_ETTh1.sh
# short-term forecast
bash ./scripts/short_term_forecast/TimesNet_M4.sh
# imputation
bash ./scripts/imputation/ETT_script/TimesNet_ETTh1.sh
# anomaly detection
bash ./scripts/anomaly_detection/PSM/TimesNet.sh
# classification
bash ./scripts/classification/TimesNet.sh
```

4. Develop your own model.

- Add the model file to the folder `./models`. You can follow the `./models/Transformer.py`.
- Include the newly added model in the `Exp_Basic.model_dict` of  `./exp/exp_basic.py`.
- Create the corresponding scripts under the folder `./scripts`.

Note: 

(1) About classification: Since we include all five tasks in a unified code base, the accuracy of each subtask may fluctuate but the average performance can be reproduced (even a bit better). We have provided the reproduced checkpoints [here](https://github.com/thuml/Time-Series-Library/issues/494).

(2) About anomaly detection: Some discussion about the adjustment strategy in anomaly detection can be found [here](https://github.com/thuml/Anomaly-Transformer/issues/14). The key point is that the adjustment strategy corresponds to an event-level metric.

## Citation

If you find this repo useful, please cite our paper.

```
@inproceedings{wu2023timesnet,
  title={TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis},
  author={Haixu Wu and Tengge Hu and Yong Liu and Hang Zhou and Jianmin Wang and Mingsheng Long},
  booktitle={International Conference on Learning Representations},
  year={2023},
}

@article{wang2024tssurvey,
  title={Deep Time Series Models: A Comprehensive Survey and Benchmark},
  author={Yuxuan Wang and Haixu Wu and Jiaxiang Dong and Yong Liu and Mingsheng Long and Jianmin Wang},
  booktitle={arXiv preprint arXiv:2407.13278},
  year={2024},
}
```

## Contact
If you have any questions or suggestions, feel free to contact our maintenance team:

Current:
- Haixu Wu (Ph.D. student, wuhx23@mails.tsinghua.edu.cn)
- Yong Liu (Ph.D. student, liuyong21@mails.tsinghua.edu.cn)
- Yuxuan Wang (Ph.D. student, wangyuxu22@mails.tsinghua.edu.cn)
- Huikun Weng (Undergraduate, wenghk22@mails.tsinghua.edu.cn)

Previous:
- Tengge Hu (Master student, htg21@mails.tsinghua.edu.cn)
- Haoran Zhang (Master student, z-hr20@mails.tsinghua.edu.cn)
- Jiawei Guo (Undergraduate, guo-jw21@mails.tsinghua.edu.cn)

Or describe it in Issues.

## Acknowledgement

This project is supported by the National Key R&D Program of China (2021YFB1715200).

This library is constructed based on the following repos:

- Forecasting: https://github.com/thuml/Autoformer.

- Anomaly Detection: https://github.com/thuml/Anomaly-Transformer.

- Classification: https://github.com/thuml/Flowformer.

All the experiment datasets are public, and we obtain them from the following links:

- Long-term Forecasting and Imputation: https://github.com/thuml/Autoformer.

- Short-term Forecasting: https://github.com/ServiceNow/N-BEATS.

- Anomaly Detection: https://github.com/thuml/Anomaly-Transformer.

- Classification: https://www.timeseriesclassification.com/.

## All Thanks To Our Contributors

<a href="https://github.com/thuml/Time-Series-Library/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=thuml/Time-Series-Library" />
</a>
