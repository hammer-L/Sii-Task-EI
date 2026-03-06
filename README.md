# TASK1

### 1.结果可视化

![loss](figures\task1-loss.png)

![acc](figures\task1-acc.png)

# TASK2

### 1.结果可视化

![1772699981985](figures\task2-acc.png)

![1772700019607](figures\task2-loss.png)

### 2.结果分析

#### （1）use_glove与no_glove：

使用glove做embedding效果显著好于不用。一方面，使用glove可以一定程度上防止过拟合，train与test上的loss差别并没有非常大，test上的loss也没有出现发散的情况，最后准确率也更高；另一方面，使用glove可以加速模型收敛速度。

#### （2）cnn的kernel_size多大比较合适：

比较了use_glove条件下kernel_size=1,2,3,5的情况。在这个情感分类的任务上，k=5的情况明显不如k=1,2,3的情况，而k=1时模型表现最好，拥有最高的acc。推测原因：句子比较短，训练集并不大，句子分类的依据可能更多由一些关键词如“happy”所决定，k=1时模型表现就已经很好。且随着k的增大，



# EI-TASK-4

#### 1. 使用Starvla训练

VLM选用Qwen3-VL-4B-Instruct，Action head选用GR00T，训练集使用libero_mix data。使用4张4090进行训练，但是因为要训练170h左右，只是训练了一百多轮就没有继续了。

![1772695918285](figures\ei-task4-train_config.png)

#### 2.推理

在hf上下载训练好的权重进行测试。下载`Qwen2.5-VL-GR00T-LIBERO-4in1_checkpoints_steps_30000_pytorch_mode.pt`的权重，在libero_goal上进行推理。

![1772696434799](figures\ei-task4-eval_config.png)

**500 episodes下正确率为0.958**

![1772698615045](figures\ei-task4-eval.png)