# GraphRAG Evaluation Report

- Created at: 2026-06-15T21:57:43
- Graph path: `data\graph\patent_graph_llm_full20_fixed.json`
- Cases: 6
- Retrieval mode: `keyword`
- top_k / graph_top_k: 5 / 5
- Answer generation: False
- Expected graph hit rate: 1.0
- Expected any hit rate: 1.0
- Graph citation rate: 0.0
- Source citation rate: 0.0
- Average answer length: 0.0

## component_cleaning_devices

- Category: component
- Question: 哪些专利使用清洗箱、喷嘴或雾化器？它们分别解决了什么技术问题？
- Expected patents: CN207641928U, CN209489954U
- Source patents: CN207641928U, CN209489954U, CN207494109U
- Graph patents: CN207641928U, CN209489954U, CN207494109U, CN110237372A, CN208612892U
- Expected in graph: CN207641928U, CN209489954U
- Missing expected: none
- Citations: source=False, graph=False
- Answer length: 0

Answer preview:

(skipped)

Graph evidence preview:

- G1 CN207641928U: components=['箱体', '转动轴', '清洗箱', '加热装置', '喷嘴', '底盘']; problems=['眼科手术器械清洗和烘干分别进行']
- G2 CN209489954U: components=['超声雾化装置', '清洗头', '吸尘器主机', '雾化器']; problems=['高压水冲法导致污水外流污染环境', '传统清洁纱窗方式工作量大、易破网、不易清洁干净']
- G3 CN207494109U: components=['方形格栅', '内框架', '清洗框本体']; problems=['奶嘴装载量少', '奶嘴易碰触设备侧壁', '奶嘴清洗效果差']
- G4 CN110237372A: components=['雾化组件', '温度调控组件', '运算系统', '视频采集传感器', '框架系统', '通信接口']; problems=['小儿不能很好配合雾化治疗导致疾病控制不佳']
- G5 CN208612892U: components=['操作台', '石蜡油涂抹机构', '行走轮', '夹持机构', '机架']; problems=['现有涂管器使用不便且难以清洗', '石蜡油涂抹操作繁琐、不均匀']

## problem_nursing_support

- Category: problem
- Question: 哪些专利解决了护理床翻身、身体支撑或长期受压相关问题？
- Expected patents: CN107049658A, CN208193238U
- Source patents: CN107049658A, CN107242942A, CN109730872A, CN209316357U
- Graph patents: CN107049658A, CN109730872A, CN107242942A, CN209316357U, CN208193238U
- Expected in graph: CN107049658A, CN208193238U
- Missing expected: none
- Citations: source=False, graph=False
- Answer length: 0

Answer preview:

(skipped)

Graph evidence preview:

- G1 CN107049658A: components=['胸部保护机构', '第二MEMS传感器模块', '第一MEMS传感器模块', '控制器', '联动机构', '臀部转动机构']; problems=['病人身体长期受压造成局部缺氧、血管栓塞、组织坏死腐脱']
- G2 CN109730872A: components=['控制器', '记录台', '按摩机构']; problems=['易造成婴幼儿不适', '功能单一', '婴幼儿发育生长健康标准测量不易操作']
- G3 CN107242942A: components=['控制器', '靠背机构', '电动升降杆', '第一MEMS传感器模块', '平台']; problems=['病人在康复过程中不能自由转动身体', '病人不能自由移动']
- G4 CN209316357U: components=['操作板', '底板', '座板', '升降板', '支撑块', '座板伸缩杆']; problems=['不能有效对患者腿部进行支撑固定']
- G5 CN208193238U: components=['四层钢架', '面板', '立杆', '遮挡铁网']; problems=['住院病区加床后每床仪器设备放置和生活物品放置矛盾', '病床医疗设备摆放预留面积缺失或锐减']

## automation_sensors

- Category: solution
- Question: 有哪些专利通过MEMS传感器模块、控制器或自动化机构改善护理或康复效果？
- Expected patents: CN107242942A, CN107049658A, CN109730872A
- Source patents: CN107242942A, CN107049658A, CN106859869A
- Graph patents: CN107242942A, CN107049658A, CN106859869A, CN109730872A, CN208193238U
- Expected in graph: CN107242942A, CN107049658A, CN109730872A
- Missing expected: none
- Citations: source=False, graph=False
- Answer length: 0

Answer preview:

(skipped)

Graph evidence preview:

- G1 CN107242942A: components=['控制器', '靠背机构', '电动升降杆', '第一MEMS传感器模块', '平台']; problems=['病人在康复过程中不能自由转动身体', '病人不能自由移动']
- G2 CN107049658A: components=['胸部保护机构', '第二MEMS传感器模块', '第一MEMS传感器模块', '控制器', '联动机构', '臀部转动机构']; problems=['病人身体长期受压造成局部缺氧、血管栓塞、组织坏死腐脱']
- G3 CN106859869A: components=['微创超低温冷冻消融肿瘤的医疗设备', '弹性支撑机构', 'X射线检测模块']; problems=['地震灾区手术室建设慢、隔离卫生不达标、设备不足', '山区医疗设备缺乏导致疾病无法诊断治疗', '转运危重患者时搬动导致病情加重或死亡']
- G4 CN109730872A: components=['控制器', '记录台', '按摩机构']; problems=['易造成婴幼儿不适', '功能单一', '婴幼儿发育生长健康标准测量不易操作']
- G5 CN208193238U: components=['四层钢架', '面板', '立杆', '遮挡铁网']; problems=['住院病区加床后每床仪器设备放置和生活物品放置矛盾', '病床医疗设备摆放预留面积缺失或锐减']

## technical_field_ophthalmology

- Category: technical_field
- Question: 哪些专利属于眼科手术或眼科器械相关技术领域？它们的关键组件是什么？
- Expected patents: CN207641928U, CN110236704A
- Source patents: CN207641928U, CN110236704A, CN209316357U, CN107692961A
- Graph patents: CN207641928U, CN110236704A, CN209316357U, CN107692961A, CN106859869A
- Expected in graph: CN207641928U, CN110236704A
- Missing expected: none
- Citations: source=False, graph=False
- Answer length: 0

Answer preview:

(skipped)

Graph evidence preview:

- G1 CN207641928U: components=['箱体', '转动轴', '清洗箱', '加热装置', '喷嘴', '底盘']; problems=['眼科手术器械清洗和烘干分别进行']
- G2 CN110236704A: components=['换刀组件', '一次性刀套', '台体', '换工具组件']; problems=['使用后的手术刀和未使用的手术刀容易接触产生交叉感染', '现有搁手台功能简单、卫生清洁程度不高']
- G3 CN209316357U: components=['操作板', '底板', '座板', '升降板', '支撑块', '座板伸缩杆']; problems=['不能有效对患者腿部进行支撑固定']
- G4 CN107692961A: components=[]; problems=[]
- G5 CN106859869A: components=['微创超低温冷冻消融肿瘤的医疗设备', '弹性支撑机构', 'X射线检测模块']; problems=['地震灾区手术室建设慢、隔离卫生不达标、设备不足', '山区医疗设备缺乏导致疾病无法诊断治疗', '转运危重患者时搬动导致病情加重或死亡']

## effect_cleaning_efficiency

- Category: effect
- Question: 哪些方案提升了清洗效率、清洁效果或降低污染风险？
- Expected patents: CN207641928U, CN209489954U, CN207494109U
- Source patents: CN207494109U, CN209489954U, CN108888820A
- Graph patents: CN207494109U, CN209489954U, CN108888820A, CN110236704A, CN207641928U
- Expected in graph: CN207641928U, CN209489954U, CN207494109U
- Missing expected: none
- Citations: source=False, graph=False
- Answer length: 0

Answer preview:

(skipped)

Graph evidence preview:

- G1 CN207494109U: components=['方形格栅', '内框架', '清洗框本体']; problems=['奶嘴装载量少', '奶嘴易碰触设备侧壁', '奶嘴清洗效果差']
- G2 CN209489954U: components=['超声雾化装置', '清洗头', '吸尘器主机', '雾化器']; problems=['高压水冲法导致污水外流污染环境', '传统清洁纱窗方式工作量大、易破网、不易清洁干净']
- G3 CN108888820A: components=['污液罐', '洗胃液储存罐', '底板', '安装架', '箱体']; problems=['洗胃罐和污液罐与洗胃设备一体设计不可拆卸，不利于充分清洗']
- G4 CN110236704A: components=['换刀组件', '一次性刀套', '台体', '换工具组件']; problems=['使用后的手术刀和未使用的手术刀容易接触产生交叉感染', '现有搁手台功能简单、卫生清洁程度不高']
- G5 CN207641928U: components=['箱体', '转动轴', '清洗箱', '加热装置', '喷嘴', '底盘']; problems=['眼科手术器械清洗和烘干分别进行']

## idea_medical_cleaning_device

- Category: idea_analysis
- Question: 我想设计一个医疗器械清洗装置，可以借鉴哪些现有专利结构，并需要避免哪些问题？
- Expected patents: CN207641928U, CN207494109U, CN209489954U
- Source patents: CN209074691U, CN207641928U, CN209108420U
- Graph patents: CN207641928U, CN209074691U, CN209108420U, CN207494109U, CN209489954U
- Expected in graph: CN207641928U, CN207494109U, CN209489954U
- Missing expected: none
- Citations: source=False, graph=False
- Answer length: 0

Answer preview:

(skipped)

Graph evidence preview:

- G1 CN207641928U: components=['箱体', '转动轴', '清洗箱', '加热装置', '喷嘴', '底盘']; problems=['眼科手术器械清洗和烘干分别进行']
- G2 CN209074691U: components=['置袋盒', '制冷箱', '置管板']; problems=['无法集中收集检验', '无法快速选择取检容器', '样本常温下易凝固影响检验效果']
- G3 CN209108420U: components=['弧形板', '中频电磁治疗仪', '理疗仪', '电极片']; problems=['只能治疗某个部位，无法全方位治疗', '无法在治疗的同时辅助调节患者内分泌紊乱', '无法根据患者情况调节治疗装置']
- G4 CN207494109U: components=['方形格栅', '内框架', '清洗框本体']; problems=['奶嘴装载量少', '奶嘴易碰触设备侧壁', '奶嘴清洗效果差']
- G5 CN209489954U: components=['超声雾化装置', '清洗头', '吸尘器主机', '雾化器']; problems=['高压水冲法导致污水外流污染环境', '传统清洁纱窗方式工作量大、易破网、不易清洁干净']
