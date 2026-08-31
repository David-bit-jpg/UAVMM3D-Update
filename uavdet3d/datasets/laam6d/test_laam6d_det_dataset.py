import os
import unittest
import numpy as np
import yaml
import logging  # 导入logging模块
from types import SimpleNamespace
from uavdet3d.datasets.laam6d.laam6d_det_dataset import LAAM6D_Det_Dataset  # 导入数据集类


class TestLAAM6DDetDataset(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """加载配置文件并初始化数据集，包含日志器初始化"""
        # 配置文件路径
        cls.config_path = '/tools/cfgs/dataset_configs/uavdet_3d/laam6d.yaml'

        # 初始化一个基础日志器（替代None）
        cls.logger = cls._init_test_logger()

        # 加载配置文件并转换为SimpleNamespace
        cls.dataset_cfg = cls._load_config()

        # 验证数据路径是否存在
        cls._validate_data_path()

        # 初始化数据集（训练模式，传入日志器）
        cls.dataset_train = LAAM6D_Det_Dataset(
            dataset_cfg=cls.dataset_cfg,
            training=True,
            root_path=cls.dataset_cfg.DATA_PATH,
            logger=cls.logger  # 使用初始化的日志器
        )

        # 初始化数据集（验证模式，传入日志器）
        cls.dataset_val = LAAM6D_Det_Dataset(
            dataset_cfg=cls.dataset_cfg,
            training=False,
            root_path=cls.dataset_cfg.DATA_PATH,
            logger=cls.logger  # 使用初始化的日志器
        )

    @classmethod
    def _init_test_logger(cls):
        """初始化一个简单的测试日志器，输出到控制台"""
        logger = logging.getLogger("LAAM6D_Test")
        logger.setLevel(logging.INFO)  # 设置日志级别

        # 避免重复添加处理器
        if not logger.handlers:
            # 创建控制台处理器
            console_handler = logging.StreamHandler()
            # 设置日志格式
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            console_handler.setFormatter(formatter)
            # 添加处理器到日志器
            logger.addHandler(console_handler)

        return logger

    @classmethod
    def _load_config(cls):
        """从YAML文件加载配置并转换为可点号访问的SimpleNamespace对象"""
        if not os.path.exists(cls.config_path):
            raise FileNotFoundError(f"配置文件不存在: {cls.config_path}")

        with open(cls.config_path, 'r') as f:
            config_dict = yaml.safe_load(f)

        # 递归将字典转换为SimpleNamespace对象
        def dict_to_simplenamespace(d):
            if isinstance(d, dict):
                return SimpleNamespace(**{
                    k: dict_to_simplenamespace(v)
                    for k, v in d.items()
                })
            elif isinstance(d, list):
                return [dict_to_simplenamespace(v) for v in d]
            else:
                return d

        return dict_to_simplenamespace(config_dict)

    @classmethod
    def _validate_data_path(cls):
        """验证配置文件中指定的数据路径是否存在"""
        data_path = cls.dataset_cfg.DATA_PATH
        if not data_path:
            raise ValueError("配置文件中未指定DATA_PATH")

        if not os.path.exists(data_path):
            raise NotADirectoryError(f"数据路径不存在: {data_path}")

        cls.logger.info(f"已确认数据路径有效: {data_path}")  # 使用日志器输出

    def test_config_loading(self):
        """测试配置文件是否正确加载为SimpleNamespace"""
        self.assertIsInstance(self.dataset_cfg, SimpleNamespace)

        # 测试基础配置项（点号访问）
        self.assertTrue(hasattr(self.dataset_cfg, 'DATA_PATH'))
        self.assertTrue(hasattr(self.dataset_cfg, 'IM_SIZE'))
        self.assertTrue(hasattr(self.dataset_cfg, 'CLASS_NAMES'))
        self.assertTrue(hasattr(self.dataset_cfg, 'DATA_PRE_PROCESSOR'))

        # 测试DATA_PRE_PROCESSOR配置
        self.assertIsInstance(self.dataset_cfg.DATA_PRE_PROCESSOR, list)
        self.assertIsInstance(self.dataset_cfg.DATA_PRE_PROCESSOR[0], SimpleNamespace)
        self.assertEqual(self.dataset_cfg.DATA_PRE_PROCESSOR[0].NAME, 'filter_box_outside')

        # 用日志器打印关键配置信息
        self.logger.info(f"数据集路径: {self.dataset_cfg.DATA_PATH}")
        self.logger.info(f"图像尺寸: {self.dataset_cfg.IM_SIZE}")
        self.logger.info(f"类别名称: {self.dataset_cfg.CLASS_NAMES}")
        self.logger.info(f"预处理步骤数量: {len(self.dataset_cfg.DATA_PRE_PROCESSOR)}")

    def test_dataset_initialization(self):
        """测试数据集初始化是否正常"""
        # 测试训练集
        self.assertIsInstance(self.dataset_train, LAAM6D_Det_Dataset)
        self.assertGreater(len(self.dataset_train), 0, "训练集样本数量为0")

        # 测试验证集
        self.assertIsInstance(self.dataset_val, LAAM6D_Det_Dataset)
        self.assertGreater(len(self.dataset_val), 0, "验证集样本数量为0")

        self.logger.info(f"训练集样本数: {len(self.dataset_train)}")
        self.logger.info(f"验证集样本数: {len(self.dataset_val)}")

    def test_getitem_structure(self):
        """测试单个数据样本的结构是否符合预期"""
        # 测试训练集样本
        train_sample = self.dataset_train[0]
        self._validate_sample_structure(train_sample, "训练集")

        # 测试验证集样本
        val_sample = self.dataset_val[0]
        self._validate_sample_structure(val_sample, "验证集")

    def _validate_sample_structure(self, sample, dataset_type):
        """验证单个样本的结构是否正确"""
        # 检查必要的键是否存在
        required_keys = ['image', 'gt_boxes', 'intrinsic', 'extrinsic',
                         'raw_im_size', 'new_im_size', 'frame_id']
        for key in required_keys:
            self.assertIn(key, sample, f"{dataset_type}样本缺少键: {key}")

        # 检查图像形状
        im_resize = self.dataset_cfg.IM_RESIZE
        self.assertEqual(
            sample['image'].shape,
            (5, 3, im_resize[1], im_resize[0]),  # (模态数, 通道数, 高, 宽)
            f"{dataset_type}图像形状不符合预期"
        )

        # 检查GT框形状 (N个目标, 9个点, 3维坐标)
        self.assertEqual(
            sample['gt_boxes'].shape[1:],
            (9, 3),
            f"{dataset_type}GT框形状不符合预期"
        )

        # 检查内参/外参形状
        self.assertEqual(sample['intrinsic'].shape, (3, 3), f"{dataset_type}内参形状错误")
        self.assertEqual(sample['extrinsic'].shape, (4, 4), f"{dataset_type}外参形状错误")

        # 检查图像尺寸参数
        self.assertTrue(
            np.array_equal(sample['raw_im_size'], self.dataset_cfg.IM_SIZE),
            f"{dataset_type}原始图像尺寸不匹配"
        )
        self.assertTrue(
            np.array_equal(sample['new_im_size'], self.dataset_cfg.IM_RESIZE),
            f"{dataset_type}调整后图像尺寸不匹配"
        )

    def test_data_preprocessing(self):
        """测试数据预处理流程是否正常工作"""
        # 检查预处理队列是否正确初始化
        self.assertTrue(hasattr(self.dataset_train, 'data_processor_queue'), "数据集未初始化预处理队列")
        self.assertEqual(
            len(self.dataset_train.data_processor_queue),
            len(self.dataset_cfg.DATA_PRE_PROCESSOR),
            "预处理步骤数量不匹配配置"
        )

        # 验证预处理后的输出
        sample = self.dataset_train[0]
        self.assertIsNotNone(sample, "预处理后样本为None")

        # 特别验证convert_box9d_to_heatmap的输出（如果配置了）
        preprocessor_names = [p.NAME for p in self.dataset_cfg.DATA_PRE_PROCESSOR]
        if 'convert_box9d_to_heatmap' in preprocessor_names:
            self.assertIn('heatmap', sample, "预处理未生成heatmap")
            self.assertEqual(
                sample['heatmap'].shape[1:],  # 忽略批次维度
                (self.dataset_cfg.IM_RESIZE[1], self.dataset_cfg.IM_RESIZE[0]),
                "heatmap形状不符合预期"
            )

    def test_collate_batch(self):
        """测试批量处理函数是否正常工作"""
        # 生成一个小批量
        batch_size = min(4, len(self.dataset_train))  # 最多取4个样本
        batch = [self.dataset_train[i] for i in range(batch_size)]

        # 测试批量处理
        collated = self.dataset_train.collate_batch(batch)

        # 检查批量数据形状
        im_resize = self.dataset_cfg.IM_RESIZE
        self.assertEqual(
            collated['image'].shape,
            (batch_size, 5, 3, im_resize[1], im_resize[0]),
            "批量图像形状不符合预期"
        )

        # 检查批量大小
        self.assertEqual(collated['batch_size'], batch_size, "批量大小不匹配")

    def test_class_names_consistency(self):
        """测试数据集中的类别名称与配置是否一致"""
        sample = self.dataset_train[0]
        if 'gt_names' in sample:
            for name in sample['gt_names']:
                self.assertIn(name, self.dataset_cfg.CLASS_NAMES,
                             f"发现未知类别: {name}")


if __name__ == '__main__':
    unittest.main(verbosity=2)
