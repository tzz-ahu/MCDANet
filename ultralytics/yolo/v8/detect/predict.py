# Ultralytics YOLO 🚀, AGPL-3.0 license

import torch

from ultralytics.yolo.engine.predictor import BasePredictor
from ultralytics.yolo.engine.results import Results
from ultralytics.yolo.utils import DEFAULT_CFG, ROOT, ops


class DetectionPredictor(BasePredictor):

    def postprocess(self, preds, img, orig_imgs):
        """Postprocesses predictions and returns a list of Results objects."""
        preds = ops.non_max_suppression(preds,
                                        self.args.conf,
                                        self.args.iou,
                                        agnostic=self.args.agnostic_nms,
                                        max_det=self.args.max_det,
                                        classes=self.args.classes)

        results = []
        for i, pred in enumerate(preds):
            orig_img = orig_imgs[i] if isinstance(orig_imgs, list) else orig_imgs
            if not isinstance(orig_imgs, torch.Tensor):
                pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
            path = self.batch[0]
            img_path = path[i] if isinstance(path, list) else path
            results.append(Results(orig_img=orig_img, path=img_path, names=self.model.names, boxes=pred))
        return results


def predict(cfg=DEFAULT_CFG, use_python=False):
    """Runs YOLO model inference on input image(s)."""
    #model = cfg.model or 'yolov8n.pt'
    model = "/DATA/wqs/yolov8/ultralytics/yolo/v8/detect/best.pt"
    source = "/DATA/wqs/yolov8/ultralytics/assets/open1_test/250212_CSPM146_1.png"

    # 定义文件路径
    # 定义文件路径
    file_path = "/DATA/wqs/yolov8/ultralytics/yolo/v8/detect/image_paths.txt"  # 替换为你的文件路径
    num =1
    # 打开文件并逐行读取
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            # 遍历文件的每一行
            for line_number, line in enumerate(file, start=1):
                # 去除行尾的换行符
                line = line.strip()
                args = dict(model=model, source=line.replace('appearance','open1_test'))
                if use_python:
                    from ultralytics import YOLO
                    YOLO(model)(**args)
                else:
                    predictor = DetectionPredictor(overrides=args)
                    predictor.predict_cli()
                # 输出当前行的内容
                print(num)
                num = num+1
    except FileNotFoundError:
        print(f"文件未找到，请检查路径是否正确：{file_path}")
    except Exception as e:
        print(f"读取文件时发生错误：{e}")




if __name__ == '__main__':
    predict()
