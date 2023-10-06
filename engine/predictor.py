import torch
import torchvision.transforms as T
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2
import yaml
import colorsys
import matplotlib.pyplot as plt

from models import build_model

class Predictor:
    def __init__(self, model_path, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # self.device = torch.device('cpu')
        self.model, self.postprocessor = self.load_model(model_path)
        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.class_names = self.load_dataset_format()
        self.colors = self.generate_colors(len(self.class_names))

    def load_dataset_format(self):
        dataset_cfg = "datasets/" + self.args.datasets_yaml
        # 读取 YAML 文件
        with open(dataset_cfg, 'r') as file:
            data = yaml.safe_load(file)
        return data['names']
    
    def load_model(self, model_path):
        model_dict = torch.load(model_path, map_location=self.device)['model']
        model, postprocessor = build_model(self.args)
        model.load_state_dict(model_dict)
        model = model.to(self.device)
        model.eval()
        return model, postprocessor

    def preprocess_image(self, image_path):
        self.image = Image.open(image_path).convert("RGB")
        plt.imshow(self.image)
        plt.show()
        image_tensor = self.transform(self.image)
        image_tensor = image_tensor.unsqueeze(0).to(self.device)
        return image_tensor
    
    @staticmethod
    def generate_colors(num_colors):
        # 生成一系列不同的颜色
        colors = []
        for i in range(num_colors):
            hue = i / num_colors  # 在 HSV 色彩空间中均匀分布的色相值
            rgb = colorsys.hsv_to_rgb(hue, 1.0, 1.0)  # 将色相转换为 RGB
            scaled_rgb = tuple(int(255 * value) for value in rgb)  # 将 RGB 值缩放到 [0, 255] 范围
            colors.append(scaled_rgb)
        return colors
    
    def draw_res(self, image, top_label, top_conf, top_boxes, font=None, thickness=2):
        if font is None:
            # 加载字体文件并创建字体对象
            font = ImageFont.truetype("util/ARIAL.TTF", size=16)

        for i, c in list(enumerate(top_label)):
            c = int(c) - 1
            predicted_class = self.class_names[int(c)]
            box             = top_boxes[i]
            score           = top_conf[i]

            top, left, bottom, right = box

            top     = max(0, np.floor(top).astype('int32'))
            left    = max(0, np.floor(left).astype('int32'))
            bottom  = min(image.size[1], np.floor(bottom).astype('int32'))
            right   = min(image.size[0], np.floor(right).astype('int32'))

            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)
            label = label.encode('utf-8')
            print(label, top, left, bottom, right)
            
            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            for i in range(thickness):
                draw.rectangle([left + i, top + i, right - i, bottom - i], outline=self.colors[c])
            draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=self.colors[c])
            draw.text(text_origin, str(label,'UTF-8'), fill=(0, 0, 0), font=font)
            del draw

        return image

    def predict(self, image_path, confidence=0.5):
        image_tensor = self.preprocess_image(image_path)
        with torch.no_grad():
            output = self.model(image_tensor)
        # Process the output as needed
        pred_logits = output['pred_logits']
        pred_boxes = output['pred_boxes']
        target_sizes = torch.tensor([image_tensor.shape[-2:]] * len(pred_logits), dtype=torch.float, device=pred_logits.device)
        
        print("class predict size: ", pred_logits.shape)
        print("boxes predict size: ", pred_boxes.shape)
        print("image size: ", image_tensor.shape)
        print("target size: ", target_sizes.shape)

        output = self.postprocessor['bbox'](output, target_sizes)
        # print("output size: ", len(output))

        outputs = torch.cat([
                torch.unsqueeze(output[0]['boxes'][ :, 1], -1),
                torch.unsqueeze(output[0]['boxes'][ :, 0], -1),
                torch.unsqueeze(output[0]['boxes'][ :, 3], -1),
                torch.unsqueeze(output[0]['boxes'][ :, 2], -1),
                torch.unsqueeze(output[0]['scores'], -1),
                torch.unsqueeze(output[0]['labels'].float(), -1),    
            ], -1)
        
        print("outputs size: ", outputs.shape)
        results = []
        results.append(outputs[outputs[:, 4] > confidence])
        # for output in outputs:
        #     print("output size: ", output.shape)
        #     results.append(output[output[4] > confidence])

        for result in results:
            pred = result.cpu().numpy()
            top_label = pred[:, 5]
            top_conf = pred[:, 4]
            top_boxes = pred[:, :4]
            pre_image = self.draw_res(self.image, top_label, top_conf, top_boxes)
            plt.imshow(pre_image)
            plt.show()

        print("results size: ", len(results))
        return results