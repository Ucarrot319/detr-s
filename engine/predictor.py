import torch
import torchvision.transforms as T
from PIL import Image

from models import build_model

class Predictor:
    def __init__(self, model_path, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model, self.postprocessor = self.load_model(model_path)
        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def load_model(self, model_path):
        model_dict = torch.load(model_path, map_location=self.device)
        model, postprocessors = build_model(self.args)
        model.load_state_dict(model_dict)
        model = model.to(self.device)
        model.eval()
        return model, postprocessors

    def preprocess_image(self, image_path):
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        image = image.unsqueeze(0).to(self.device)
        return image

    def predict(self, image_path):
        image = self.preprocess_image(image_path)
        with torch.no_grad():
            output = self.model(image)
        # Process the output as needed
        pred_logits = output['pred_logits']
        pred_boxes = output['pred_boxes']

        print(pred_logits.shape)
        print(pred_boxes.shape)

        output = self.postprocessor(output, torch.tensor(image.shape[-2:]))
        return output