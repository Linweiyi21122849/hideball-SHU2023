import torch
import torch.nn as nn

class NeuralNet(nn.Module):
    def __init__(
        self, 
        #上肢判断是12
        input_size=12,
        hidden_size=256, 
        num_classes=7,
    ):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        return out

class HandsClassification:
    def __init__(self, path_model):
        self.path_model = path_model
        self.classes = ['Billet', 'Flat', 'Handsclaspedtogether', 'Handsup', 'Natural', 'Rightangle', 'Singleraise']
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.device = torch.device("cpu")
        self.load_model()

    def load_model(self):
        self.model = NeuralNet()
        self.model.load_state_dict(
            torch.load(self.path_model, map_location=self.device)
        )
        self.loss_fn = nn.CrossEntropyLoss()

    def __call__(self, input_keypoint):
        # 指定数据类型为 torch.float32
        if not type(input_keypoint) == torch.Tensor:
            input_keypoint = torch.tensor(input_keypoint, dtype=torch.float32)
        out = self.model(input_keypoint.unsqueeze(0))
        probabilities = torch.softmax(out, dim=-1)
        confidence, predict = torch.max(probabilities, -1)
        label_predict = self.classes[predict.item()]
        return label_predict, confidence.item()

if __name__ == '__main__':
    keypoint_classification = HandsClassification(
        path_model='models\Handspose_classification.pt'
    )
    dummy_input = torch.randn(23)
    classification, confidence = keypoint_classification(dummy_input)
    print(f'Classification: {classification}, Confidence: {confidence}')
