import numpy as np
import torch
import torch.nn.functional as f

from rebalta.models.util import build_convolutional_block, AttentionBlock, \
    build_classifier_block, \
    build_classifier_confidence


class Model2(torch.nn.Module):

    def __init__(self, num_of_classes: int, input_dimension: int, uses_attention: bool = False):
        """
        Initializer of model with two attention modules.

        :param num_of_classes: Number of classes.
        :param input_dimension: Input spectral size.
        :param uses_attention: Boolean variable indicating whether the model uses attention mechanism or not.
        """
        super().__init__()
        self._conv_block_1 = build_convolutional_block(1, 96)
        self._conv_block_2 = build_convolutional_block(96, 54)
        assert int(input_dimension / 4) > 0, "The spectral size is to small for model with two attention modules."
        self._classifier = build_classifier_block(54 * int(input_dimension / 4), num_of_classes)
        if uses_attention:
            print("Model with 2 attention modules.")
            self._attention_block_1 = AttentionBlock(96, int(input_dimension / 2), num_of_classes)
            self._attention_block_2 = AttentionBlock(54, int(input_dimension / 4), num_of_classes)
            self._classifier_confidence = build_classifier_confidence(54 * int(input_dimension / 4))
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.0001)
        self.loss = torch.nn.BCELoss()
        self.uses_attention = uses_attention

    def forward(self, x: torch.Tensor, y: torch.Tensor, infer: bool) -> torch.Tensor:
        """
       Feed forward method for model with two attention modules.

       :param x: Input tensor.
       :param y: Labels.
       :param infer: Boolean variable indicating whether to save attention heatmap which is later used in the
                     band selection process.
       :return: Weighted classifier hypothesis.
       """
        global first_module_prediction, second_module_prediction
        z = self._conv_block_1(x)
        if self.uses_attention:
            first_module_prediction = self._attention_block_1(z, y, infer)
        z = self._conv_block_2(z)
        if self.uses_attention:
            second_module_prediction = self._attention_block_2(z, y, infer)
        prediction = self._classifier(z.view(z.shape[0], -1))
        if self.uses_attention:
            prediction *= self._classifier_confidence(z.view(z.shape[0], -1))
        if self.uses_attention:
            return f.softmax(prediction + first_module_prediction + second_module_prediction, dim=1)
        return f.softmax(prediction, dim=1)

    def get_heatmaps(self, input_size: int) -> np.ndarray:
        """
        Return averaged heatmaps for model with two attention modules.

        :param input_size: Designed spectral size.
        :return: Array containing averaged scores for interpolated heatmaps.
        """
        return np.mean([self._attention_block_1.get_heatmaps(input_size).squeeze(),
                        self._attention_block_2.get_heatmaps(input_size).squeeze()], axis=0).squeeze()