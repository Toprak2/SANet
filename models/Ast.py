from transformers import ASTForAudioClassification, ASTFeatureExtractor
from torch import nn


class ASTWithPreprocessor(nn.Module):
    def __init__(self, speaker_to_id, sample_rate=16000,  max_length="max_length"):
        super().__init__()
        id_to_speaker = {i: name for name, i in speaker_to_id.items()}
        self.ast = ASTForAudioClassification.from_pretrained(
            "MIT/ast-finetuned-audioset-10-10-0.4593",
            num_labels=10,      # Set to our number of speakers
            label2id=speaker_to_id,       # Provide mappings
            id2label=id_to_speaker,
            ignore_mismatched_sizes=True  # IMPORTANT: Reinitialize the classifier head
        )
        self.feature_extractor = ASTFeatureExtractor.from_pretrained(
            "MIT/ast-finetuned-audioset-10-10-0.4593",
            sampling_rate=sample_rate,  # Ensure it knows the target SR
            # Usually AST doesn't need separate attention mask for input processing stage
            return_attention_mask=False

        )
        self.sample_rate = sample_rate
        self.max_length = max_length

    def forward(self, waveforms):
        # waveforms: list of np.ndarray or torch.Tensor of shape (n_samples,)
        waveforms = waveforms.squeeze(1).cpu().numpy()
        inputs = self.feature_extractor(
            waveforms,
            sampling_rate=self.sample_rate,
            padding=self.max_length,
            return_tensors="pt"
        )

        # move inputs to same device as AST
        pixel_values = inputs["input_values"].to(self.ast.device)
        outputs = self.ast(pixel_values)
        return outputs.logits
