import torch

IGNORE_INDEX = -100

def _get_ltor_masks_and_position_ids(self, max_seq_len, target, pad_token):
        """Build masks and position id for left to right model for SFT"""

        assert not self.config.reset_position_ids and not self.config.reset_attention_mask

        # Position ids.
        position_ids = torch.arange(max_seq_len, dtype=torch.long)

        # Loss mask.
        loss_mask = torch.ones(max_seq_len, dtype=torch.float)
        loss_mask[target == pad_token] = 0.0  # mask paddings
        loss_mask[target == IGNORE_INDEX] = 0.0  # mask prompts

        if self.config.create_attention_mask:
            attention_mask = torch.tril(
                torch.ones((max_seq_len, max_seq_len), device=target.device)
            ).unsqueeze(0)
            # Convert attention mask to binary:
            attention_mask = attention_mask < 0.5
        else:
            attention_mask = None

        return loss_mask, position_ids, attention_mask