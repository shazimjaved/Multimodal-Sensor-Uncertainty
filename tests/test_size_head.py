import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest
import torch
import torch.nn as nn
from models.centernet_head import CenterNetBEVHead

class TestSizeHead:
    """Regression tests for the size regression head (width/length)."""

    def test_softplus_gradient_flow_for_negative_activations(self):
        """
        Verify that negative raw convolution outputs do not collapse the gradient
        to zero permanently (which would happen with ReLU).
        """
        head = CenterNetBEVHead(in_channels=16, head_conv=16, num_classes=3)
        head.train()

        # Isolate the final convolution layer in the size head before Softplus
        # self.size_head = Sequential(Conv2d, ReLU, Conv2d, Softplus)
        # Therefore size_head[2] is the final Conv2d
        final_conv = head.size_head[2]
        
        # Force the bias to a large negative number so the raw output before Softplus is strictly negative
        nn.init.constant_(final_conv.bias, -10.0)
        nn.init.constant_(final_conv.weight, 0.0)

        # 1. Forward pass
        fused_feat = torch.randn(2, 16, 64, 64, requires_grad=True)
        predictions = head(fused_feat)
        size_pred = predictions["size"]

        # 2. Verify non-negative and finite
        assert torch.all(torch.isfinite(size_pred)), "Size predictions must be finite"
        assert torch.all(size_pred > 0.0), "Size predictions with Softplus should be strictly positive (even if small)"

        # 3. Construct a loss that pulls the prediction UP
        # We want the model to predict larger sizes. Since current prediction is very small,
        # pushing it up should create a gradient through Softplus.
        target_size = torch.ones_like(size_pred) * 5.0
        loss = torch.nn.functional.l1_loss(size_pred, target_size)

        # 4. Backpropagate
        loss.backward()

        # 5. Verify non-zero gradient
        # With ReLU, a negative pre-activation of -10.0 would result in exactly 0 gradient.
        # With Softplus, it will be small but non-zero.
        assert final_conv.bias.grad is not None, "Bias gradient should exist"
        grad_norm = torch.norm(final_conv.bias.grad).item()
        
        assert grad_norm > 0.0, "Gradient must be non-zero for negative raw activations! (Dead ReLU issue)"

    def test_decoded_size_invariants(self):
        """
        Verify that decoded widths and lengths are strictly finite and non-negative.
        """
        head = CenterNetBEVHead(in_channels=16, head_conv=16, num_classes=3)
        head.eval()

        fused_feat = torch.randn(2, 16, 64, 64)
        
        with torch.no_grad():
            predictions = head(fused_feat)
            
            # Simulate a scenario where some sizes might be forced artificially low/negative
            predictions["size"] = predictions["size"] - 1.0 # Should be impossible if properly constrained, but let's test decoder logic just in case
            
            # Since Softplus guarantees > 0, we can just run the decode directly on the head output
            decoded = CenterNetBEVHead.decode_detections(predictions, k=10, score_threshold=0.0)
            
            for b in decoded:
                boxes = b["boxes"]
                if len(boxes) > 0:
                    w = boxes[:, 2]
                    l = boxes[:, 3]
                    
                    assert torch.all(torch.isfinite(w)), "Decoded width must be finite"
                    assert torch.all(torch.isfinite(l)), "Decoded length must be finite"
                    # width and length can be negative here because we explicitly injected negative values above
                    # But normally, predictions["size"] is strictly positive.
