import unittest

import numpy as np

from src.core import (
    BahdanauAttention,
    LSTMCell,
    LSTMSequenceModel,
    MultiHeadAttention,
    TransformerBlock,
    VanillaRNNLanguageModel,
    create_causal_mask,
    positional_encoding,
    scaled_dot_product_attention,
)


class AttentionSmokeTests(unittest.TestCase):
    def test_scaled_dot_product_attention_shapes_and_normalization(self):
        rng = np.random.default_rng(0)
        q = rng.standard_normal((4, 8))
        k = rng.standard_normal((4, 8))
        v = rng.standard_normal((4, 8))
        out, weights = scaled_dot_product_attention(q, k, v)

        self.assertEqual(out.shape, (4, 8))
        self.assertEqual(weights.shape, (4, 4))
        np.testing.assert_allclose(weights.sum(axis=-1), np.ones(4))

    def test_causal_mask_blocks_future_positions(self):
        rng = np.random.default_rng(1)
        q = rng.standard_normal((3, 6))
        k = rng.standard_normal((3, 6))
        v = rng.standard_normal((3, 6))
        _, weights = scaled_dot_product_attention(q, k, v, mask=create_causal_mask(3))

        self.assertAlmostEqual(weights[0, 1], 0.0, places=8)
        self.assertAlmostEqual(weights[0, 2], 0.0, places=8)
        self.assertAlmostEqual(weights[1, 2], 0.0, places=8)

    def test_multi_head_attention_shape(self):
        rng = np.random.default_rng(2)
        mha = MultiHeadAttention(d_model=8, num_heads=2, rng=rng)
        x = rng.standard_normal((5, 8))
        out = mha.forward(x, x, x)

        self.assertEqual(out.shape, (5, 8))
        self.assertEqual(mha.attention_weights.shape, (2, 5, 5))

    def test_multi_head_attention_rejects_bad_shapes(self):
        mha = MultiHeadAttention(d_model=8, num_heads=2, rng=np.random.default_rng(22))
        with self.assertRaises(ValueError):
            mha.forward(np.ones((5, 7)), np.ones((5, 8)), np.ones((5, 8)))

    def test_bahdanau_attention_context_shape(self):
        rng = np.random.default_rng(3)
        attention = BahdanauAttention(decoder_hidden_size=6, annotation_size=8, rng=rng)
        decoder_hidden = rng.standard_normal((6, 1))
        annotations = [rng.standard_normal((8, 1)) for _ in range(4)]
        context, weights = attention.forward(decoder_hidden, annotations)

        self.assertEqual(context.shape, (8, 1))
        self.assertEqual(weights.shape, (4,))
        self.assertAlmostEqual(float(weights.sum()), 1.0, places=8)

    def test_bahdanau_attention_accepts_matrix_annotations(self):
        rng = np.random.default_rng(23)
        attention = BahdanauAttention(decoder_hidden_size=6, annotation_size=8, rng=rng)
        decoder_hidden = rng.standard_normal((6, 1))
        annotations = rng.standard_normal((4, 8))
        context, weights = attention.forward(decoder_hidden, annotations)

        self.assertEqual(context.shape, (8, 1))
        self.assertEqual(weights.shape, (4,))


class SequenceModelSmokeTests(unittest.TestCase):
    def test_vanilla_rnn_forward_backward(self):
        model = VanillaRNNLanguageModel(vocab_size=7, hidden_size=5, rng=np.random.default_rng(4))
        inputs = [0, 1, 2, 3]
        targets = [1, 2, 3, 4]
        xs, hs, _, ps = model.forward(inputs)
        loss = model.loss(ps, targets)
        grads = model.backward(xs, hs, ps, targets)

        self.assertGreater(loss, 0.0)
        self.assertEqual(grads["Wxh"].shape, model.Wxh.shape)
        self.assertEqual(grads["Whh"].shape, model.Whh.shape)
        self.assertEqual(grads["Why"].shape, model.Why.shape)

    def test_lstm_cell_shapes(self):
        cell = LSTMCell(input_size=4, hidden_size=6, rng=np.random.default_rng(5))
        x = np.random.default_rng(6).standard_normal((4, 1))
        h_prev = np.zeros((6, 1))
        c_prev = np.zeros((6, 1))
        h_next, c_next, cache = cell.forward(x, h_prev, c_prev)

        self.assertEqual(h_next.shape, (6, 1))
        self.assertEqual(c_next.shape, (6, 1))
        self.assertIn("f", cache)
        self.assertIn("i", cache)
        self.assertIn("o", cache)

    def test_lstm_sequence_model_outputs(self):
        rng = np.random.default_rng(7)
        model = LSTMSequenceModel(input_size=3, hidden_size=5, output_size=2, rng=rng)
        inputs = [rng.standard_normal((3, 1)) for _ in range(4)]
        y, h_states, c_states, gates = model.forward(inputs)

        self.assertEqual(y.shape, (2, 1))
        self.assertEqual(len(h_states), 4)
        self.assertEqual(len(c_states), 4)
        self.assertEqual(len(gates["f"]), 4)


class TransformerSmokeTests(unittest.TestCase):
    def test_positional_encoding_shape(self):
        encoding = positional_encoding(seq_len=6, d_model=8)
        self.assertEqual(encoding.shape, (6, 8))

    def test_transformer_block_forward(self):
        rng = np.random.default_rng(8)
        block = TransformerBlock(d_model=8, num_heads=2, d_ff=16, rng=rng)
        x = rng.standard_normal((5, 8))
        out, weights = block.forward(x, mask=create_causal_mask(5))

        self.assertEqual(out.shape, (5, 8))
        self.assertEqual(weights.shape, (2, 5, 5))

    def test_transformer_block_rejects_bad_width(self):
        block = TransformerBlock(d_model=8, num_heads=2, d_ff=16, rng=np.random.default_rng(24))
        with self.assertRaises(ValueError):
            block.forward(np.ones((5, 7)))


if __name__ == "__main__":
    unittest.main()
