import pytest
import torch

from nystrom_attention import NystromAttention


def test_attention_is_independent_of_other_batch_items():
    torch.manual_seed(0)
    attn = NystromAttention(dim = 16, heads = 2, dim_head = 8, num_landmarks = 4,
                            pinv_iterations = 3, residual = False).double()
    x = torch.randn(1, 8, 16, dtype = torch.double, requires_grad = True)
    other = 12 * torch.randn_like(x)
    alone = attn(x)
    together = attn(torch.cat((x, other)))[:1]
    torch.testing.assert_close(together, alone)
    torch.testing.assert_close(torch.autograd.grad(together.sum(), x)[0],
                               torch.autograd.grad(alone.sum(), x)[0])


@pytest.mark.skipif(not torch.cuda.is_available(), reason = 'CUDA required')
@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
def test_attention_with_low_precision_weights(dtype):
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip('bfloat16 not supported')
    torch.manual_seed(0)
    attn = NystromAttention(dim = 16, heads = 2, dim_head = 8, num_landmarks = 4,
                            residual = False).cuda()
    x = torch.randn(1, 8, 16, device = 'cuda')
    expected = attn(x)
    actual = attn.to(dtype)(x.to(dtype))
    assert actual.dtype == dtype
    torch.testing.assert_close(actual.float(), expected, rtol = 0.03, atol = 0.01)
