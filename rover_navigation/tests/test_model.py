from rover_navigation.perception.model import MyRandLANet
import torch

def test_forward_gpu():
    if not torch.cuda.is_available():
        return
    
    B, N, d_in = 1, 512, 6

    model = MyRandLANet(d_in=6, num_classes=3, device="cuda").cuda()

    x = torch.randn(B,N,d_in).cuda() 

    out = model(x)
    assert out.shape == (1,3,512)