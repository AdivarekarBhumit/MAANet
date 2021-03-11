import torch
import torch.nn as nn
import torch.nn.functional as F

class LocalAwareAttention(nn.Module):
    '''
        Local Aware Attention Module
    '''
    def __init__(self, kernel_size=4, stride=4, beta=0.07):
        super(LocalAwareAttention, self).__init__()
        # Module Network Parameters
        self.beta = beta
        self.kernel_size = kernel_size
        self.stride = stride
        self.scale_factor = kernel_size
        # Layers
        self.avg_pool = nn.AvgPool2d(kernel_size=self.kernel_size, stride=self.stride)
        self.upsample = nn.Upsample(scale_factor=self.scale_factor)
    
    def forward(self, x):
        # print('Input-Shape', x.shape)
        avg_pool = self.avg_pool(x)
        # print('AVG_Pool', avg_pool.shape)
        upsample = self.upsample(avg_pool)
        # print('UpSample2d', upsample.shape)

        sub_relu = self.beta * F.relu(torch.sub(x, upsample))
        # print('Sub Relu', sub_relu.shape)
        mul_op = torch.mul(sub_relu, x)
        # print('Mul OP', mul_op.shape)

        op = torch.add(x, mul_op)
        # print('OP', op.shape)

        return op

class MicroStructure(nn.Module):
    '''
        Building Block for LARD
    '''
    def __init__(self):
        super(MicroStructure, self).__init__()
        # Layers
        self.conv_initial = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.local_aware_attention = LocalAwareAttention(kernel_size=4, stride=4)
    
    def forward(self, x):
        input = x
        # print(x.shape)

        if x.shape[1] != 32:
            x = self.conv_initial(x)

        conv_la_1 = self.conv(x)
        conv_la_1 = self.local_aware_attention(conv_la_1)

        conn_1 = torch.add(conv_la_1, x)
        # print('1', conn_1.shape)

        conv_la_2 = self.conv(conn_1)
        conv_la_2 = self.local_aware_attention(conv_la_2)

        conn_2 = torch.add(x, torch.add(conn_1, conv_la_2))
        # print('2', conn_2.shape)

        conv_la_3 = self.conv(conn_2)
        conv_la_3 = self.local_aware_attention(conv_la_3)

        conn_3 = torch.add(x, torch.add(conn_1, torch.add(conn_2, conv_la_3)))
        # print('3', conn_3.shape)

        conv_4 = self.conv(conn_3)

        output = torch.add(x, conv_4)
        # print('OP', output.shape)

        return output/3 # Scaled down as contains 3 LA blocks

class LARD(nn.Module):
    '''
        LARD Block
    '''
    def __init__(self, n_microstructures=3):
        super(LARD, self).__init__()
        self.n_microstructures = n_microstructures
        self.micro_structure = MicroStructure()
        self.conv_final = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
    
    def forward(self, x):
        op = x
        for i in range(self.n_microstructures):
            op = self.micro_structure(op)
        if op.shape[1] != 64:
            op = self.conv_final(op)
        print(op.shape)

        op = torch.add(op, x)
        
        return op / 3

class DeepExtractionUnit(nn.Module):
    '''
        Deep Extraction Unit
    '''
    def __init__(self, n_blocks=4):
        super(DeepExtractionUnit, self).__init__()
        self.n_blocks = n_blocks
        self.shallow_extraction_unit = nn.Conv2d(3, 64, kernel_size=3, padding=1, stride=1)
        self.lard = LARD()
        self.last_conv = nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=1)
        self.upsample = nn.UpsamplingNearest2d(scale_factor=2)

    def forward(self, x):
        shallow_op = self.shallow_extraction_unit(x)
        op = shallow_op
        for i in range(self.n_blocks):
            op = self.lard(op)
        last_conv = self.last_conv(op)

        final_op = torch.add(shallow_op, last_conv)

        final_op = self.upsample(final_op)

        return final_op

class GlobalAwareAttention(nn.Module):
    '''
        Global Aware Attention Module
    '''
    def __init__(self):
        super(GlobalAwareAttention, self).__init__()
        self.conv0 = nn.Conv2d(64, 64, kernel_size=(256,256))
        self.conv1 = nn.Conv2d(64, 4, kernel_size=1)
        self.conv2 = nn.Conv2d(4,64, kernel_size=1)
    
    def forward(self, x):
        conv0 = self.conv0(x)
        conv1 = F.relu(self.conv1(conv0))
        conv2 = F.sigmoid(self.conv2(conv1))
        op = torch.mul(x, conv2)
        return op

class GlobalAttentionUnit(nn.Module):
    '''
        Global Attention Unit with GA Module and Conv Layers
    '''
    def __init__(self, n_ga_units=2):
        super(GlobalAttentionUnit, self).__init__()
        self.n_ga_units = n_ga_units
        self.conv = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.ga = GlobalAwareAttention()

        self.conv_final = nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1)

    
    def forward(self, x):
        ip = x
        for _ in range(self.n_ga_units):
            ip = self.conv(ip)
            ip = self.ga(ip)
        
        # Reconstruction Unit
        reconst_conv1 = self.conv(ip)
        final = self.conv_final(reconst_conv1)

        return final

class MAANet(nn.Module):
    '''
        Complete MAANet with all blocks
    '''
    def __init__(self):
        super(MAANet, self).__init__()
        self.deep_extraction_unit = DeepExtractionUnit()
        self.global_attention = GlobalAttentionUnit()
    
    def forward(self, x):
        x = self.deep_extraction_unit(x)
        x = self.global_attention(x)

        return x
