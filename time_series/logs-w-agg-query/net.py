import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange



class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU, drop=0.):
        super(Mlp, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class SAttention(nn.Module):
    def __init__(self, dim, num_heads=2, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super(SAttention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        #x = self.proj_drop(x)
        
        # x: (B, N, C)
        return x


class CAttention(nn.Module):
    def __init__(self, dim, num_heads=2, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super(CAttention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTEscale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, q, k, v, trans_mat):
        q_B, q_N, q_C = q.shape
        k_B, k_N, k_C = k.shape
        v_B, v_N, v_C = k.shape

        q = self.q(q).reshape(q_B, q_N, self.num_heads, q_C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k(k).reshape(k_B, k_N, self.num_heads, k_C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v(v).reshape(v_B, v_N, self.num_heads, v_C // self.num_heads).permute(0, 2, 1, 3)
        trans_mat = trans_mat.unsqueeze(1).repeat(1, self.num_heads, 1, 1)
        #import pdb;pdb.set_trace()
    

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn * trans_mat
        attn = torch.where(attn!=0, attn.softmax(dim=-1), torch.zeros_like(attn))

        x = (attn @ v).transpose(1, 2).reshape(q_B, q_N, v_C)
        x = self.proj(x)
        #x = self.proj_drop(x)
        
        # x: (B, N, C)
        return x


class G_SP(nn.Module):
    def __init__(self, dim):
        super(G_SP, self).__init__()
        self.dim = dim
        self.v = nn.Linear(dim, dim)
        #self.fc = nn.Linear(dim, dim)
        self.fc = Mlp(in_features = dim, out_features = dim, hidden_features=dim)
        self.activate = nn.Tanh()
        
    def forward(self, dec_embed, enc_embed, trans_mat):
        attn = (dec_embed @ enc_embed.transpose(-2, -1))
        attn = attn * trans_mat
        attn = torch.where(attn!=0, attn.softmax(dim=-1), torch.zeros_like(attn))

        v = self.v(enc_embed)
        v = torch.einsum('bnm, bmc -> bnc', attn, v)
        v = self.activate(v)
        
        #dec_embed = self.activate(dec_embed)

        gated_value = dec_embed * v
        out  = self.fc(gated_value + dec_embed)
        return out 

        


class STAR_CAttention(nn.Module):
    def __init__(self, dim, num_heads=2, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super(STAR_CAttention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTEscale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, q, k, v):
        k_B, k_N1, k_N2, k_C = k.shape
        v_B, v_N1, v_N2, v_C = v.shape
        q  = q.repeat(k_B, k_N1, 1, 1)
        q_B, q_N1, q_N2, q_C = q.shape


        q = self.q(q).reshape(q_B, q_N1, q_N2, self.num_heads, q_C // self.num_heads).permute(0, 1, 3, 2, 4)
        k = self.k(k).reshape(k_B, k_N1, k_N2, self.num_heads, k_C // self.num_heads).permute(0, 1, 3, 2, 4)
        v = self.v(v).reshape(v_B, v_N1, v_N2, self.num_heads, v_C // self.num_heads).permute(0, 1, 3, 2, 4)
    

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(2, 3).reshape(q_B, q_N1, q_N2, q_C)
        x = self.proj(x)
        #x = self.proj_drop(x)
        
        # x: (B, N, C)
        return x

class ENCODE(nn.Module):
    def __init__(self, dim, num_heads=2, mlp_ratio=1, qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super(ENCODE, self).__init__()
        self.norm1 = norm_layer(dim)
        self.attn = SAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

# to do 
class DECODE(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=1, qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super(DECODE, self).__init__()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.s_norm1 = norm_layer(dim)
        self.s_attn = SAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.s_norm2 = norm_layer(dim)
        self.s_mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)


        self.c_norm1 = norm_layer(dim)
        self.c_attn = CAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.c_norm2 = norm_layer(dim)
        self.c_mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, dec_inputs, enc_outputs, trans_mat):
        #dec_inputs = dec_inputs + self.s_attn(self.s_norm1(dec_inputs))
        #dec_inputs = dec_inputs + self.s_mlp(self.s_norm2(dec_inputs))
        
        x = self.c_attn(q=self.c_norm1(dec_inputs), 
                    k=self.c_norm1(enc_outputs), 
                    v=self.c_norm1(enc_outputs), trans_mat = trans_mat) + dec_inputs
        #x = x + self.c_mlp(self.c_norm2(x))
        x = self.c_mlp(self.c_norm2(x))

        return x


class STAR_Embed(nn.Module):
    def __init__(self, num_node, seq_len, embed_dim, num_q):
        super(STAR_Embed, self).__init__()
        self.embed_o = nn.Linear(seq_len, embed_dim)
        self.embed_d = nn.Linear(seq_len, embed_dim)
        self.induce = nn.Parameter(torch.randn(num_q, embed_dim), requires_grad=True)
        self.cattn = STAR_CAttention(embed_dim, num_heads=2)
        self.fc = nn.Linear(2*num_node*seq_len, embed_dim)

    def forward(self, data):
        #B, T, N ,N =data.shape
        #tmp_o= rearrange(data, 'b t o d -> b o (d t)')
        #tmp_d = rearrange(data, 'b t o d -> b d (o t)')
        #star_patch = torch.cat([tmp_o, tmp_d], dim=2)
        #x = self.fc(star_patch)

        tmp_o = self.embed_o(rearrange(data, 'b t o d -> b d o t'))
        tmp_d = self.embed_d(rearrange(data, 'b t o d -> b o d t'))
        star_patch = torch.cat([tmp_o, tmp_d], dim=2)
        #star_patch = tmp_d
        #patch_embedding = self.cattn(q = self.induce,
        #                            k = star_patch,
        #                            v = star_patch, )
        #x= torch.sum(patch_embedding, dim=2)
        x= torch.sum(star_patch, dim=2)
        return x

class PHEAD(nn.Module):
    def __init__(self, embed_dim, num_node):
        super(PHEAD, self).__init__()
        self.embed_dim = embed_dim
        self.num_node = num_node
        #self.o_mlp = Mlp(in_features = embed_dim, out_features = embed_dim, hidden_features=embed_dim*2)
        #self.d_mlp = Mlp(in_features = embed_dim, out_features = embed_dim, hidden_features=embed_dim*2)
        self.norm = nn.LayerNorm(embed_dim)
        self.mlp = Mlp(in_features = embed_dim, out_features = num_node, hidden_features=embed_dim)
    def forward(self, x):
        #trans_o = self.mlp(self.norm(x))
        #trans_o = F.softmax(trans_o, dim = -1)        
        #out  = torch.matmul(x, trans_o.permute(0, 2, 1))
        #trans_o = F.normalize(self.o_mlp(x), dim=-1)
        #trans_o = self.o_mlp(x)
        #trans_d = F.normalize(self.d_mlp(x), dim=-1)
        #trans_d = self.d_mlp(x)
        #out  = torch.matmul(trans_o, trans_d.permute(0, 2, 1))
        #out  = torch.matmul(x, trans_o.permute(0, 2, 1))
        #out = self.mlp(self.norm(x))
        out = self.mlp(x)
        out =  out.unsqueeze(1)
        return out


class Dec_Embed(nn.Module):
    def __init__(self, embed_dim, num_node, num_q):
        super(Dec_Embed, self).__init__()
        self.embed_dim = embed_dim
        self.num_node = num_node
        self.num_q = num_q

        self.time_conv = nn.Sequential(nn.Conv2d(6, 64, kernel_size=(1,1)),
                                        nn.ReLU(),
                                        nn.Conv2d(64, 32, kernel_size=(1,1)),
                                        nn.ReLU(),
                                        nn.Conv2d(32, 16, kernel_size=(1,1)),
                                       nn.Conv2d(16, 1, kernel_size=(1,1)))
        #self.weight_conv = nn.Sequential(nn.Conv2d(6, 64, kernel_size=(1,1)),
        #                                nn.ReLU(),
        #                                nn.Conv2d(64, 32, kernel_size=(1,1)),
        #                                nn.ReLU(),
        #                                nn.Conv2d(32, 16, kernel_size=(1,1)),
        #                                nn.Conv2d(16, 1, kernel_size=(1,1)))

        #self.time_conv = nn.Sequential(nn.Conv2d(6, embed_dim, kernel_size=(1,1)),
        #                                nn.ReLU(),
        #                                nn.Conv2d(embed_dim, embed_dim, kernel_size=(1,1)),
        #                                nn.ReLU(),
        #                                nn.Conv2d(embed_dim, embed_dim, kernel_size=(1,1)))

        self.mlp = Mlp(in_features = self.num_node, out_features = embed_dim, hidden_features=embed_dim)
        self.star_mlp= Mlp(in_features = self.embed_dim*2, out_features = embed_dim, hidden_features=embed_dim)
        #self.weight = nn.Parameter(torch.randn(self.num_node,  self.embed_dim), requires_grad=True)
        
        
        #self.mlp = Mlp(in_features = num_node, out_features = embed_dim, hidden_features=embed_dim//2)
    def forward(self, x):
        x = self.time_conv(x).squeeze(1)
        x_o = self.mlp(x)
        x_d = self.mlp(x.permute(0, 2, 1))
        star_patch = torch.cat([x_o, x_d], dim=2)
        out = self.star_mlp(star_patch)
        #x_o = self.mlp(tmp_o)
        #x_d = self.mlp(tmp_o.permute(0, 2, 1))
        #tmp_d = self.time_conv(rearrange(data, 'b t o d -> b o d t')).squeeze(-1)
        
        #star_patch = torch.cat([x_o, x_d], dim=2)
        #out= torch.sum(patch_embedding, dim=2)
        #weight = self.weight_conv(x).squeeze(1)
        #x = self.time_conv(x).squeeze(1).permute(0, 2, 3, 1)
        #tmp_w = torch.einsum('nk, mk -> nm', self.weight, self.weight)
        #b_s = x.shape[0]
        #weight = tmp_w.unsqueeze(-1).repeat(b_s, 1, 1, self.embed_dim)
        #x_o  = (weight * x).sum(-2)
        #x_d  = (weight * x).sum(-3)
        #o_d = torch.cat([x_o, x_d], dim=-1)
        #out = self.mlp(o_d)
        #x = self.mlp(x)
        #x_o  = (weight.unsqueeze(-1).repeat(1, 1, 1, self.embed_dim) * x_od).sum(-2)
        #x_d  = (weight.unsqueeze(-1).repeat(1, 1, 1, self.embed_dim) * x_od).sum(-3)
        #x_o = x_od.sum(dim=-1).permute(0, 2, 1)
        #x_d = x_od.sum(dim=-2).permute(0, 2, 1)
        #o_d = torch.cat([x_o, x_d], dim=-1)
        #out = self.mlp(o_d)
        #x_mean = torch.mean(x, dim=-1).permute(0, 2, 1)
        #x_sum = x.sum(dim=-1).permute(0, 2, 1)

        #x_max, _ = torch.max(x, dim=-1)
        #x_max = x_max.permute(0, 2, 1)

        #x_min, _ = torch.min(x, dim=-1)
        #x_min = x_min.permute(0, 2, 1)

        #x_med, _ = torch.median(x, dim=-1)
        #x_med = x_med.permute(0, 2, 1)

        #x_ = torch.cat([x_sum, x_mean, x_min, x_max, x_med], dim=-1)
        #x = self.mlp(x_)
        return out



class STFORMER(nn.Module):
    def __init__(self, cfg):
        super(STFORMER, self).__init__()
        self.device = cfg.DEVICE
        self.embed_dim = cfg.MODEL.EMBED_DIM
        self.num_head = cfg.MODEL.NUM_HEADS
        self.window = cfg.DATA.TIMESTEP
        self.horizon = cfg.DATA.HORIZON
        self.enc_num_node = cfg.MODEL.ENC_NUM_NODE
        self.dec_num_node = cfg.MODEL.DEC_NUM_NODE
        self.num_q = cfg.MODEL.NUM_QUERY
        self.enc_star_embed = STAR_Embed(self.enc_num_node, self.window, self.embed_dim, self.num_q)
        #self.enc_star_embed = Dec_Embed(self.embed_dim, self.enc_num_node, self.num_q)
        self.encode = ENCODE(dim=self.embed_dim, num_heads=self.num_head)
        self.decode = DECODE(dim=self.embed_dim, num_heads=self.num_head)
        #self.decode = G_SP(dim=self.embed_dim)
        #self.out_layer_dec = PHEAD(self.embed_dim)
        #self.out_layer_enc = PHEAD(self.embed_dim)
        self.out_layer_dec = PHEAD(self.embed_dim, self.dec_num_node)
        self.out_layer_enc = PHEAD(self.embed_dim, self.enc_num_node)
        #self.dec_embed = Dec_Embed(self.embed_dim, self.dec_num_node, self.num_q)
        #self.dec_embed = STAR_Embed(self.dec_num_node, self.window, self.embed_dim, self.num_q)
        #self.outconv = nn.Conv2d(7, 1, kernel_size=(1,1))
        self.outconv = nn.Sequential(nn.Conv2d(7, 64, kernel_size=(1,1)),
                                          nn.ReLU(),
                                        nn.Conv2d(64, 32, kernel_size=(1,1)),
                                        nn.ReLU(),
                                        nn.Conv2d(32, 16, kernel_size=(1,1)),
                                        nn.Conv2d(16, self.horizon, kernel_size=(1,1)))

        self.outconv_enc = nn.Sequential(nn.Conv2d(7, 64, kernel_size=(1,1)),
                                        nn.ReLU(inplace=True),
                                        nn.Conv2d(64, 32, kernel_size=(1,1)),
                                        nn.ReLU(inplace=True),
                                        nn.Conv2d(32, 16, kernel_size=(1,1)),
                                        nn.Conv2d(16, self.horizon, kernel_size=(1,1)))
        self.time_embedding  = nn.Embedding(24, self.embed_dim)
                                
        self.dec_embedding = nn.Parameter(torch.randn(self.dec_num_node, self.embed_dim), requires_grad=True)
        #self.poi_mlp =  nn.Sequential(
        #                            nn.Linear(cfg.MODEL.POIDIM, self.embed_dim),
        #                            nn.ReLU(),
        #                            nn.Linear(self.embed_dim, self.embed_dim),
        #                             )
        self.mse = nn.MSELoss(reduction="mean")
        self.to(self.device)
    def forward(self, batch_data):
        """
        :param batch_data:[b,t,o,d]
        :return:
        """
        enc_inputs = batch_data["enc_x"].to(self.device).float()
        dec_inputs = batch_data["dec_x"].to(self.device).float()
        target = batch_data["dec_y"].to(self.device).float()
        enc_y = batch_data["enc_y"].to(self.device).float()
        dec_mask = batch_data['dec_mask'].to(self.device)
        enc_mask = batch_data['enc_mask'].to(self.device)
        trans_mat = batch_data['trans_mat'].to(self.device).float()

        enc_time = batch_data['enc_time'].to(self.device)
        dec_time = batch_data['dec_time'].to(self.device)
        #import pdb;pdb.set_trace()
        
        #enc_poi = batch_data['enc_poi'].to(self.device)
        #dec_poi = batch_data['dec_poi'].to(self.device)
        #enc_poi_embed  = self.poi_mlp(enc_poi)
        #dec_poi_embed  = self.poi_mlp(dec_poi)
        
        b_1, t_1 = enc_inputs.shape[0], enc_inputs.shape[1]
        dec_embed = self.dec_embedding.repeat(b_1, 1, 1)
        dec_time_embedding = self.time_embedding(dec_time).unsqueeze(2).repeat(1, 1, self.dec_num_node, 1)
        #dec_embed = dec_embed + dec_poi_embed
        #dec_embed = self.dec_embed(dec_inputs)
        dec_embed = dec_embed + dec_time_embedding.squeeze(1)
        enc_embed = self.enc_star_embed(enc_inputs)
        #enc_time_embedding = self.time_embedding(enc_time).unsqueeze(2).repeat(1, 1, self.enc_num_node, 1)
        #enc_embed =  enc_embed +  enc_time_embedding
        #enc_embed = enc_embed + enc_poi_embed 

        #enc_outputs = self.encode(enc_embed)
        enc_outputs = enc_embed
        
        output = self.decode(dec_embed, enc_outputs, trans_mat)
        output = self.out_layer_dec(output) 
        output = self.outconv(torch.cat((dec_inputs, output),dim=1))


        #output = self.outconv(dec_inputs[:, -1:, :, :] + output)
        #output = self.outconv(dec_inputs)

        b, t= output.shape[0], output.shape[1]
        dec_mask = dec_mask.unsqueeze(1).repeat(1 ,t, 1, 1)
        output_mask = torch.masked_select(output, dec_mask)
        target_mask = torch.masked_select(target, dec_mask)
        loss_dec = self.mse(target_mask, output_mask)
        #loss_dec = self.mse(target, output)

        enc_out = self.out_layer_enc(enc_outputs)
        enc_out = self.outconv_enc(torch.cat((enc_inputs, enc_out),dim=1))
        #enc_out = self.outconv_enc(enc_inputs[:, -1:, :, :] + enc_out)
        #enc_out = self.outconv_enc(enc_out)
        enc_mask = enc_mask.unsqueeze(1).repeat(1 ,t, 1, 1)
        enc_output_mask = torch.masked_select(enc_out, enc_mask)
        enc_target_mask = torch.masked_select(enc_y, enc_mask)
        loss_enc = self.mse(enc_target_mask, enc_output_mask)


        mask = torch.gt(target, 0)
        output_mask = torch.masked_select(output, mask)
        target_mask = torch.masked_select(target, mask)
        dense_loss = self.mse(output_mask, target_mask)
        #loss = 0.33*loss_enc + 0.33*loss_dec + 0.33*dense_loss
        #loss = loss_enc + loss_dec + 0.1*dense_loss
        #loss = 0.3*loss_enc + loss_dec 
        loss = loss_enc + loss_dec 
        #loss = loss_dec

        #loss = self.mse(enc_y, enc_out)
        if self.training:
            return loss 
        else:
            return F.relu(output), target, loss
        

