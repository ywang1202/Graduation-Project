# DenseFuse Network

import tensorflow as tf

from encoder import Encoder
from decoder import Decoder
from fusion_addition import Strategy
from fusion_wt import feature_wt

class DenseFuseNet(object):
    def __init__(self, model_pre_path):
        self.encoder = Encoder(model_pre_path)
        self.decoder = Decoder(model_pre_path)


    def transform_addition(self, img1, img2):
        # encode image
        enc_1 = self.encoder.encode(img1)
        enc_2 = self.encoder.encode(img2)
        # fuse feature maps
        self.target_features = Strategy(enc_1, enc_2)
        print('target_features:', self.target_features.shape)
        # decode target features back to image
        generated_img = self.decoder.decode(self.target_features)
        return generated_img


    def transform_wt(self, img1, img2):
        # encode image
        enc_1 = self.encoder.encode(img1)
        enc_2 = self.encoder.encode(img2)
        # fuse feature maps
        self.target_features = feature_wt(enc_1, enc_2)
        # decode image
        generated_img = self.decoder.decode(self.target_features)

        return generated_img


    def transform_recons(self, img):
        # encode image
        enc = self.encoder.encode(img)
        self.target_features = enc
        # decode image
        generated_img = self.decoder.decode(self.target_features)
        return generated_img


    # def transform_cbf(self, img1, img2):
    #     # encode image
    #     enc_1 = self.encoder.encode(img1)
    #     enc_2 = self.encoder.encode(img2)
    #     # fuse feature maps
    #     self.target_features = CBF_Strategy(enc_1, enc_2, sigmas=1.8, sigmar=25, ksize=11, cov_wsize=5)
    #     # decode image
    #     generated_img = self.decoder.decode(self.target_features)
    #
    #     return generated_img


    def transform_encoder(self, img):
        # encode image
        enc = self.encoder.encode(img)
        return enc


    def transform_decoder(self, feature):
        # decode image
        generated_img = self.decoder.decode(feature)
        return generated_img

