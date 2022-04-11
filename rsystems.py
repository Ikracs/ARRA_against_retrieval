# coding: utf-8 
import os
import sys
import time
import base64
import argparse
import torch
import numpy as np
from PIL import Image

import myutils
from adv_loss import RRL, ARL
from black_box import ZOO, NES
from black_box import SignHunter as SH
from black_box import SquareAttack as SA

from pydoc import describe
from tkinter import image_names
from huaweicloudsdkcore.auth.credentials import BasicCredentials
from huaweicloudsdkimagesearch.v1 import *
from huaweicloudsdkimagesearch.v1.region.imagesearch_region import ImageSearchRegion

SUPPORT_FORMAT   = ['.jpg', '.jpeg', '.png', '.bmp']


class BingSearch(object):
    def __init__(self, cfg):
        self.N = cfg['N']   # len of visible ranking list

        self.MAXIMUM_TPS = 3
        self.BASE_URI    = 'https://api.bing.microsoft.com/v7.0/images/visualsearch'
        self.HEADERS     = {
            'Connection': 'close',
            'Ocp-Apim-Subscription-Key': 'YOUR_SUB_KEY',
            'Content-Type': 'multipart/form-data; boundary=ebf9f03029db4c2799ae16b5428b06bd'
        }
        
        self.idx2url = {}
        self.last_req = time.time()
    
    def _wait_for_next_query(self):
        elapsed = time.time() - self.last_req
        if elapsed < (1 / self.MAXIMUM_TPS):
            time.sleep((1 / self.MAXIMUM_TPS) - elapsed)
        self.last_req = time.time()
    
    def _construct_query(self, img_pth):
        image = open(img_pth, 'rb')
        return {'image' : ('myfile', image), 'mkt': 'cn-zh'}
    
    def download_img(self, value, save_root):
        try:
            idx = value['imageId']
            ext = '.' + value['encodingFormat']
            if ext.lower() in SUPPORT_FORMAT:
                url = value['contentUrl']
                header = {'Connection': 'close'}
                response = requests.get(url, timeout=50, headers=header)
                save_pth = os.path.join(save_root, idx + ext)
                if 'image' in response.headers["content-type"]:
                    with open(save_pth, 'wb') as f:
                        f.write(response.content)
                    return True
            return False
        except requests.exceptions.RequestException:
            return False
    
    def retrieval(self, img):
        query = self._construct_query(img)
        self._wait_for_next_query()
        response = requests.post(self.BASE_URI, headers=self.HEADERS, files=query)
        response.raise_for_status()
        meta_data = response.json()

        ranked_ids = []
        assert(meta_data['tags'][0]['displayName'] == '')
        for action in meta_data['tags'][0]['actions']:
            if action['actionType'] == 'VisualSearch':
                assert(action['data']['currentOffset'] == 0)
                values = action['data']['value']
                
                for value in values[:self.N]:
                    ranked_ids.append(value['imageId'])
                    if value['imageId'] not in self.idx2url.keys():
                        self.idx2url[value['imageId']] = value['contentUrl']
        return ranked_ids


class HuaweiCloudSearch(object):
    def __init__(self, cfg):
        self.N        = cfg['N']   # len of visible ranking list
        self.instance = cfg['dataset']  # instance name of gallery

        self.MAXIMUM_TPS = 6
        self.ACCESS_KEY = 'YOUR_ACCESS_KEY'
        self.SECRET_KEY = 'YOUR_SECRET_KEY'
        self.credentials = BasicCredentials(self.ACCESS_KEY, self.SECRET_KEY)
        self.client = ImageSearchClient.new_builder()\
            .with_credentials(self.credentials)\
            .with_region(ImageSearchRegion.value_of("cn-north-4"))\
            .build()

        self.last_req = time.time()
        
    def create_instance(self, instance, description=''):
        request = RunCreateInstanceRequest()
        request.body = CreateInstanceReq(
            name=instance, model='common-search',
            description=description, tags=None
        )
        self._wait_for_next_query()
        response = self.client.run_create_instance(request)
        print('Response: ', response)

        self.instance = instance
    
    def _wait_for_next_query(self):
        elapsed = time.time() - self.last_req
        if elapsed < (1 / self.MAXIMUM_TPS):
            time.sleep((1 / self.MAXIMUM_TPS) - elapsed)
        self.last_req = time.time()

    def encode_img(self, img_pth):
        with open(img_pth, "rb") as f:
            img = f.read()
        data = base64.b64encode(img)
        img_name = '_'.join(img_pth.split('/')[-2:])
        return data, img_name
    
    def delete_instance(self, instance):
        request = RunDeleteInstanceRequest()
        request.instance_name = instance
        self._wait_for_next_query()
        response = self.client.run_delete_instance(request)
        print('Response: ', response)
    
    def add_img(self, img):
        data, img_name = self.encode_img(img)
        request = RunAddPictureRequest()
        request.instance_name = self.instance
        request.body = AddPictureRequestReq(file=data, path=img_name)
        self._wait_for_next_query()
        response = self.client.run_add_picture(request)
        print('Response: ', response)
    
    def retrieval(self, img):
        data, img_name = self.encode_img(img)
        request = RunSearchPictureRequest()
        request.instance_name = self.instance
        request.body = SearchPictureReq(file=data, path=img_name, limit=self.N)
        self._wait_for_next_query()
        response = self.client.run_search_picture(request)
        ranked_list = [i['path'] for i in response.to_dict()['result']]
        return ranked_list

    def delete_img(self, img):
        data, img_name = self.encode_img(img)
        request = RunDeletePictureRequest()
        request.instance_name = self.instance
        request.body = DeletePictureReq(path=img_name)
        self._wait_for_next_query()
        response = self.client.run_delete_picture(request)
        print('Response: ', response)
