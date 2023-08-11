from __future__ import print_function, division
import torch
import numpy as np
import json
from torch.utils.data import Dataset, DataLoader
from topo_cell_seg.swinunet_transform.nptransform import ToTensor, RandomHorizontalFlip, RandomVerticalFlip

class TsnDataset(Dataset):
    """TissueNet DataSet object."""

    def __init__(self, root_dir, transform='default'):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        with open("{}/tis.json".format(root_dir),'r') as f:
            tis_dic = json.load(f)
        self.tis_dic = tis_dic
        self.paths = [k for k in tis_dic.keys()]
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # print("The index is {}".format(idx))
        # img_name = os.path.join(self.root_dir,
        #                         idx)
        img_name = "{}/X.npy".format(self.paths[idx])
        image = np.load(img_name)
        tislbl = self.tis_dic[self.paths[idx]]['tis']
        sample = {'image': image, 'tissuelabel': tislbl}

        if self.transform == 'default':
            sample = self.defaultrans(sample)
        elif self.transform:
            sample = self.transform(sample)

        return sample
    
    def defaultrans(self, sample):
        img, tslbl = sample['image'], sample['tissuelabel']
        img = RandomHorizontalFlip()(img)
        img = RandomVerticalFlip()(img)
        return {'image': ToTensor()(img), 'tissuelabel': torch.FloatTensor(tslbl)}
    

class InstanceSeg(Dataset):
    """TissueNet DataSet object."""

    def __init__(self, root_dir, transform='default', load_inseg = False):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        with open("{}/tis.json".format(root_dir),'r') as f:
            tis_dic = json.load(f)
        self.tis_dic = tis_dic
        self.paths = [k for k in tis_dic.keys()]
        self.root_dir = root_dir
        self.transform = transform
        self.load_inseg = load_inseg

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # print("The index is {}".format(idx))
        # img_name = os.path.join(self.root_dir,
        #                         idx)
        img_name = "{}/X.npy".format(self.paths[idx])
        image = np.load(img_name)
        tislbl = self.tis_dic[self.paths[idx]]['tis']
        
        cel_indnm, cel_semnm, nuc_indnm, nuc_semnm = '{}/cellinnd.npy'.format(self.paths[idx]), '{}/cellcls.npy'.format(self.paths[idx]), '{}/nucinnd.npy'.format(self.paths[idx]), '{}/nuccls.npy'.format(self.paths[idx])
        cel_ind, cel_sem, nuc_ind, nuc_sem = np.load(cel_indnm), np.load(cel_semnm), np.load(nuc_indnm), np.load(nuc_semnm)
        
        sample = {
            'image': image, 
            'tissuelabel': tislbl,
            'y': [cel_ind, cel_sem, nuc_ind, nuc_sem]
        }

        if self.transform == 'default':
            sample = self.defaultrans(sample)
        elif self.transform:
            sample = self.transform(sample)
            
        if self.load_inseg :
            insnm = '{}/y.npy'.format(self.paths[idx])
            inseg = np.load(insnm)
            sample['insnm'] = inseg

        return sample
    
    def defaultrans(self, sample):
        img, tslbl = sample['image'], sample['tissuelabel']
        def rdnpipl(imgi, randtrans = [RandomHorizontalFlip(), RandomVerticalFlip(), ToTensor()]):
            for randf in randtrans:
                imgi = randf(imgi)
            return imgi
        
        img = rdnpipl(img)
        newy = list()
        for yi in sample['y']:
            newy.append(
                rdnpipl(yi)
            )
        return {'image': img, 'tissuelabel': torch.FloatTensor(tslbl), 'y': newy}
    
class SeprtSeg(Dataset):
    """TissueNet DataSet for Seperated Segmentation object."""

    def __init__(self, root_dir, transform='default', load_inseg = False, segtyp = 'nuc'):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        with open("{}/newtis.json".format(root_dir),'r') as f:
            tis_dic = json.load(f)
        self.tis_dic = {"{}/{}".format(root_dir,k):v for k,v in tis_dic.items()}
        self.paths = ["{}/{}".format(root_dir, k) for k in tis_dic.keys()]
        self.root_dir = root_dir
        self.transform = transform
        self.load_inseg = load_inseg
        assert segtyp in ['nuc', 'cell'], 'segtype must be either nuc or cell'
        self.segtyp = segtyp

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # print("The index is {}".format(idx))
        # img_name = os.path.join(self.root_dir,
        #                         idx)
        img_name = "{}/X.npy".format(self.paths[idx])
        image = np.load(img_name)
        tislbl = self.tis_dic[self.paths[idx]]['tis']
        
        _indnm = '{}/{}innd.npy'.format(self.paths[idx], self.segtyp)
        _ind = np.load(_indnm)
        
        sample = {
            'image': image, 
            'tissuelabel': tislbl,
            'ind': _ind
        }
        
        if self.load_inseg :
            insnm = '{}/y.npy'.format(self.paths[idx])
            inseg = np.load(insnm)
            sample['insnm'] = inseg

        if self.transform == 'default':
            sample = self.defaultrans(sample)
        elif self.transform:
            sample = self.transform(sample)

        return sample
    
    def defaultrans(self, sample):
        img, _ind, tslbl = sample['image'], sample['ind'], sample['tissuelabel']
        def rdnpipl(imgi, randtrans = [RandomHorizontalFlip(), RandomVerticalFlip(), ToTensor()]):
            for randf in randtrans:
                imgi = randf(imgi)
            return imgi
        
        img = rdnpipl(img)
        newy = list()
        _ind = rdnpipl(_ind)
        if 'insnm' in sample:
            _insseg = rdnpipl(sample['insnm'])
            return {'image': img, 'tissuelabel': torch.FloatTensor(tslbl), 'ind': _ind, 'insnm':_insseg}
        
        return {'image': img, 'tissuelabel': torch.FloatTensor(tslbl), 'ind': _ind}
        
def main():
    traindt = InstanceSeg(root_dir = '../imgs/train')
    # print(traindt.__getitem__(0)['image'].shape)
    trainlder = DataLoader(
        traindt,
        batch_size=16,
        shuffle=True
    )
    # osample = traindt[0]
    # print("The zero sample is {}".format(osample))
    # print('The first data in the dataset has shape {} and {}'.format(imagi.shape, lbl.shape))
    for i_batch, sample_batched in enumerate(trainlder):
        print(i_batch, sample_batched['image'].size(), sample_batched['tissuelabel'].size(), sample_batched['y'][0].size(), sample_batched['y'][1].size())
        
if __name__ == '__main__':
    main()